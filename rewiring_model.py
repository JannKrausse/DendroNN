import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

import warnings

warnings.filterwarnings("ignore", module="pytorch_lightning")


class RewiringProgressCallback(Callback):
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # Log the current epoch and frozen units
        print(f"Max of longevity: {pl_module.longevity.max()}")
        print(f"Total number of frozen units: {pl_module.frozen_units.sum().item()}\n")


class RewiringModel(pl.LightningModule):
    """TODO: implement logic for multiple hidden layers"""

    def __init__(
        self,
        model,
        num_classes,
        target_num_frozen_units,
        lower_longevity_threshold=-9,
        upper_longevity_threshold=45,
        penalty_fac=1.02,
        experiment_name="",
        dataset=None,
        class_selectivity_threshold=0.3,
        selectivity_gap_to_other_classes=0.05,
        experiment_mode="std",
    ):
        super().__init__()
        self.model = model  # Your neural network
        self.num_classes = num_classes  # used to calculate longevity changes
        self.target_num_frozen_units = (
            target_num_frozen_units  # Target number of units to freeze
        )
        self.lower_longevity_threshold = (
            lower_longevity_threshold  # When to rewire units
        )
        self.upper_longevity_threshold = (
            upper_longevity_threshold  # When to freeze units
        )
        self.penalty_fac = penalty_fac  # Scaling between penalty and reward, should be slightly higher than 1
        self.experiment_name = experiment_name

        self.total_samples_processed = 0
        self.register_buffer("longevity", torch.zeros((self.model.num_hidden_units,)))
        self.register_buffer(
            "frozen_units", torch.zeros((self.model.num_hidden_units,)).to(bool)
        )

        self.register_buffer(
            "label_stats",
            torch.zeros((self.model.num_hidden_units, self.model.out_shape)),
        )  # keeps track of occurence of sequence for each class for each unit
        self.register_buffer(
            "label_counts", torch.zeros((self.model.out_shape,))
        )  # keeps track of occurence of each class
        self.register_buffer(
            "frozen_label_stats",
            torch.zeros((self.target_num_frozen_units, self.model.out_shape)),
        )
        self.register_buffer("data_stats", torch.zeros((self.model.out_shape,)))
        self.register_buffer(
            "num_selective_units_per_class", torch.zeros((self.num_classes,))
        )  # keeps track of number of frozen units per class
        self.register_buffer(
            "class_max_selectivity_history",
            torch.zeros((self.num_classes,)).to(torch.float32),
        )
        self.register_buffer(
            "actively_searched_classes", torch.ones((self.num_classes,)).to(bool)
        )

        self.units_per_class = self.model.num_hidden_units // self.num_classes
        self.target_units_per_class_div = int(
            self.target_num_frozen_units // self.num_classes
        )
        self.target_units_per_class = self.target_units_per_class_div * torch.ones(
            self.num_classes, dtype=torch.int
        )
        self.target_units_per_class[
            : self.target_num_frozen_units % self.num_classes
        ] += 1
        self.register_buffer(
            "units_to_classes_map",
            torch.zeros(self.target_units_per_class.max().item(), self.num_classes),
        )
        self.temp_units_to_classes_map = {}
        self.temp_class_selective_units = {}
        self.dataset = dataset
        self.class_num_frozen_units = None
        self.class_selectivity_threshold = class_selectivity_threshold
        self.frozen_label_stats_idx = None
        self.selectivity_gap_to_other_classes = selectivity_gap_to_other_classes
        self.selective_units = None
        self.experiment_mode = experiment_mode
        self.prev_selective_units = None  # used to mitigate the deletion of previously found selective units with in the case of a low pen-rew
        self.prev_class_selective_units = {}
        self.final_validation = False
        self.collect_validation_stats = False
        self.reset_idx = torch.tensor([])
        self.last_num_frozen_units = 0

        self.pos_longevity_change, self.neg_longevity_change = (
            self.calc_longevity_changes()
        )

    def calc_longevity_changes(self):
        pos_longevity_change = 1
        neg_longevity_change = 1 / (
            self.penalty_fac * self.actively_searched_classes.nonzero().shape[0]
        )

        return pos_longevity_change, neg_longevity_change

    def calc_longevity(self, spikes):
        activity_counts = (spikes.sum(dim=0) > 0).sum(
            dim=0
        )  # count of neurons that are active within a sample, value is the number of samples in which they were active
        non_activation_counts = self.total_samples_processed - activity_counts

        longevity_change = (self.pos_longevity_change * activity_counts) - (
            self.neg_longevity_change * non_activation_counts
        )

        self.longevity = (
            self.longevity + longevity_change
        )  # * self.class_learning_mask  # add longevity change to all units
        self.longevity = (
            self.longevity * ~self.frozen_units
        )  # reset longevity of frozen units to 0 (neutral state where no operation is applied)

        self.total_samples_processed = (
            0  # Reset sample counter after each longevity calculation
        )

    def rewire_and_freeze_units(self):
        rewire_mask = self.longevity < self.lower_longevity_threshold
        freeze_mask = self.longevity > self.upper_longevity_threshold

        # Rewire units
        self.model.hidden_layer["0"].rewire_synapses(
            rewire_mask
        )
        self.longevity[rewire_mask] = 0  # Reset longevity for rewired units
        self.longevity[freeze_mask] = (
            0  # Reset longevity for frozen units (debugging purposes)
        )

        # Draw new inter_spike_intervals for rewired units
        self.model.units["0"].set_new_inter_spike_intervals(
            delay_mask=rewire_mask,
            seq_len=self.model.hyperparameters["seq_len"],
            dataset=self.dataset,
            class_ids=self.actively_searched_classes,
        )
        # Draw new sequence_lengths for rewired units
        self.model.units["0"].set_new_num_spines(rewire_mask=rewire_mask)

        # Freeze units
        self.frozen_units = self.frozen_units | freeze_mask

    def forward(self, x):
        return self.model(x)

    def rewiring_step(self, batch):
        x, _ = batch  # Ignore labels (rewiring)
        with torch.no_grad():  # No gradients needed here
            _, spikes = self(x)
        return spikes

    def get_label_stats(self, spikes, labels):
        # Calculate statistics for the current batch
        activity_counts = spikes.sum(dim=0) > 0
        frozen_activity_counts = activity_counts * self.frozen_units

        self.label_stats.scatter_add_(
            1,  # Dimension to scatter along (columns for labels)
            labels.unsqueeze(0).expand(
                self.label_stats.size(0), -1
            ),  # Expand labels to match label_stats shape
            frozen_activity_counts.to(torch.float32).T,  # Transpose to match dimensions
        )
        self.label_counts.scatter_add_(
            0,  # Dimension to scatter along (columns for labels)
            labels,  # Expand labels to match label_stats shape
            torch.ones_like(labels).to(
                self.label_counts.dtype
            ),  # Transpose to match dimensions
        )

    def training_step(self, batch, batch_idx):
        # Perform random inference
        spikes = self.rewiring_step(batch)
        spikes = spikes[:, 0, :, :]  # Only for first hidden layer

        # set class_id
        if torch.all(batch[1] == batch[1][0]):
            self.class_id = batch[1][0].item()

        # Increment sample counter
        # The collate function may filter the raw DataLoader batch to active
        # classes, so count the samples that actually reached the model.
        self.total_samples_processed += spikes.shape[1]

        # calc longevity
        self.calc_longevity(spikes)

        # rewire and freeze units
        self.rewire_and_freeze_units()

        # check class stopping condition
        if (
            self.frozen_units.sum() - self.last_num_frozen_units
            > 2
            * self.actively_searched_classes.sum()
            * self.target_units_per_class.max()
        ):
            self.collect_validation_stats = True

        # Return dummy loss (not used in rewiring phase)
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def validation_step(self, batch, batch_idx):
        if not self.collect_validation_stats and not self.final_validation:
            return None

        # Perform random inference
        spikes = self.rewiring_step(batch)
        spikes = spikes[:, 0, :, :]

        # calc statistics
        self.get_label_stats(spikes, batch[1])

    def on_validation_epoch_end(self):
        if not self.collect_validation_stats and not self.final_validation:
            return

        # get the label statistics for the frozen units
        label_stats_frozen = self.label_stats[self.frozen_units, :]
        temp_frozen_label_stats_idx = self.frozen_units.nonzero()

        label_stats_frozen_norm = label_stats_frozen / self.label_counts.unsqueeze(
            0
        )  # .expand(label_stats_frozen.size(0), -1)  # normalize label stats by class counts
        label_stats_frozen_norm_sum = label_stats_frozen_norm.sum(dim=1, keepdim=True)
        nonzero_label_stats_mask = (
            label_stats_frozen_norm_sum > 0
        ).squeeze()  # this can happen if pen-rew is low and validation batch count is small
        self.frozen_label_stats = (
            label_stats_frozen_norm[nonzero_label_stats_mask, :]
            / label_stats_frozen_norm_sum[nonzero_label_stats_mask, :]
        )  # normalize label stats to sum to 1

        self.frozen_units[temp_frozen_label_stats_idx[~nonzero_label_stats_mask]] = (
            0  # filter out sequences that did not occur during validation
        )
        self.frozen_label_stats_idx = temp_frozen_label_stats_idx[
            nonzero_label_stats_mask
        ]

        self.evaluate_validation()

    def evaluate_validation(self):
        selective_units = torch.zeros_like(self.frozen_label_stats[:, 0]).to(
            bool
        )  # mask for good units that are selective for at least one class
        for class_id in self.actively_searched_classes.nonzero().squeeze(1):
            class_id = class_id.item()  # convert tensor to int
            class_selective_units, class_non_selective_units = (
                self.determine_selective_units(class_id)
            )  # determine selective units for the current class

            selective_units = (
                selective_units | class_selective_units
            )  # update mask for good units
            self.temp_class_selective_units[class_id] = (
                class_selective_units  # store selective units for the current class
            )

        non_selective_units = ~selective_units  # mask for non-selective units

        frozen_units_idx = self.frozen_units.nonzero().squeeze()
        non_selective_idx = frozen_units_idx[non_selective_units]

        # removal of non-selective units and storage of found selective units
        if self.prev_selective_units is not None:
            non_selective_idx_filtered = non_selective_idx[
                ~torch.isin(non_selective_idx, self.prev_selective_units)
            ]  # filter out units that were previously selectiv
        else:
            non_selective_idx_filtered = non_selective_idx
        all_selective_units = ~torch.isin(
            frozen_units_idx, non_selective_idx_filtered
        )  # == torch.isin(self.frozen_label_stats_idx, selective_idx)
        selective_idx = frozen_units_idx[
            all_selective_units
        ]  # get the indices of all previously and newly found selective units

        self.prev_selective_units = (
            selective_idx  # set previous selective units to current selective units
        )

        # removal of non-selective units and storage of found selective units for each class
        for class_id in self.actively_searched_classes.nonzero().squeeze(1):
            class_id = class_id.item()  # convert tensor to int
            class_non_selective_idx = frozen_units_idx[
                ~(self.temp_class_selective_units[class_id])
            ]
            if class_id not in self.prev_class_selective_units.keys():
                class_non_selective_idx_filtered = class_non_selective_idx
            else:
                class_non_selective_idx_filtered = class_non_selective_idx[
                    ~torch.isin(
                        class_non_selective_idx,
                        self.prev_class_selective_units[class_id],
                    )
                ]
            class_all_selective_units = ~torch.isin(
                frozen_units_idx, class_non_selective_idx_filtered
            )
            class_selective_idx = frozen_units_idx[
                class_all_selective_units
            ]  # get the indices of all previously and newly found selective units for the current class
            self.prev_class_selective_units[class_id] = (
                class_selective_idx  # store selective units for the current class
            )

            self.temp_class_selective_units[class_id] = torch.isin(
                frozen_units_idx, class_selective_idx
            )  # store selective units for the current class as a mask

            self.num_selective_units_per_class[class_id] = (
                self.prev_class_selective_units[class_id].shape[0]
            )

        self.frozen_units[non_selective_idx_filtered] = 0  # reset non-selective units
        self.longevity[non_selective_idx_filtered] = -float(
            "inf"
        )  # implicitly force rewiring of non-selective units

        for class_id in self.actively_searched_classes.nonzero().squeeze(1):
            class_id = class_id.item()  # convert tensor to int

            max_selectivity = self.frozen_label_stats[:, class_id].max().item()
            if max_selectivity > self.class_max_selectivity_history[class_id]:
                self.class_max_selectivity_history[class_id] = max_selectivity

            print(
                f"\nNumber of selective units for class {class_id}: {int(self.num_selective_units_per_class[class_id])}. Maximum selectivity: {max_selectivity}. Maximum ever: {self.class_max_selectivity_history[class_id]}."
            )

            if (
                self.num_selective_units_per_class[class_id]
                >= self.target_units_per_class[class_id]
            ):
                self.finalize_class(class_id)
        print(
            f"Classes that are still active: {self.actively_searched_classes.nonzero().squeeze(1).tolist()}."
        )

        # reset units that have not been selected for any class
        self.reset_idx = self.reset_idx.to(self.device)
        filtered_reset_idx_mask = torch.ones_like(self.reset_idx.squeeze()).to(
            torch.bool
        )  # mask for reset indices that are not already frozen for another class
        for class_id in (~self.actively_searched_classes).nonzero().squeeze(1):
            class_id = class_id.item()  # convert tensor to int
            filtered_reset_idx_mask *= ~torch.isin(
                self.reset_idx.squeeze(), self.units_to_classes_map[:, class_id]
            )  # filter out units that are already frozen for another class
        filtered_reset_idx = self.reset_idx[filtered_reset_idx_mask].to(
            torch.int
        )  # get the indices of all units that are not already frozen for another class
        self.frozen_units[filtered_reset_idx] = (
            0  # reset frozen units that are not already frozen for another class
        )
        self.longevity[filtered_reset_idx] = -float("inf")  # implicitly force
        self.reset_idx = torch.tensor([], device=self.device)

        self.collect_validation_stats = False

        # check if all classes have been processed update frozen_label_stats to final frozen units
        if self.actively_searched_classes.sum() == 0:
            # make sure there are no leftover frozen units that are not assigned to a class
            self.frozen_units[
                self.frozen_units.nonzero()[
                    (
                        ~torch.isin(
                            self.frozen_units.nonzero(), self.units_to_classes_map
                        )
                    ).squeeze()
                ]
            ] = 0

            print("\nAll classes have been processed. Stopping training.\n")
            self.frozen_label_stats = (
                self.label_stats[self.frozen_units, :] / self.label_counts.unsqueeze(0)
            ) / (
                self.label_stats[self.frozen_units, :] / self.label_counts.unsqueeze(0)
            ).max(
                dim=1, keepdim=True
            )[
                0
            ]
        else:
            # recalculate longevity changes based on the classes that are still actively searched
            self.pos_longevity_change, self.neg_longevity_change = (
                self.calc_longevity_changes()
            )

        # reset label stats and counts for next validation epoch
        self.label_stats.zero_()
        self.label_counts.zero_()

        self.last_num_frozen_units = (
            self.frozen_units.sum()
        )  # store number of frozen units for next validation step

    def finalize_class(self, class_id):
        self.actively_searched_classes[class_id] = (
            0  # disable any evaluation for this class in the future
        )
        if self.experiment_mode == "specific_selectivity":
            descending = False
        else:
            descending = True
        selectivity_order = torch.argsort(
            self.frozen_label_stats[
                self.temp_class_selective_units[class_id], class_id
            ],
            descending=descending,
        )
        ordered_class_frozen_idx = self.frozen_label_stats_idx[
            self.temp_class_selective_units[class_id]
        ][
            selectivity_order
        ]  # order idx by selectivity
        class_remove_idx = ordered_class_frozen_idx[
            self.target_units_per_class[class_id] :
        ]  # remove lowest selectivity units that exceed target number of units
        if self.target_units_per_class[class_id] == self.target_units_per_class[0]:
            self.units_to_classes_map[:, class_id] = ordered_class_frozen_idx[
                : self.target_units_per_class[class_id]
            ].squeeze()
        elif (
            self.target_units_per_class[class_id] > 0
        ):  # this only happens when there are less target units than classes, then give unit frozen for class 0 as placeholder
            self.units_to_classes_map[:, class_id] = torch.cat(
                [
                    ordered_class_frozen_idx[
                        : self.target_units_per_class[class_id]
                    ].squeeze(-1),
                    self.units_to_classes_map[0, 0].unsqueeze(0),
                ]
            )
        else:
            self.units_to_classes_map[:, class_id] = self.units_to_classes_map[:, 0]

        kept_units = self.temp_class_selective_units[class_id].nonzero()[
            selectivity_order
        ][: self.target_units_per_class[class_id]]

        for c in range(self.num_classes - class_id - 1):
            target_class = class_id + c + 1

            target_tensor = self.temp_class_selective_units[target_class]

            # Ensure kept_units are within bounds
            if kept_units.numel() > 0:
                valid_mask = (kept_units.squeeze() >= 0) & (
                    kept_units.squeeze() < target_tensor.shape[0]
                )
                valid_kept_units = kept_units[valid_mask]

                if valid_kept_units.numel() > 0:
                    try:
                        self.temp_class_selective_units[target_class][
                            valid_kept_units.squeeze()
                        ] = 0
                        if (
                            self.temp_class_selective_units[target_class].sum()
                            < self.target_units_per_class[target_class]
                        ):
                            self.num_selective_units_per_class[target_class] = (
                                self.temp_class_selective_units[target_class]
                                .sum()
                                .item()
                            )
                    except RuntimeError as e:
                        print(f"ERROR in class {target_class}:")
                        print(f"target_tensor shape: {target_tensor.shape}")
                        print(f"valid_kept_units: {valid_kept_units}")
                        print(f"valid_kept_units shape: {valid_kept_units.shape}")
                        print(f"Error: {e}")
                        raise e

        self.reset_idx = torch.cat(
            (self.reset_idx.to(class_remove_idx.device), class_remove_idx)
        )

        self.trainer.should_stop = True

    def determine_selective_units(self, class_id):
        # calculating selectivities
        if class_id is not None:
            if self.experiment_mode == "std":
                selective_units = (
                    self.frozen_label_stats[:, class_id]
                    > self.class_selectivity_threshold
                )  # this value could be anything > chance, 0.5 allows for units that are selective for more than one class
            elif self.experiment_mode == "selectivity_gap":
                selective_units_1 = (
                    self.frozen_label_stats[:, class_id]
                    > self.class_selectivity_threshold
                )  # this value could be anything > chance, 0.5 allows for units that are selective for more than one class
                other_classes_frozen_label_stats = (
                    self.frozen_label_stats.clone()
                )  # get max of frozen label stats for all other classes
                other_classes_frozen_label_stats[:, class_id] = 0
                selective_units_2 = self.frozen_label_stats[:, class_id] > (
                    other_classes_frozen_label_stats.max(dim=1)[0]
                    + self.selectivity_gap_to_other_classes
                )  # this value could be anything > chance, 0.5 allows for units that are selective for more than one class
                selective_units = (
                    selective_units_1 & selective_units_2
                )  # only keep units that are selective for the current class and not for any other class
            elif self.experiment_mode == "specific_selectivity":
                wiggle_room = 0.02  # width of interval of possible selectivity values
                selective_units_1 = (
                    self.frozen_label_stats[:, class_id]
                    > self.class_selectivity_threshold
                )
                selective_units_2 = (
                    self.frozen_label_stats[:, class_id]
                    < self.class_selectivity_threshold + wiggle_room
                )
                other_classes_frozen_label_stats = (
                    self.frozen_label_stats.clone()
                )  # get max of frozen label stats for all other classes
                other_classes_frozen_label_stats[:, class_id] = 0
                selective_units_3 = self.frozen_label_stats[:, class_id] > (
                    other_classes_frozen_label_stats.max(dim=1)[0] - wiggle_room
                )
                selective_units = (
                    selective_units_1 & selective_units_2 & selective_units_3
                )  # only keep units that have selectivity in a specific range for the current class and are not more selective for any other class
            else:
                raise ValueError(
                    f"Experiment mode {self.experiment_mode} is not supported."
                )
        else:
            if self.experiment_mode == "std":
                selective_units = (
                    self.frozen_label_stats > self.class_selectivity_threshold
                ).sum(
                    dim=1
                ) > 0  # this value could be anything > chance, 0.5 allows for units that are selective for more than one class
            else:
                raise ValueError(
                    f"Experiment mode {self.experiment_mode} is not supported."
                )
        non_selective_units = ~selective_units

        return selective_units, non_selective_units

    def configure_optimizers(self):
        # Optimizer not required since no backpropagation in this phase
        return None

    def set_specific_sequence(
        self, sequence: torch.Tensor, sequence_length: int, delays: torch.Tensor
    ):
        rewire_mask = torch.ones(
            self.model.num_hidden_units, dtype=torch.bool, device=self.model.device
        )
        self.longevity[rewire_mask] = 0.7 * self.upper_longevity_threshold

        self.model.units["0"].sequence_len_dist_rep[rewire_mask] = (
            sequence_length * torch.ones((rewire_mask.sum(),))
        ).to(self.model.units["0"].sequence_len_dist_rep.device, torch.int64)

        self.model.units["0"].inter_spike_intervals[rewire_mask] = delays.repeat(
            (rewire_mask.sum(), 1)
        ).to(self.model.units["0"].inter_spike_intervals.device)
        self.model.units["0"].updates_per_time_step[rewire_mask] = torch.clamp(
            (self.model.units["0"].num_parallel_seq - 1)
            / self.model.units["0"].inter_spike_intervals[rewire_mask],
            max=self.model.units["0"].inter_spike_intervals.max().item() + 1,
        )

        sequence_indices = sequence.nonzero(as_tuple=True)[0]
        if sequence_indices.numel() != 1:
            raise ValueError("A specific sequence must contain one active synapse.")
        self.model.hidden_layer["0"].synapse_indices[
            rewire_mask.repeat_interleave(
                self.model.hidden_layer["0"].spines_per_output_channel
            )
        ] = sequence_indices.to(self.model.device).repeat(
            rewire_mask.sum() * self.model.hidden_layer["0"].spines_per_output_channel
        )
