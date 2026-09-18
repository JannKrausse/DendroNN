import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")


class DendroNNUnit(nn.Module):
    def __init__(
        self,
        num_spines,  # max length of sequence
        num_units,
        max_seq_len,
        inter_spike_intervals,
        spike_acceptance_window,
        batch_size,
        min_num_spines=None,
        parallel_seqs=True,
    ):
        super().__init__()
        self.num_spines = num_spines
        self.min_num_spines = min_num_spines
        self.num_units = num_units
        self.max_seq_len = max_seq_len
        self.register_buffer("inter_spike_intervals", inter_spike_intervals)
        self.spike_acceptance_window = spike_acceptance_window
        self.batch_size = batch_size
        self.parallel_seqs = parallel_seqs

        if self.min_num_spines is None:
            self.min_num_spines = self.num_spines
        assert (
            self.min_num_spines <= self.num_spines
        ), "Minimum sequence length must be smaller than or equal to the maximum sequence length."
        self.sequence_len_probs = (
            0.3 ** torch.arange(self.num_spines - self.min_num_spines + 1)
        ).flip(0)
        sequence_len_dist = self.min_num_spines + torch.multinomial(
            self.sequence_len_probs, num_samples=self.num_units, replacement=True
        )
        self.register_buffer("sequence_len_dist_rep", sequence_len_dist)

        num_parallel_seq = torch.max(inter_spike_intervals).item() + 1
        self.num_parallel_seq = num_parallel_seq
        self.spine_memory_size = self.num_parallel_seq + self.spike_acceptance_window

        self.register_buffer(
            "expected_spikes",
            torch.zeros(
                (
                    self.batch_size,
                    self.num_units,
                    max(self.num_spines - 1, 1),
                    self.spine_memory_size,
                )
            ),
        )

        self.register_buffer(
            "expected_spikes_update_mask",
            torch.cat(
                (
                    torch.ones(
                        (
                            self.batch_size,
                            self.num_units,
                            max(self.num_spines - 1, 1),
                            self.spine_memory_size - 1,
                        )
                    ),
                    torch.zeros(
                        (
                            self.batch_size,
                            self.num_units,
                            max(self.num_spines - 1, 1),
                            1,
                        )
                    ),
                ),
                dim=-1,
            ),
        )
        self.register_buffer(
            "expecting_spines_padding", torch.ones((self.batch_size, self.num_units, 1))
        )
        batch_idx = (
            torch.arange(self.batch_size)
            .repeat_interleave(self.num_units * max(self.num_spines - 1, 1))
            .unsqueeze(-1)
        )
        unit_idx = (
            torch.arange(self.num_units)
            .repeat(self.batch_size)
            .repeat_interleave((max(self.num_spines - 1, 1)))
            .unsqueeze(-1)
        )
        spine_idx = (
            torch.arange(max(self.num_spines - 1, 1))
            .repeat(self.batch_size * self.num_units)
            .unsqueeze(-1)
        )
        self.register_buffer(
            "index_put_idx_other_dims",
            torch.cat((batch_idx, unit_idx, spine_idx), dim=-1),
        )

    def forward(self, x):
        expecting_spines = torch.cat(
            (
                self.expecting_spines_padding,
                torch.any(
                    self.expected_spikes[:, :, :, : (self.spike_acceptance_window + 1)]
                    != 0,
                    dim=-1,
                ),
            ),
            dim=2,
        )  # first spine is always expecting, TODO: change way of processing for spike_acceptance_window in case of trainable_spines=True

        accepted_spikes = (
            x * expecting_spines
        )  # only accept spikes if the spine is expecting one

        accepted_spikes[:, :, :-1] = self.parallel_seqs * accepted_spikes[:, :, :-1] + (
            1 - self.parallel_seqs
        ) * accepted_spikes[:, :, :-1] * (
            1
            - (
                self.expected_spikes[:, :, :, (self.spike_acceptance_window + 1) :].sum(
                    dim=-1
                )
                > 0
            ).to(torch.float)
        )

        self.expected_spikes = (
            torch.roll(self.expected_spikes, shifts=-1, dims=-1)
            * self.expected_spikes_update_mask
        )

        # insert accepted_spikes into expected_spikes at correct positions given by self.inter_spike_intervals
        accepted_spikes_mask = accepted_spikes[:, :, :-1].flatten() != 0
        index_put_values = accepted_spikes[:, :, :-1].flatten()[accepted_spikes_mask]
        index_put_idx = torch.cat(
            (
                self.index_put_idx_other_dims,
                self.inter_spike_intervals.flatten()
                .repeat(self.batch_size)
                .unsqueeze(-1),
            ),
            dim=-1,
        )[accepted_spikes_mask]
        index_put_idx_tuple = tuple(index_put_idx.unbind(dim=-1))
        self.expected_spikes.index_put_(index_put_idx_tuple, index_put_values)

        activations = torch.gather(
            accepted_spikes,
            dim=2,
            index=(
                self.sequence_len_dist_rep.to(x.device)
                .repeat((self.batch_size, 1))
                .unsqueeze(dim=-1)
                - 1
            ),
        ).squeeze(
            dim=-1
        )  # output is determined by last spine
        return activations

    def reset(self, device):
        self.expected_spikes.data = torch.zeros(
            (
                self.batch_size,
                self.num_units,
                max(self.num_spines - 1, 1),
                self.spine_memory_size,
            )
        ).to(device)

        self.expected_spikes_update_mask.data = torch.cat(
            (
                torch.ones(
                    (
                        self.batch_size,
                        self.num_units,
                        max(self.num_spines - 1, 1),
                        self.spine_memory_size - 1,
                    )
                ),
                torch.zeros(
                    (self.batch_size, self.num_units, max(self.num_spines - 1, 1), 1)
                ),
            ),
            dim=-1,
        ).to(device)
        self.expecting_spines_padding.data = torch.ones(
            (self.batch_size, self.num_units, 1)
        ).to(device)

        batch_idx = (
            torch.arange(self.batch_size)
            .repeat_interleave(self.num_units * max(self.num_spines - 1, 1))
            .unsqueeze(-1)
        )
        unit_idx = (
            torch.arange(self.num_units)
            .repeat(self.batch_size)
            .repeat_interleave((max(self.num_spines - 1, 1)))
            .unsqueeze(-1)
        )
        spine_idx = (
            torch.arange(max(self.num_spines - 1, 1))
            .repeat(self.batch_size * self.num_units)
            .unsqueeze(-1)
        )
        self.index_put_idx_other_dims.data = torch.cat(
            (batch_idx, unit_idx, spine_idx), dim=-1
        ).to(device)

    def set_new_inter_spike_intervals(self, delay_mask, seq_len, dataset, class_ids):
        if dataset == "SHD":
            relevant_timescale_for_class_list = [
                120,
                105,
                81,
                107,
                104,
                105,
                46,
                109,
                63,
                111,
                101,
                88,
                96,
                115,
                102,
                85,
                63,
                146,
                48,
                102,
            ]
            relevant_timescale_for_class = max(
                [relevant_timescale_for_class_list[i] for i in class_ids.tolist()]
            )  # take the max of all classes that are currently searched
            relevant_timescale_for_class = int(
                relevant_timescale_for_class * (seq_len / 292) * 3 / 4
            )
        elif dataset == "sMNIST":  # sMNIST
            relevant_timescale_for_class = 600
        elif dataset == "p-sMNIST":  # sMNIST
            relevant_timescale_for_class = 28 * 28
        elif dataset == "NeuroMorse":
            relevant_timescale_for_class = 100
        else:
            raise ValueError(f"Dataset {dataset} not supported.")
        max_delay = max(
            int(relevant_timescale_for_class / (max(self.num_spines - 1, 1))), 1
        )
        self.inter_spike_intervals[delay_mask] = torch.randint(
            low=0, high=max_delay, size=self.inter_spike_intervals[delay_mask].shape
        ).to(
            self.inter_spike_intervals.device
        )  # max_delay so that the total seq_len distributes over all participating synapses

    def set_new_num_spines(self, rewire_mask):
        if rewire_mask.sum() > 0:
            sequence_len_dist = self.min_num_spines + torch.multinomial(
                self.sequence_len_probs, num_samples=rewire_mask.sum(), replacement=True
            )
            self.sequence_len_dist_rep[rewire_mask] = sequence_len_dist.to(
                self.sequence_len_dist_rep.device
            )
