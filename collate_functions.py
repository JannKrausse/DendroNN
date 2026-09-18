import random
import warnings

import torch
import numpy as np


class CollateFN:
    def __init__(
        self,
        slice_amounts=False,
        num_slices=7,
        slicing_thrs=None,
        slicing_thrs_maxs=None,
        include_zero_slice=False,
        jitter_std=0.0,
        dropout_prob=0.0,
        poisson_lambda=0.0,
        padding_len=400,
        permute_data=False,
        permutation_seed=None,
    ):
        self.slice_amounts = slice_amounts
        if self.slice_amounts:
            self.data_max = None
            self.num_slices = num_slices
            self.slicing_thrs = slicing_thrs  # starting points for each slice, RELATIVE AMOUNTS!, np.inf will be attached during inference
            self.slicing_thrs_maxs = slicing_thrs_maxs
            self.include_zero_slice = include_zero_slice
            if self.slicing_thrs is not None:
                if isinstance(self.slicing_thrs, list):
                    self.slicing_thrs = np.array(self.slicing_thrs)
                if not isinstance(self.slicing_thrs, np.ndarray):
                    raise TypeError("slicing_thrs should be a numpy array.")

        if jitter_std == "none":  # in NeuorMorse: low=1, high=2
            self.jitter_std = 0
        elif jitter_std == "low":
            self.jitter_std = 1
        elif jitter_std == "high":
            self.jitter_std = 2
        else:
            self.jitter_std = jitter_std
        self.jitter_axis = 1  # before permutation, time axis is at dim 1
        if dropout_prob == "none":  # in NeuorMorse: low=0.0333, high=0.0666
            self.dropout_prob = 0
        elif dropout_prob == "low":
            self.dropout_prob = 0.0333
        elif dropout_prob == "high":
            self.dropout_prob = 0.0666
        else:
            self.dropout_prob = dropout_prob
        if (
            poisson_lambda == "none"
        ):  # in NeuorMorse: low=0.05, high=0.1 --> translates to low=7.5, high=14.2 (different implementations)
            self.poisson_lambda = 0
        elif poisson_lambda == "low":
            self.poisson_lambda = 7.5
        elif poisson_lambda == "high":
            self.poisson_lambda = 14.2
        else:
            self.poisson_lambda = poisson_lambda

        self.padding_len = padding_len
        self.permute_data = permute_data
        self.permutation_seed = permutation_seed
        if self.permute_data:
            self.permutation_generator = torch.Generator()
            self.permutation_generator.manual_seed(self.permutation_seed)
            self.permutation_idx = torch.randperm(
                padding_len, generator=self.permutation_generator
            )

    def __call__(self, batch):
        """
        Used to make sure sequence length comes as the first dimension.
        """
        input_data, targets = list(zip(*batch))  # shape: (batch_size, seq_len, ...)
        if isinstance(targets[0], int):  # sMNIST
            input_data, targets = list(zip(*batch))
            input_data = torch.stack(input_data)
            input_flat = input_data.flatten(
                start_dim=2
            )  # same as torch.Tensor.view(-1, 1) as often done for sMNIST
            input_data = input_flat.permute(
                0, 2, 1
            )  # permute to (batch_size, seq_len, in_shape), later reshaped to (seq_len, batch_size, in_shape)
            targets = torch.tensor(targets)
        elif isinstance(input_data[0], torch.Tensor):  # NeuroMorse, Braille
            input_data = torch.stack(input_data)
            targets = torch.stack(targets)
        else:  # Tonic datasets
            try:
                input_data = torch.tensor(np.asarray(input_data))
            except ValueError:  # inputs need to be padded
                batch_padding_len = max(
                    self.padding_len, max(arr.shape[0] for arr in input_data)
                )
                input_data = tuple(
                    np.pad(
                        arr,
                        pad_width=[(0, batch_padding_len - arr.shape[0])]
                        + [(0, 0)] * (arr.ndim - 1),
                        mode="constant",
                    )
                    for arr in input_data
                )
                input_data = torch.tensor(np.asarray(input_data)).squeeze()
            targets = torch.tensor(np.asarray(targets))

        if input_data.dim() > 3:
            input_data = torch.flatten(input_data, start_dim=2, end_dim=-1)

        if self.slice_amounts:
            if input_data.shape[0] < 128:
                warnings.warn(
                    "Batch size is less than 128, data_max might not be representative. Consider increasing the batch size."
                )
            if self.data_max is None:
                self.data_max = input_data.max()
                if self.slicing_thrs is None:
                    self.slicing_thrs = np.linspace(
                        0, self.data_max / 4, self.num_slices
                    )
                else:
                    if self.slicing_thrs_maxs is None:
                        self.slicing_thrs = self.slicing_thrs * self.data_max.item()
                if self.slicing_thrs_maxs is None:
                    self.slicing_thrs_maxs = np.append(self.slicing_thrs[1:], np.inf)

            if not self.include_zero_slice:
                sliced_data = torch.zeros(
                    (
                        input_data.shape[0],
                        input_data.shape[1],
                        input_data.shape[2] * self.num_slices,
                    )
                )
            else:
                sliced_data = torch.zeros(
                    (
                        input_data.shape[0],
                        input_data.shape[1],
                        input_data.shape[2] * (self.num_slices + 1),
                    )
                )
                sliced_data[
                    :,
                    :,
                    input_data.shape[2]
                    * self.num_slices : input_data.shape[2]
                    * (self.num_slices + 1),
                ] = (
                    input_data == 0
                )
            for i in range(self.num_slices):
                sliced_data[
                    :, :, input_data.shape[2] * i : input_data.shape[2] * (i + 1)
                ] = (input_data > self.slicing_thrs[i]) * (
                    input_data < self.slicing_thrs_maxs[i]
                )  # * (input_data > 0)
            input_data = sliced_data

        # apply custom functions altered by inheriting classes
        input_data, targets = self.additional_stuff(input_data, targets)

        # apply noise
        if self.jitter_std > 0 or self.dropout_prob > 0 or self.poisson_lambda > 0:
            input_data = self.apply_noise(input_data)

        # input_data = input_data.squeeze().permute(1, 0, 2).float()  # shape: (seq_len, batch_size, ...)
        input_data = input_data.permute(
            1, 0, 2
        ).float()  # shape: (seq_len, batch_size, ...)

        # permute data if required, e.g., for permuted sMNIST (psMNIST)
        if self.permute_data:
            input_data = input_data[self.permutation_idx, :, :]

        return input_data, targets

    def additional_stuff(self, data, targets):
        return data, targets

    def apply_noise(self, data):
        # Find indices of 1s
        ones = (data == 1).nonzero(as_tuple=False)
        new_ones = ones.clone()

        # Jitter: every 1 is jittered, amount drawn from normal distribution
        if self.jitter_std > 0:
            jitter_amount = torch.normal(
                mean=0.0, std=self.jitter_std, size=(len(ones),)
            )
            jitter_amount = jitter_amount.round().long()
            new_ones[:, self.jitter_axis] += jitter_amount
            # Clamp to valid range
            new_ones[:, self.jitter_axis] = new_ones[:, self.jitter_axis].clamp(
                0, data.shape[self.jitter_axis] - 1
            )

        # Dropout
        if self.dropout_prob > 0:
            keep_mask = torch.rand(len(new_ones)) > self.dropout_prob
            new_ones = new_ones[keep_mask]

        # Create new tensor and set 1s
        out = torch.zeros_like(data)
        out[tuple(new_ones.t())] = 1

        # Poissonian noise
        if self.poisson_lambda > 0:
            for i in range(data.shape[0]):  # for every sample
                num_noise = (
                    torch.poisson(
                        torch.tensor([self.poisson_lambda], dtype=torch.float32)
                    )
                    .long()
                    .item()
                )
                if num_noise > 0:
                    for c in range(data.shape[-1]):  # for each channel
                        for _ in range(num_noise):
                            idx = [i, torch.randint(0, data.shape[1], (1,)).item()]
                            idx.append(c)  # add channel index
                            out[tuple(idx)] = 1

        return out


class OnlySomeClassesCollateFn(CollateFN):
    def __init__(
        self,
        class_ids=None,
        rename_labels=True,
        redraw_each_batch=False,
        num_classes=None,
        num_kept_classes=None,
        slice_amounts=False,
        num_slices=7,
        slicing_thrs=None,
        slicing_thrs_maxs=None,
        include_zero_slice=False,
        surrogate_labels=None,
        jitter_std=0.0,
        dropout_prob=0.0,
        poisson_lambda=0.0,
        padding_len=400,
        permute_data=False,
        permutation_seed=None,
    ):
        super().__init__(
            slice_amounts=slice_amounts,
            num_slices=num_slices,
            slicing_thrs=slicing_thrs,
            slicing_thrs_maxs=slicing_thrs_maxs,
            include_zero_slice=include_zero_slice,
            jitter_std=jitter_std,
            dropout_prob=dropout_prob,
            poisson_lambda=poisson_lambda,
            padding_len=padding_len,
            permute_data=permute_data,
            permutation_seed=permutation_seed,
        )
        self.class_ids = class_ids
        self.rename_labels = rename_labels
        self.redraw_each_batch = redraw_each_batch
        self.num_classes = num_classes
        self.num_kept_classes = num_kept_classes
        self.surrogate_labels = surrogate_labels

        if self.redraw_each_batch:
            self.rename_labels = False

        if (self.class_ids is None and not self.redraw_each_batch) or (
            self.class_ids is not None and self.redraw_each_batch
        ):
            raise ValueError("Either class_ids are pre-defined or randomly sampled.")
        if self.redraw_each_batch and (
            self.num_classes is None or self.num_kept_classes is None
        ):
            raise ValueError(
                "If labels are randomly sampled then num_classes and num_kept_classes should be defined."
            )

    def additional_stuff(self, data, targets):
        og_data = data.clone()
        og_targets = targets.clone()
        if self.redraw_each_batch:
            # randomly sample class_ids
            self.class_ids = random.sample(
                range(self.num_classes), self.num_kept_classes
            )
            print(f"\nRedrawing class_ids: {self.class_ids}\n")

        # get only samples from self.class_ids classes
        mask = torch.ones_like(targets)
        for class_id in self.class_ids:
            class_indices = (targets == class_id).to(torch.int)
            mask = mask * (1 - class_indices)
        mask_idx = (mask == 0).nonzero()

        data = data[mask_idx, :].squeeze(dim=1)
        targets = targets[mask_idx].squeeze(dim=1)

        if self.rename_labels:
            if self.surrogate_labels is None:
                labels_to_rename = self.class_ids
            else:
                labels_to_rename = self.surrogate_labels
            # rename class labels starting with 0
            for i, class_id in enumerate(labels_to_rename):
                class_indices = (targets == class_id).to(torch.int)
                targets = targets * (1 - class_indices) + i * class_indices

        if data.dim() == 2:  # can only happen if a single sample exists per batch
            data.unsqueeze(dim=0)

        # If no samples remain, pick one random sample from the original batch
        if data.shape[0] == 0:
            rand_idx = random.randint(0, og_targets.shape[0] - 1)
            data = og_data[rand_idx].unsqueeze(dim=0)
            targets = og_targets[rand_idx].unsqueeze(dim=0)

        return data, targets
