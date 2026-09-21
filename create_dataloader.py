import torch
import torchvision
from torch.utils.data import DataLoader, random_split

from data import TonicDataset
from collate_functions import CollateFN
from neuromorse_data.get_neuromorse_data import create_neuromorse_dataset


def create_dataloaders(
    dataset,
    seq_len,
    time_window,
    spat_ds_fac,
    batch_size,
    slice_input_by_amount,
    num_slices,
    slicing_thrs,
    slicing_thrs_maxs,
    include_zero_slice,
    padding_size,
    aug_og_data,
    aug_kwargs,
    denoise_data,
    denoise_mode,
    num_crop_pixel,
    task_type,
    jitter_level,
    dropout_level,
    poisson_level,
    permute_data,
    permutation_seed,
    data_root="./data",
    num_workers=2,
):
    if (
        dataset != "NeuroMorse" and "sMNIST" not in dataset
    ):  # everything but NeuroMorse, sMNIST, and p-sMNIST
        data = TonicDataset(
            nb_bins=seq_len,
            time_window=time_window,
            spat_ds_fac=spat_ds_fac,
            dataset_name=dataset,
            aug_og_data=aug_og_data,
            aug_kwargs=aug_kwargs,
            denoise=denoise_data,
            denoise_mode=denoise_mode,
            num_crop_pixel=num_crop_pixel,
            data_root=data_root,
        )

        print("Defining dataloader...")
        if task_type is not None:
            augmentation_kwargs = {
                "size_share": 0.7,
                "num_noise_events": 1,
                "drop_event_prob": 0.9,
                "std_t": 10,
                "var_x": 40,  # this is the best value from SHD_augmentation_30000units_spatialjitter
                # "var_x": 10,
            }

            data.apply_augmentation(task_type, augmentation_kwargs)
            non_aug_data = [data.data_dict["train_set"]]
            aug_data = data.augmented_datasets
            all_sets = non_aug_data + aug_data
            total_train_data = torch.utils.data.ConcatDataset(all_sets)
        else:
            total_train_data = data.data_dict["train_set"]

        collate_fn = CollateFN(
            slice_amounts=slice_input_by_amount,
            num_slices=num_slices,
            slicing_thrs=slicing_thrs,
            slicing_thrs_maxs=slicing_thrs_maxs,
            include_zero_slice=include_zero_slice,
            padding_len=padding_size,
        )

        train_loader = DataLoader(
            total_train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
        val_loader = DataLoader(
            data.data_dict["valid_set"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
        if "test_set" in data.data_dict:
            test_loader = DataLoader(
                data.data_dict["test_set"],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )
        else:
            test_loader = None

        in_shape = data.sensor_size[0]
        if slice_input_by_amount:
            in_shape = in_shape * (num_slices + include_zero_slice)
        if dataset == "SHD":
            out_shape = 20
            limit_train_batches = 1.0
            limit_val_batches = 0.25
            check_val_every_n_epoch = 1
        else:
            raise Exception(f"Dataset {dataset} is not yet implemented.")

    elif dataset == "NeuroMorse":
        train_dataset, test_dataset, val_dataset = create_neuromorse_dataset()
        total_train_data = train_dataset

        train_collate_fn = CollateFN(
            jitter_std=jitter_level,
            dropout_prob=dropout_level,
            poisson_lambda=poisson_level,
        )
        test_collate_fn = CollateFN(
            jitter_std=jitter_level,
            dropout_prob=dropout_level,
            poisson_lambda=poisson_level,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=test_collate_fn,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=test_collate_fn,
            drop_last=True,
        )

        slice_input_by_amount = False

        in_shape = 2
        out_shape = 50

        limit_train_batches = 1.0
        limit_val_batches = 1.0
        check_val_every_n_epoch = 1
    else:  # sMNIST variants
        # Define augmentation transforms
        aug_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomRotation(degrees=15),
                torchvision.transforms.RandomAffine(
                    degrees=0,  # rotation already handled above
                    translate=(2 / 28, 2 / 28),  # ±2 pixels for 28x28 images
                    scale=(0.95, 1.05),  # ±5% scaling
                ),
                torchvision.transforms.ToTensor(),
            ]
        )

        # No augmentation for test data
        no_aug_transform = torchvision.transforms.ToTensor()

        aug_train_and_val_dataset = torchvision.datasets.MNIST(
            root=data_root, train=True, transform=aug_transform, download=True
        )
        no_aug_train_and_val_dataset = torchvision.datasets.MNIST(
            root=data_root, train=True, transform=no_aug_transform, download=True
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_root, train=False, transform=no_aug_transform, download=True
        )

        # split into train and val data
        train_size = int(0.8 * len(no_aug_train_and_val_dataset))
        val_size = len(no_aug_train_and_val_dataset) - train_size

        if not aug_og_data:
            train_dataset, val_dataset = random_split(
                no_aug_train_and_val_dataset, [train_size, val_size]
            )
        else:
            generator = torch.Generator().manual_seed(
                42
            )  # Use fixed seed for reproducibility
            train_dataset, _ = random_split(
                aug_train_and_val_dataset, [train_size, val_size], generator=generator
            )
            generator = torch.Generator().manual_seed(
                42
            )  # Same seed for identical split
            _, val_dataset = random_split(
                no_aug_train_and_val_dataset,
                [train_size, val_size],
                generator=generator,
            )

        total_train_data = train_dataset

        collate_fn = CollateFN(
            slice_amounts=slice_input_by_amount,
            num_slices=num_slices,
            slicing_thrs=slicing_thrs,
            include_zero_slice=include_zero_slice,
            padding_len=padding_size,
            permute_data=permute_data,
            permutation_seed=permutation_seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            drop_last=True,
        )

        in_shape = 1
        if slice_input_by_amount:
            in_shape = in_shape * (num_slices + include_zero_slice)
        out_shape = 10

        limit_train_batches = 0.2
        limit_val_batches = 0.2
        check_val_every_n_epoch = 1

    return (
        train_loader,
        val_loader,
        test_loader,
        in_shape,
        out_shape,
        total_train_data,
        limit_train_batches,
        limit_val_batches,
        check_val_every_n_epoch,
    )
