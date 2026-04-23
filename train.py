import os
import sys
import time
import joblib
import random
import yaml

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

from pl_snn import SpikingNetwork, ConfusionMatrixPlotterCallback
from network import DendroNN
from collate_functions import OnlySomeClassesCollateFn
from rewiring_model import RewiringModel, RewiringProgressCallback
from create_dataloader import create_dataloaders
from compression import test_model_with_compression

import warnings
warnings.filterwarnings("ignore", module="pytorch_lightning")


def main():
    NUM_RUNS = 1
    GPU = True

    DEVICE = 4
    EXPERIMENT_NAME = "test"
    DATASET = "SHD"  # SHD, NMNIST, NeuroMorse, sMNIST
    permute_data = False  # want p-sMNIST? set DATASET = sMNIST and permute_data = True.
    if permute_data:
        if DATASET != "sMNIST":
            raise Exception(f"Permuting data is only implemented for sMNIST dataset. Current dataset: {DATASET}.")
        DATASET = "p-sMNIST"
        permutation_seed = random.randint(0, 1000)
    else:
        permutation_seed = None

    REWIRING = True
    SUPERVISED = True
    CONTINUE_REWIRING_FROM_CKPT = False
    CONTINUE_SUPERVISED_FROM_CKPT = False

    load_checkpoint_folder = None
    
    for _ in range(NUM_RUNS):
        # seed = 42
        seed = random.randint(0, 1000)
        torch.manual_seed(seed)
        print(f"This run's random seed is {seed}.")

        dataset = DATASET
        folder_name = dataset + "/"

        batch_size = 256
        time_window = 8  # corresponds to ms of data per time bin
        if dataset == "SHD":
            padding_size = int(4 / time_window * 292)  # 292 is max sequence in
            seq_len = padding_size
        elif "sMNIST" in dataset:  # sMNIST or p-sMNIST
            # padding_size = 14**2  # 28x28 images downsampled to 14x14
            padding_size = 28**2
            seq_len = padding_size
        elif dataset == "NeuroMorse":
            seq_len = 100
            padding_size = None
        else:
            if dataset != "NeuroMorse" and "sMNIST" not in dataset:  # NeuroMorse, sMNIST, or p-sMNIST
                raise Exception(f"Dataset {dataset} is not yet implemented for padding size calculation.")
        spat_ds_fac = 1/7
        num_crop_pixel = 0
        denoise_data = True
        denoise_mode = 'tonic'
        binary_data = False
        aug_og_data = False
        aug_kwargs = None
        task_type = None

        if dataset != "SHD":
            slice_input_by_amount = False
            num_slices = 1
            include_zero_slice = False
            slicing_thrs = [0.5]
            slicing_thrs_maxs = None
        else:
            slice_input_by_amount = True
            offset = 3.0
            scaling = 1.3
            max_value = 25  # only for timewindow 8 and spat_ds_fac 1/7
            slice_centers = [i + 1 for i in range(max_value)]
            num_slices = len(slice_centers)
            include_zero_slice = False
            slicing_thrs = []
            slicing_thrs_maxs = []
            for i in slice_centers:
                slicing_thrs.append(i / scaling - offset)
                slicing_thrs_maxs.append(i * scaling + offset)

        jitter_level = 'none'  # 'none', 'low', 'high', or any float
        dropout_level = 'none'  # 'none', 'low', 'high', or any float
        poisson_level = 'none'  # 'none', 'low', 'high', or any float

        # rewiring phase
        num_seaching_units = 30000  # number of units to search for in the rewiring phase
        thr_factor = 30
        thr_low = -3 * thr_factor
        thr_high = 15 * thr_factor
        penalty_fac = 1 / 0.5  # prefactor in penalty term: 1 / (fac * num_classes), BE CAREFUL WITH CHOOSING THIS AS A SMALL FACTOR IN COMBINATION WITH EXCLUSIVE SEQUENCES CAN LEAD TO ALWAYS NEGATIVE LONGEVITY CHANGES
        class_selectivity_threshold = 0.05
        selectivity_gap_to_other_classes = 0.03
        experiment_mode = "std"  # std, selectivity_gap, specific_selectivity

        net_type = "den"  # den, lif
        if net_type == "lif":
            REWIRING = False

        # for all nets
        num_hidden_layer = 1
        num_hidden_units = 3000
        output_bias = False
        output_decoder = "max"  # max, avg, last, train, sum:drop_stop
        assert num_seaching_units > 2 * num_hidden_units or not REWIRING, "Number of searching units should be at least twice the number of hidden units for better selection. Otherwise validation of rewiring phase will never get triggered."

        # for DendroNN
        num_spines = 2
        min_num_spines = 2
        max_seq_len = seq_len # int(seq_len * 0.9)
        spike_acceptance_window = 0
        parallel_sequences = True
        refrac_period = False  # every unit outputs only one spike per inference at max

        lr = 1e-3
        lr_scheduled = False
        lr_decay = 15
        lr_update_freq = 5
        l2_regu = 0
        dropout = 0.

        folder_name = folder_name + EXPERIMENT_NAME

        print("Getting data...")
        train_loader, val_loader, test_loader, in_shape, out_shape, total_train_data, limit_train_batches, limit_val_batches, check_val_every_n_epoch = create_dataloaders(
            dataset, seq_len, time_window, spat_ds_fac, batch_size, slice_input_by_amount, num_slices, slicing_thrs, slicing_thrs_maxs, include_zero_slice, padding_size, aug_og_data, aug_kwargs, denoise_data, denoise_mode, num_crop_pixel, task_type, jitter_level, dropout_level, poisson_level, permute_data, permutation_seed
        )

        num_classes = out_shape
        target_num_frozen_units = num_hidden_units

        print("Arranging data logging...")
        logs_folder = f"{os.getcwd()}/logs"
        os.makedirs(logs_folder, exist_ok=True)

        print("Defining Callbacks...")
        callbacks_rewiring = [
            RewiringProgressCallback(),
        ]

        hyperparameters = {
            "num_spines": num_spines,
            "min_num_spines": min_num_spines,
            "num_hidden_layer": num_hidden_layer,
            "num_hidden_units": num_hidden_units,
            "max_seq_len": max_seq_len,
            "spike_acceptance_window": spike_acceptance_window,
            "in_shape": in_shape,
            "out_shape": out_shape,
            "dataset_name": dataset,
            "seq_len": seq_len,
            "spat_ds": spat_ds_fac,
            "batch_size": batch_size,
            "lr": lr,
            "output_decoder": output_decoder,
            "binary_data": binary_data,
            "net_type": net_type,
            "dropout": dropout,
            "seed": seed,
            "lr_scheduled": lr_scheduled,
            "lr_decay": lr_decay,
            "lr_update_freq": lr_update_freq,
            "task_type": task_type,
            "aug_og_data": aug_og_data,
            "output_bias": output_bias,
            "thr_low": thr_low,
            "thr_high": thr_high,
            "penalty_fac": penalty_fac,
            "class_selectivity_threshold": class_selectivity_threshold,
            "num_seaching_units": num_seaching_units,
            "denoise_data": denoise_data,
            "denoise_mode": denoise_mode,
            "slice_input_by_amount": slice_input_by_amount,
            "num_slices": num_slices,
            "slicing_thrs": slicing_thrs,
            "selectivity_gap_to_other_classes": selectivity_gap_to_other_classes,
            "experiment_mode": experiment_mode,
            "include_zero_slice": include_zero_slice,
            "num_crop_pixel": num_crop_pixel,
            "jitter_level": jitter_level,
            "dropout_level": dropout_level,
            "poisson_level": poisson_level,
            "time_window": time_window,
            "padding_size": padding_size,
            "l2_regu": l2_regu,
            "permute_data": permute_data,
            "permutation_seed": permutation_seed,
            "slicing_thrs_maxs": slicing_thrs_maxs,
            "parallel_sequences": parallel_sequences,
            "refrac_period": refrac_period,
        }
        tbl_rewiring = TensorBoardLogger(save_dir=logs_folder, name=(folder_name + "/rewiring"))
        hyperparameters_path = f"{tbl_rewiring.log_dir}/hyperparameters.save"
        os.makedirs(os.path.dirname(hyperparameters_path), exist_ok=True)
        joblib.dump(hyperparameters, hyperparameters_path)

        if load_checkpoint_folder:
            try:
                hyperparameters = joblib.load(
                    f"{load_checkpoint_folder}/hyperparameters.save"
                )
            except FileNotFoundError:
                hyperparameters = yaml.safe_load(open(f"{load_checkpoint_folder}/hparams.yaml"))
            target_num_frozen_units = hyperparameters["num_hidden_units"]
            num_classes = hyperparameters["out_shape"]

        print("Defining network...")
        if hyperparameters["net_type"] == "den":
            if not isinstance(hyperparameters["num_spines"], list):
                net_rewiring = DendroNN(
                    num_spines=hyperparameters["num_spines"],
                    min_num_spines=hyperparameters["min_num_spines"],
                    num_hidden_layer=hyperparameters["num_hidden_layer"],
                    num_hidden_units=hyperparameters["num_seaching_units"],
                    max_seq_len=hyperparameters["max_seq_len"],
                    spike_acceptance_window=hyperparameters["spike_acceptance_window"],
                    in_shape=hyperparameters["in_shape"],
                    out_shape=hyperparameters["out_shape"],
                    batch_size=hyperparameters["batch_size"],
                    hyperparameters=hyperparameters,
                    dropout_p=hyperparameters["dropout"],
                    bias=hyperparameters["output_bias"],
                    refrac_period=hyperparameters["refrac_period"],
                    parallel_sequences=hyperparameters["parallel_sequences"],
                )
            else:
                raise Exception(f"Ambiguous length of sequences.")
        else:
            raise Exception(f"Net type {net_type} unknown.")
        
        if GPU:
            acc = "gpu"
            devices = [DEVICE]  # 0 or 1
        else:
            acc = "cpu"
            devices = 1

        if REWIRING:
            model_rewiring = RewiringModel(
                net_rewiring,
                num_classes=num_classes,
                target_num_frozen_units=target_num_frozen_units,
                experiment_name=EXPERIMENT_NAME,
                lower_longevity_threshold=hyperparameters["thr_low"],
                upper_longevity_threshold=hyperparameters["thr_high"],
                penalty_fac=hyperparameters["penalty_fac"],
                dataset=hyperparameters["dataset_name"],
                class_selectivity_threshold=hyperparameters["class_selectivity_threshold"],
                selectivity_gap_to_other_classes=hyperparameters["selectivity_gap_to_other_classes"],
                experiment_mode=hyperparameters["experiment_mode"],
            )


            if load_checkpoint_folder:
                # list all files in the folder that end with .ckpt
                checkpoint_files = [
                    f for f in os.listdir(load_checkpoint_folder + "/checkpoints") if f.endswith(".ckpt")
                ]
                rewiring_checkpoint = checkpoint_files[0]  # should just be one
                model_dict = torch.load(f"{load_checkpoint_folder}/checkpoints/{rewiring_checkpoint}", map_location=f"cuda:{DEVICE}")  # load checkpoint
                try:
                    actual_batch_size = model_dict['state_dict']['model.units.0.expected_spikes_update_mask'].shape[0]
                except KeyError:
                    actual_batch_size = model_dict['state_dict']['model.units.0.zeros_buffer'].shape[0]
                for i in range(model_rewiring.model.num_hidden_layer):
                    model_rewiring.model.units[str(i)].batch_size = actual_batch_size
                    model_rewiring.model.units[str(i)].reset(f"cuda:{DEVICE}")
                del model_dict["state_dict"]["frozen_label_stats"]  # remove best model state dict to avoid size mismatch
                model_rewiring.load_state_dict(model_dict["state_dict"], strict=False)  # load model state dict from checkpoint
                
            if load_checkpoint_folder is None or CONTINUE_REWIRING_FROM_CKPT:
                joblib.dump(hyperparameters, hyperparameters_path)

                print("Defining lightning trainer...")
                trainer_rewiring = Trainer(
                    max_epochs=10000000,
                    accelerator=acc,
                    devices=devices,
                    log_every_n_steps=1,
                    callbacks=callbacks_rewiring,
                    logger=tbl_rewiring,
                    limit_train_batches=limit_train_batches,
                    limit_val_batches=limit_val_batches,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                )

                rename_labels = False
                surrogate_labels = None
                try:
                    while model_rewiring.actively_searched_classes.sum() > 0:
                        collate_fn_rewiring = OnlySomeClassesCollateFn(class_ids=model_rewiring.actively_searched_classes.nonzero().squeeze(1).tolist(), rename_labels=rename_labels, slice_amounts=slice_input_by_amount, num_slices=num_slices, slicing_thrs=slicing_thrs, slicing_thrs_maxs=slicing_thrs_maxs, include_zero_slice=include_zero_slice, surrogate_labels=surrogate_labels, padding_len=padding_size, permute_data=permute_data, permutation_seed=permutation_seed,
                                                                        #    jitter_std=jitter_level, dropout_prob=dropout_level, poisson_lambda=poisson_level
                                                                           )
                        train_loader_rewiring = DataLoader(
                            total_train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            collate_fn=collate_fn_rewiring,
                            drop_last=True,
                        )

                        trainer_rewiring.fit(model_rewiring, train_loader_rewiring, train_loader)  # use train_loader for validation (only stats calculation)
                        trainer_rewiring.should_stop = False
                        
                        time.sleep(1)  # wait for a second to enable keyboard interrupt
                except KeyboardInterrupt:
                    print("Rewiring training interrupted by user. Proceeding to supervised phase...")
                    sys.exit(0)
                model_rewiring.final_validation = True  # set final validation to True to get final label stats
                trainer_rewiring.validate(model_rewiring, train_loader)  # validate on full training set to get final label stats
            print("Rewiring learning phase finished.\n")

        if SUPERVISED:
            if REWIRING:
                rewired_hidden_units = model_rewiring.frozen_units.sum().item()
            else:
                rewired_hidden_units = hyperparameters["num_hidden_units"]
            net = DendroNN(
                num_spines=hyperparameters["num_spines"],
                min_num_spines=hyperparameters["min_num_spines"],
                num_hidden_layer=hyperparameters["num_hidden_layer"],
                num_hidden_units=rewired_hidden_units,
                max_seq_len=hyperparameters["max_seq_len"],
                spike_acceptance_window=hyperparameters["spike_acceptance_window"],
                in_shape=hyperparameters["in_shape"],
                out_shape=hyperparameters["out_shape"],
                batch_size=hyperparameters["batch_size"],
                hyperparameters=hyperparameters,
                dropout_p=hyperparameters["dropout"],
                bias=hyperparameters["output_bias"],
                parallel_sequences=hyperparameters["parallel_sequences"],
                refrac_period=hyperparameters["refrac_period"],
            )

            if REWIRING:
                net.hidden_layer["0"].synapse_mask[:(model_rewiring.frozen_units.sum() * hyperparameters["num_spines"])] = net_rewiring.hidden_layer["0"].synapse_mask[model_rewiring.frozen_units.repeat_interleave(hyperparameters["num_spines"])].clone()
                net.units["0"].inter_spike_intervals[:model_rewiring.frozen_units.sum(), :] = net_rewiring.units["0"].inter_spike_intervals[model_rewiring.frozen_units].clone()
                net.units["0"].sequence_len_dist_rep[:model_rewiring.frozen_units.sum()] = net_rewiring.units["0"].sequence_len_dist_rep[model_rewiring.frozen_units].clone()
                print("Successfully transferred frozen units from rewiring model to supervised model.\n")

        if SUPERVISED:            
            tbl = TensorBoardLogger(save_dir=logs_folder, name=(folder_name + "/supervised"))
            cvl = CSVLogger(
                save_dir=logs_folder, name=(folder_name + "/supervised"), version=tbl.version
            )

            checkpoint_callback_val_acc = ModelCheckpoint(
                monitor="val_acc",  # Monitor validation accuracy for improvement
                dirpath=tbl.log_dir,
                filename="model-{epoch:02d}-{val_acc:.2f}",  # Filename format
                save_top_k=1,  # Save only the best model
                mode="max",  # Maximize validation accuracy
                save_last=False,  # Additionally, always save the last model
            )
            checkpoint_callback_train_acc = ModelCheckpoint(
                monitor="train_acc",  # Monitor training accuracy for improvement
                dirpath=tbl.log_dir,
                filename="model-{epoch:02d}-{train_acc:.2f}",  # Filename format
                save_top_k=1,  # Save only the best model
                mode="max",  # Maximize training accuracy
                save_last=False,  # Additionally, always save the last model
            )
            callbacks = []
            if val_loader is not None:
                callbacks.extend([
                    EarlyStopping(
                        monitor="val_loss",
                        min_delta=0.00,
                        patience=30,
                        verbose=False,
                        mode="min",
                    ),
                    checkpoint_callback_val_acc,
                ])
            else:
                callbacks.extend([
                    checkpoint_callback_train_acc,
                ])
            callbacks.extend([
                ConfusionMatrixPlotterCallback(log_path=tbl.log_dir),
            ])

            model = SpikingNetwork(
                net,
                lr=hyperparameters["lr"],
                hyperparameters=hyperparameters,
                output_decoder=hyperparameters["output_decoder"],
                lr_scheduled=hyperparameters["lr_scheduled"],
                lr_update_freq=hyperparameters["lr_update_freq"],
                lr_decay=hyperparameters["lr_decay"],
                l2_regu=hyperparameters["l2_regu"],
            )

            if load_checkpoint_folder is not None and CONTINUE_SUPERVISED_FROM_CKPT:
                checkpoint_files = [
                    f for f in os.listdir(load_checkpoint_folder) if f.endswith(".ckpt")
                ]
                supervised_checkpoint = [f for f in checkpoint_files if "val_loss" in f][0]
                model_dict = torch.load(f"{load_checkpoint_folder}/{supervised_checkpoint}", map_location=f"cuda:{DEVICE}")  # load checkpoint

                model.load_state_dict(model_dict["state_dict"])  # load model state dict from checkpoint


            print("Defining lightning trainer...")
            if val_loader is not None:
                max_epochs = 1000000  # unreasonably high number of epochs to ensure training continues until stopped by EarlyStopping
            else:
                max_epochs = 200
            trainer = Trainer(
                max_epochs=max_epochs,
                accelerator=acc,
                devices=devices,
                log_every_n_steps=1,
                enable_progress_bar=True,
                callbacks=callbacks,
                logger=[tbl, cvl],
            )

            trainer.fit(model, train_loader, val_loader)

            # Load the best checkpoint after training
            if val_loader is not None:
                # Load the best model based on validation loss
                best_model_path = checkpoint_callback_val_acc.best_model_path
                if best_model_path:
                    print(f"Loading best model from: {best_model_path}")
                    model = SpikingNetwork.load_from_checkpoint(
                        best_model_path,
                        net=net,  # Pass the network architecture
                        hyperparameters=hyperparameters,
                        lr=hyperparameters["lr"],
                        output_decoder=hyperparameters["output_decoder"],
                        lr_scheduled=hyperparameters["lr_scheduled"],
                        lr_update_freq=hyperparameters["lr_update_freq"],
                        lr_decay=hyperparameters["lr_decay"],
                        l2_regu=hyperparameters["l2_regu"],
                    )
                    print("Best model loaded successfully!")

            if test_loader is not None:
                loader_for_eval = test_loader
            else:
                loader_for_eval = val_loader
            # Test both original and compressed models
            pruning_level = 0.7
            test_results = test_model_with_compression(
                model=model,
                test_loader=loader_for_eval,
                train_loader=train_loader,
                val_loader=val_loader,
                hyperparameters=hyperparameters,
                logs_folder=logs_folder,
                folder_name=folder_name,
                logger=[tbl, cvl],
                acc=acc,
                devices=devices,
                pruning_level=pruning_level,  # fraction of output weights to prune
                bit_width=8,        # for integer quantization of output weights
                device=DEVICE
            )
            
            print(f"Test Results Summary:")
            print(f"  Original accuracy: {test_results['original'].get('test_acc', 'N/A')}")
            print(f"  Compressed accuracy: {test_results['compressed'].get('test_acc', 'N/A')}")
            if 'original' in test_results and 'compressed' in test_results:
                orig_acc = test_results['original'].get('test_acc', 0)
                comp_acc = test_results['compressed'].get('test_acc', 0)
                if orig_acc > 0:
                    acc_retention = (comp_acc / orig_acc) * 100
                    print(f"  Accuracy retention: {acc_retention:.2f}%")


if __name__ == "__main__":
    main()
    