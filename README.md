# DendroNN

This repository contains the code for “DendroNN: Dendrocentric Neural Networks for Energy-Efficient Classification of Event-Based Data”. Training has two phases: rewiring the DendroNN layers, followed by supervised training of the output layer.

## Prerequisites

The documented environment uses Python 3.12, Git, and network access for installing packages and downloading datasets. GPU training requires an NVIDIA GPU with a driver that supports the CUDA 12.1 runtime. Git LFS is additionally required for NeuroMorse experiments. The SHD and MNIST-family datasets require sufficient local disk space below the configured data directory.

## Installation

The tested environment uses **Python 3.12**. For the intended GPU-based training workflow, use the CUDA installation below. Create a virtual environment, activate it, and install the PyTorch CUDA 12.1 build first:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
python -m pip install --no-deps -r requirements.txt
```

The CUDA wheel determines the CUDA runtime used by PyTorch, while the host NVIDIA driver must be compatible with it. Check the NVIDIA driver and CUDA compatibility before starting training. The `--no-deps` flag prevents the second command from replacing the CUDA-enabled PyTorch packages with packages resolved from the default package index.

Verify the host driver before running a GPU experiment:

```bash
nvidia-smi
```

The PyTorch wheel supplies the CUDA runtime; a separate CUDA toolkit installation is not required for this project. The host NVIDIA driver must nevertheless support CUDA 12.1.

For CPU-only use, install the same environment with the default PyPI packages instead:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CPU-only execution is mainly suitable for configuration and smoke checks; the rewiring phase is intended for a machine with substantial memory. The documented environment uses Python's `venv` and does not require conda.

## Configuration and training

Experiment hyperparameters are stored in YAML files. The canonical SHD configuration is [config/shd.yaml](config/shd.yaml). Dataset paths, device selection, seeds, and worker counts can be changed there or overridden without editing Python:

To verify the installation with one training, validation, compression, and test batch, start with the small smoke configuration:

```bash
python train.py --config config/smoke.yaml --device cpu
```

The smoke configuration disables rewiring, uses a small supervised model, and is intended only to verify the end-to-end software path. It does not measure model quality.

Once the smoke test passes, run the full two-phase SHD training workflow on a GPU:

```bash
python train.py --config config/shd.yaml --device 0 --data-root ./data
```

The command-line overrides are `--device` (`cpu`, `auto`, or a CUDA index), `--seed`, `--data-root`, `--experiment-name`, and `--num-runs`. The `--device auto` setting selects CUDA device 0 when CUDA is available and otherwise uses the CPU. Dataset selection is controlled by the selected YAML file. In particular, `config/p_smnist.yaml` uses `dataset: sMNIST` together with `permute_data: true` to enable permuted sequential MNIST.

The full training command can take a long time.

For a CPU experiment, for example (full training may be impractical on CPU):

```bash
python train.py --config config/shd.yaml --device cpu --seed 42 --num-runs 1
```

The repository includes configurations for the considered dataset variants and a smoke test:

| Configuration | Dataset |
| --- | --- |
| [config/shd.yaml](config/shd.yaml) | SHD |
| [config/neuromorse.yaml](config/neuromorse.yaml) | NeuroMorse |
| [config/smnist.yaml](config/smnist.yaml) | sequential MNIST |
| [config/p_smnist.yaml](config/p_smnist.yaml) | permuted sequential MNIST |
| [config/smoke.yaml](config/smoke.yaml) | one-batch SHD software check |

These files make the intended hyperparameters explicit. Dataset support still follows the implementation in `create_dataloader.py`.

The provided YAML files define the intended training workflows for the four datasets reported in the paper. Each configuration defaults to one run with seed `42`; figure- or table-level aggregation is not encoded in the training script, so the number of runs averaged for reported paper results should be taken from the manuscript and its supplements. Exact accuracy values can vary with the selected hardware, package versions, and random seeds.

SHD and MNIST-family datasets are downloaded or read below `data_root`. NeuroMorse does not use `data_root`: its preparation module clones the upstream repository into `neuromorse_data/NeuroMorse/` when needed, downloads its Git LFS data, and creates PyTorch datasets from the upstream corpus when the training data is requested. Results, checkpoints, TensorBoard logs, CSV metrics, compression reports, and confusion matrices are written below `logs/<dataset>/<experiment_name>/` relative to the current working directory.

For NeuroMorse, install and initialize Git LFS before running the configuration:

```bash
git lfs install
python train.py --config config/neuromorse.yaml --device 0
```

The first NeuroMorse run requires network access and sufficient disk space. If the upstream checkout contains a Git LFS pointer instead of the corpus, run `git lfs pull` from `neuromorse_data/NeuroMorse/`.

To continue from a checkpoint, set `load_checkpoint_folder` to an existing experiment output directory and enable the relevant `continue_rewiring_from_checkpoint` or `continue_supervised_from_checkpoint` option. The folder must contain the checkpoint layout produced by the corresponding training phase; use the same dataset and compatible model configuration.

After supervised training, the workflow evaluates the original model and a compressed copy. `pruning_level` controls the target fraction of pruned output weights and `bit_width` controls manual output-weight quantization. These utilities report model-weight sparsity and quantization results; they are not an energy model or an accelerator implementation.

## Hardware and runtime

The paper experiments used NVIDIA RTX 6000 Ada Generation and A6000 hardware. Runtime depends on the selected dataset configuration, hardware, and number of runs; this repository does not prescribe a fixed wall-clock duration for either rewiring or supervised training. Depending on the chosen rewiring phase hyperparameters, rewiring can last anywhere between a single batch and hundreds of epochs. Supervised training duration depends on the configuration.

To accelerate training progress, the rewiring phase uses a network that is significantly larger than the final network trained during the supervised phase. The rewiring network size is controlled by `num_searching_units`, which is a main driver of GPU memory consumption. Reduce this value if you encounter GPU out-of-memory errors.

The rewiring phase allocates large intermediate buffers and, with the default SHD configuration, can require several gigabytes of GPU memory. It is not expected to run comfortably on a small GPU. Use `--device cpu` only for configuration and smoke checks unless the machine has sufficient memory.

## Reproducibility and reported results

The seed is part of each YAML configuration and is recorded in the experiment hyperparameters. This repository supports reproduction of the reported accuracy results, but not the efficiency claims. Details regarding the efficiency numbers should be taken from the main manuscript and its supplements; the pruning and output-weight quantization code here is not an energy model or accelerator implementation.

## Citation

```bibtex
@article{krausse2026dendronn,
  title={DendroNN: Dendrocentric Neural Networks for Energy-Efficient Classification of Event-Based Data},
  author={Krausse, Jann and Su, Zhe and Mama, Kyrus and Knobloch, Klaus and Indiveri, Giacomo and Becker, J{\"u}rgen and others},
  journal={arXiv preprint arXiv:2603.09274},
  year={2026}
}
```
