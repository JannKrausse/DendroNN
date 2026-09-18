# DendroNN

This repository contains the code for “DendroNN: Dendrocentric Neural Networks for Energy-Efficient Classification of Event-Based Data”. Training has two phases: rewiring the DendroNN layers, followed by supervised training of the output layer.

## Installation

The tested environment uses **Python 3.12**. Create a virtual environment, install a PyTorch build appropriate for the target machine, and then install the remaining dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file pins Python-compatible versions of PyTorch (`torch==2.2.1`, `torchvision==0.17.1`, and `torchaudio==2.2.1`). The default PyPI installation provides the PyTorch wheel published for that platform; for GPU use, install the matching official PyTorch CUDA wheel instead of the default wheel, then install the remaining requirements. The CUDA wheel determines the CUDA runtime used by PyTorch, while the host NVIDIA driver must be compatible with it. For example, a CUDA 12.1 installation can use:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
python -m pip install --no-deps -r requirements.txt
```

For CPU-only use, the default command is sufficient, although the rewiring phase is intended for a machine with substantial memory. The repository does not require conda.

## Configuration and training

Experiment hyperparameters are stored in YAML files. The canonical SHD configuration is [config/shd.yaml](config/shd.yaml). Dataset paths, device selection, seeds, worker counts, and output locations can be changed there or overridden without editing Python:

```bash
python train.py --config config/shd.yaml --device 0 --data-root ./data
```

Useful overrides are `--dataset`, `--device` (`cpu`, `auto`, or a CUDA index), `--seed`, `--data-root`, `--experiment-name`, and `--num-runs`. For example:

```bash
python train.py --config config/shd.yaml --device cpu --seed 42 --num-runs 1
```

The repository includes configurations for the considered dataset variants:

| Configuration | Dataset |
| --- | --- |
| [config/shd.yaml](config/shd.yaml) | SHD |
| [config/neuromorse.yaml](config/neuromorse.yaml) | NeuroMorse |
| [config/smnist.yaml](config/smnist.yaml) | sequential MNIST |
| [config/p_smnist.yaml](config/p_smnist.yaml) | permuted sequential MNIST |

These files make the intended hyperparameters explicit. Dataset support still follows the implementation in `create_dataloader.py`.

The results for DendroNN on each of the four datasets reported in the paper can be reproduced by using the respective dataset configuration YAML file above.

SHD and MNIST-family datasets are downloaded or read below `data_root`. NeuroMorse additionally requires the NeuroMorse data loader and its prepared data files in `neuromorse_data/`; see that directory for the expected external repository. Results, checkpoints, TensorBoard logs, and confusion matrices are written below `logs/`.

## Hardware and runtime

The paper experiments were run on a single NVIDIA RTX 6000 Ada Generation/A6000. Runtime depends on the selected dataset configuration, hardware, and number of runs; this repository does not prescribe a fixed wall-clock duration for either rewiring or supervised training. Depending on the chosen selectivity criterion, rewiring can last anywhere between a single batch and hundreds of epochs. However, the supervised training duration depends on the user's configuration.

The rewiring phase allocates large intermediate buffers and is not expected to run comfortably on a small GPU. Use `--device cpu` only for configuration and smoke checks unless the machine has sufficient memory.

## Reproducibility and reported results

The seed is part of each YAML configuration and is recorded in the experiment hyperparameters. Each table or figure in the paper should be mapped to a named configuration and list the number of independent seeds averaged. For now, this repository aids in reproducing the paper's accuracy numbers only; it does not reproduce the efficiency claims. The pruning and output-weight quantization code is not an energy model or accelerator implementation, and the accelerator and energy numbers reported in the paper were produced externally.

## NeuroMorse

To run NeuroMorse experiments, the first attempt to create the dataloader clones the complete [upstream NeuroMorse repository](https://github.com/Ben-E-Walters/NeuroMorse) into `neuromorse_data/NeuroMorse/` when that directory is absent. This requires Git, Git LFS, and network access. Install Git LFS using your operating system's package manager, then initialize it with:

```bash
git lfs install
```

The loader downloads the upstream Git LFS data during the clone. The prepared train/test HDF5 files are external data and require sufficient disk space; if an existing checkout contains LFS pointer files, run `git lfs pull` from `neuromorse_data/NeuroMorse/`.

## Citation

```bibtex
@article{krausse2026dendronn,
  title={DendroNN: Dendrocentric Neural Networks for Energy-Efficient Classification of Event-Based Data},
  author={Krausse, Jann and Su, Zhe and Mama, Kyrus and Knobloch, Klaus and Indiveri, Giacomo and Becker, J{\"u}rgen and others},
  journal={arXiv preprint arXiv:2603.09274},
  year={2026}
}
```
