from pathlib import Path

import yaml


REQUIRED_KEYS = {
    "dataset",
    "experiment_name",
    "num_runs",
    "seed",
    "device",
    "data_root",
    "batch_size",
    "time_window",
    "spat_ds_fac",
    "num_hidden_units",
    "num_searching_units",
}
SUPPORTED_DATASETS = {"SHD", "NeuroMorse", "sMNIST"}


def load_config(config_path, overrides=None):
    """Load and validate an experiment configuration from YAML."""
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    missing_keys = sorted(REQUIRED_KEYS.difference(config))
    if missing_keys:
        raise ValueError(
            f"Configuration {config_path} is missing required keys: {', '.join(missing_keys)}"
        )

    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value

    if config["num_runs"] < 1:
        raise ValueError("num_runs must be at least 1.")
    if config["dataset"] not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Unsupported dataset {config['dataset']!r}. Supported datasets: {supported}."
        )
    if config["num_hidden_units"] < 1 or config["num_searching_units"] < 1:
        raise ValueError("The number of hidden and searching units must be positive.")
    if config["seed"] is not None and not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer or null.")

    return config
