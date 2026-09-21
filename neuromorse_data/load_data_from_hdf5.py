import h5py
import numpy as np


def create_datasets_from_hdf5_file(fname):

    test_a = h5py.File(f"neuromorse_data/NeuroMorse/data/Test/{fname}.h5", "r+")
    train_a = h5py.File(f"neuromorse_data/NeuroMorse/data/Train/{fname}.h5", "r+")

    arr = np.empty(
        shape=test_a["Spikes"]["Times"][0].shape,
        dtype=[("t", "<f4"), ("x", "<f4"), ("p", "<f4")],
    )

    arr["t"] = test_a["Spikes"]["Times"][0]
    arr["x"] = test_a["Spikes"]["Channels"][0]
    arr["p"] = np.ones_like(arr["t"])

    train_dataset = []
    test_dict = {}
    for i in range(50):
        train_arr = np.empty(
            shape=train_a["Spikes"]["Times"][i].shape,
            dtype=[("t", "<f4"), ("x", "<f4"), ("p", "<f4")],
        )
        train_arr["t"] = train_a["Spikes"]["Times"][i]
        train_arr["x"] = train_a["Spikes"]["Channels"][i]
        train_arr["p"] = np.ones_like(train_arr["t"])
        train_dataset.append((train_arr, train_a["Labels"]["Labels"][i]))

        label = test_a["Labels"]["Labels"][i]
        word_count = np.shape(test_a["Labels"]["Start Times"][0])[0]
        test_dict[label] = [
            word_count,
            test_a["Labels"]["Start Times"][i],
            test_a["Labels"]["End Times"][i],
        ]

    test_dataset = (arr, test_dict)

    return train_dataset, test_dataset


if __name__ == "__main__":
    data = create_datasets_from_hdf5_file("Clean")
    data_noise = create_datasets_from_hdf5_file("Dropout-High Jitter-High Poisson-High")

    print("Testing dataset creation...")
