import torch
from torch.utils.data import random_split
import tonic
import numpy as np


class CustomAudioUniformNoiseTransform:
    def __init__(self, sensor_size, nb_noise_events, random=False):
        self.sensor_size = sensor_size

        if random:
            self.nb_noise_events = nb_noise_events
        else:
            self.nb_noise_events = int(np.random.random_sample() * nb_noise_events)

        self.structured_type = np.dtype(
            [("t", np.int32), ("x", np.int32), ("p", np.int32)]
        )

    def __call__(self, e):
        noise_events = np.array([], dtype=self.structured_type)
        min_time, max_time = e[0][0], e[-1][0]
        random_time = np.random.randint(0, max_time, (self.nb_noise_events,))
        random_channel = np.random.randint(
            0, self.sensor_size[0], (self.nb_noise_events,)
        )
        for i in range(self.nb_noise_events):
            noise_events = np.append(
                noise_events,
                np.array(
                    [(random_time[i], random_channel[i], 1)], dtype=self.structured_type
                ),
            )
        noisy_e = np.append(e, noise_events)
        return noisy_e


class Custom1DCropTransform:
    def __init__(self, sensor_size, num_crop_pixel):
        self.sensor_size = sensor_size
        self.num_crop_pixel = num_crop_pixel

    def __call__(self, events):
        width = self.sensor_size[0] - self.num_crop_pixel
        height = self.sensor_size[1]
        x_mask = (events["x"] >= 0) & (events["x"] < width)
        mask = x_mask  # & y_mask

        masked_events = events[mask]
        return masked_events


class CustomAudioDenoiseTransform:
    def __init__(self, filter_time=10000, mode="tonic"):
        self.filter_time = filter_time
        self.mode = mode  # 'tonic' or 'thr'
        if mode == "tonic":
            self.call_fn = self.tonic_call
        elif mode == "thr":
            self.call_fn = self.call_thr

    def __call__(self, events):
        denoised_events = self.call_fn(events)

        return denoised_events

    def tonic_call(self, events):
        events_copy = np.zeros_like(events)
        copy_index = 0
        width = int(events["x"].max()) + 1
        timestamp_memory = np.zeros((width,)) + self.filter_time

        for event in events:
            x = int(event["x"])
            t = event["t"]
            timestamp_memory[x] = t + self.filter_time
            if (x > 0 and timestamp_memory[x - 1] > t) or (
                x < width - 1 and timestamp_memory[x + 1] > t
            ):
                events_copy[copy_index] = event
                copy_index += 1

        return events_copy[:copy_index]

    def call_thr(self, events):
        thr = 5
        events[events <= thr] = 0

        return events


class OtherAudioUniformNoiseTransform(tonic.transforms.UniformNoise):
    def __init__(self, sensor_size, n):
        super().__init__(sensor_size, n)

    def __call__(self, events):
        e = super().__call__(events=events)
        e["p"] = np.ones(
            len(e)
        )  # UniformNoise transform does not work with sensor_size=(700, 1, 1) because noise events have polarity 0 instead of 1
        return e


class CustomAudioSpatialJitterTransform:
    def __init__(self, sensor_size, var_x, clip_outliers=True):
        self.sensor_size = sensor_size
        self.var_x = var_x
        self.clip_outliers = clip_outliers

        self.structured_type = np.dtype(
            [("t", np.int32), ("x", np.int32), ("p", np.int32)]
        )

    def __call__(self, e):
        spec_type = e[0].dtype
        min_x, max_x = 0, self.sensor_size[0] - 1
        spatial_jitter = np.random.normal(
            loc=0, scale=self.var_x, size=(len(e),)
        ).astype(np.int32)
        for i in range(len(e)):
            if self.clip_outliers:
                new_x = min(max_x, max(min_x, e[i][1] + spatial_jitter[i]))
            else:
                new_x = e[i][1] + spatial_jitter[i]
            e[i] = np.array((e[i][0], new_x, e[i][2]), dtype=spec_type)

        return e


class Jitter1D:
    """
    Apply random jitter to event coordinates
    Parameters:
        max_roll (int): maximum number of pixels to roll by
    """

    def __init__(self, sensor_size, var):
        self.sensor_size = sensor_size
        self.var = var

    def __call__(self, events):
        # roll x, y coordinates by a random amount
        shift = np.random.normal(0, self.var, len(events)).astype(np.int32)
        events["x"] += shift
        # remove events who got shifted out of the sensor size
        mask = (events["x"] >= 0) & (events["x"] < self.sensor_size[0])
        events = events[mask]
        return events


class DropEventChunk:
    """
    This is directly copied from
    https://github.com/Efficient-Scalable-Machine-Learning/event-ssm/blob/main/event_ssm/transform.py

    Randomly drop a chunk of events
    """

    def __init__(self, p, max_drop_size):
        self.drop_prob = p
        self.max_drop_size = max_drop_size

    def __call__(self, events):
        max_drop_events = self.max_drop_size * len(events)
        if np.random.rand() < self.drop_prob:
            drop_size = np.random.randint(1, max_drop_events)
            start = np.random.randint(0, len(events) - drop_size)
            events = np.delete(events, slice(start, start + drop_size), axis=0)
        return events


class Roll:
    """
    This is directly copied from
    https://github.com/Efficient-Scalable-Machine-Learning/event-ssm/blob/main/event_ssm/transform.py

    Roll event x, y coordinates by a random amount

    Parameters:
        max_roll (int): maximum number of pixels to roll by
    """

    def __init__(self, sensor_size, p, max_roll):
        self.sensor_size = sensor_size
        self.max_roll = max_roll
        self.p = p

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events
        # roll x, y coordinates by a random amount
        roll_x = np.random.randint(-self.max_roll, self.max_roll)
        roll_y = np.random.randint(-self.max_roll, self.max_roll)
        events["x"] += roll_x
        events["y"] += roll_y
        # remove events who got shifted out of the sensor size
        mask = (
            (events["x"] >= 0)
            & (events["x"] < self.sensor_size[0])
            & (events["y"] >= 0)
            & (events["y"] < self.sensor_size[1])
        )
        events = events[mask]
        return events


class Rotate:
    """
    This is directly copied from
    https://github.com/Efficient-Scalable-Machine-Learning/event-ssm/blob/main/event_ssm/transform.py

    Rotate event x, y coordinates by a random angle
    """

    def __init__(self, sensor_size, p, max_angle):
        self.p = p
        self.sensor_size = sensor_size
        self.max_angle = 2 * np.pi * max_angle / 360

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events
        # rotate x, y coordinates by a random angle
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        x = events["x"] - self.sensor_size[0] / 2
        y = events["y"] - self.sensor_size[1] / 2
        x_new = x * np.cos(angle) - y * np.sin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)
        events["x"] = (x_new + self.sensor_size[0] / 2).astype(np.int32)
        events["y"] = (y_new + self.sensor_size[1] / 2).astype(np.int32)
        # clip to original range
        events["x"] = np.clip(events["x"], 0, self.sensor_size[0])
        events["y"] = np.clip(events["y"], 0, self.sensor_size[1])
        return events


class Scale:
    """
    This is directly copied from
    https://github.com/Efficient-Scalable-Machine-Learning/event-ssm/blob/main/event_ssm/transform.py

    Scale event x, y coordinates by a random factor
    """

    def __init__(self, sensor_size, p, max_scale):
        assert max_scale >= 1
        self.p = p
        self.sensor_size = sensor_size
        self.max_scale = max_scale

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events
        # scale x, y coordinates by a random factor
        scale = np.random.uniform(1 / self.max_scale, self.max_scale)
        x = events["x"] - self.sensor_size[0] / 2
        y = events["y"] - self.sensor_size[1] / 2
        x_new = x * scale
        y_new = y * scale
        events["x"] = (x_new + self.sensor_size[0] / 2).astype(np.int32)
        events["y"] = (y_new + self.sensor_size[1] / 2).astype(np.int32)
        # remove events who got shifted out of the sensor size
        mask = (
            (events["x"] >= 0)
            & (events["x"] < self.sensor_size[0])
            & (events["y"] >= 0)
            & (events["y"] < self.sensor_size[1])
        )
        events = events[mask]
        return events


class TonicDataset:
    def __init__(
        self,
        nb_bins,
        time_window,
        spat_ds_fac,
        dataset_name,
        binary_data=True,
        aug_og_data=True,
        aug_kwargs=None,
        denoise=False,
        denoise_mode="tonic",
        num_crop_pixel=0,
        val_on_test=True,
        data_root="./data",
    ):
        self.nb_bins = nb_bins
        self.time_window = time_window
        self.spat_ds_fac = spat_ds_fac
        self.dataset_name = dataset_name
        self.binary_data = binary_data
        self.denoise = denoise
        self.denoise_mode = denoise_mode
        self.num_crop_pixel = num_crop_pixel
        self.val_on_test = val_on_test
        if self.dataset_name == "SHD" or self.dataset_name == "SSC":
            self.has_one_spatial_dim = True
        else:
            self.has_one_spatial_dim = False

        self.save_path = data_root

        self.dataset = getattr(tonic.datasets, self.dataset_name)

        sensor_size = self.dataset.sensor_size
        self.not_downsampled_sensor_size = sensor_size
        self.sensor_size = (
            max(int((sensor_size[0] - self.num_crop_pixel) * spat_ds_fac), 1),
            max(
                int(
                    (sensor_size[1] - self.has_one_spatial_dim * self.num_crop_pixel)
                    * spat_ds_fac
                ),
                1,
            ),
            sensor_size[2],
        )

        self.transforms = []

        self.aug_og_data = aug_og_data
        self.aug_kwargs = aug_kwargs
        if (
            self.aug_og_data
        ):  # transforms and parameter values taken from event-by-event ssm repo
            if self.dataset_name == "SHD":
                if self.aug_kwargs is None:
                    augmentation_kwargs = {
                        "num_noise_events": 35,
                        "drop_event_prob": 0.1,
                        "std_t": 1,
                        "var_x": 0.55,
                        "max_drop_chunk": 0.02,
                        "time_skew": 1.2,
                    }
                    self.transforms.extend(
                        [
                            tonic.transforms.DropEvent(
                                p=augmentation_kwargs["drop_event_prob"]
                            ),
                            DropEventChunk(
                                p=0.3,
                                max_drop_size=augmentation_kwargs["max_drop_chunk"],
                            ),
                            Jitter1D(
                                sensor_size=self.not_downsampled_sensor_size,
                                var=augmentation_kwargs["var_x"],
                            ),
                            tonic.transforms.TimeSkew(
                                coefficient=(
                                    1 / augmentation_kwargs["time_skew"],
                                    augmentation_kwargs["time_skew"],
                                ),
                                offset=0,
                            ),
                            tonic.transforms.TimeJitter(
                                std=augmentation_kwargs["std_t"],
                                clip_negative=False,
                                sort_timestamps=True,
                            ),
                            OtherAudioUniformNoiseTransform(
                                sensor_size=self.not_downsampled_sensor_size,
                                n=(0, augmentation_kwargs["num_noise_events"]),
                            ),
                        ]
                    )
                else:
                    augmentation_kwargs = self.aug_kwargs
                    if "num_noise_events" in augmentation_kwargs.keys():
                        self.transforms.extend(
                            [
                                OtherAudioUniformNoiseTransform(
                                    sensor_size=self.not_downsampled_sensor_size,
                                    n=(0, augmentation_kwargs["num_noise_events"]),
                                ),
                            ]
                        )
                    if "drop_event_prob" in augmentation_kwargs.keys():
                        self.transforms.extend(
                            [
                                tonic.transforms.DropEvent(
                                    p=augmentation_kwargs["drop_event_prob"]
                                ),
                            ]
                        )
                    if "std_t" in augmentation_kwargs.keys():
                        self.transforms.extend(
                            [
                                tonic.transforms.TimeJitter(
                                    std=augmentation_kwargs["std_t"],
                                    clip_negative=False,
                                    sort_timestamps=True,
                                ),
                            ]
                        )
                    if "var_x" in augmentation_kwargs.keys():
                        self.transforms.extend(
                            [
                                Jitter1D(
                                    sensor_size=self.not_downsampled_sensor_size,
                                    var=augmentation_kwargs["var_x"],
                                ),
                            ]
                        )
                    if "max_drop_chunk" in augmentation_kwargs.keys():
                        self.transforms.extend(
                            [
                                DropEventChunk(
                                    p=0.3,
                                    max_drop_size=augmentation_kwargs["max_drop_chunk"],
                                ),
                            ]
                        )
                    if "time_skew" in augmentation_kwargs.keys():
                        self.transforms.extend(
                            [
                                tonic.transforms.TimeSkew(
                                    coefficient=(
                                        1 / augmentation_kwargs["time_skew"],
                                        augmentation_kwargs["time_skew"],
                                    ),
                                    offset=0,
                                ),
                            ]
                        )
            else:
                raise Exception(f"Dataset {self.dataset_name} is not supported here...")

        if self.num_crop_pixel > 0:
            self.transforms.extend(
                [
                    Custom1DCropTransform(
                        sensor_size=self.not_downsampled_sensor_size,
                        num_crop_pixel=num_crop_pixel,
                    ),
                ]
            )
        self.transforms.extend(
            [
                tonic.transforms.Downsample(spatial_factor=spat_ds_fac),
                tonic.transforms.ToFrame(
                    sensor_size=self.sensor_size, time_window=self.time_window
                ),  # n_time_bins=nb_bins),
            ]
        )

        if self.denoise:
            filter_time = 5000
            if self.dataset_name == "SHD":
                denoise_transform = CustomAudioDenoiseTransform(
                    filter_time=filter_time, mode=self.denoise_mode
                )
            else:
                raise Exception(
                    f"For dataset {self.dataset_name} denoising is not implemented yet."
                )

        if self.denoise_mode == "tonic":
            self.transforms = (
                self.transforms[:-2] + [denoise_transform] + self.transforms[-2:]
            )
        elif self.denoise_mode == "thr":
            self.transforms.extend([denoise_transform])
        else:
            raise Exception(f"Denoise mode {self.denoise_mode} is not supported.")

        transform = tonic.transforms.Compose(self.transforms)

        self.data_dict = {}
        if dataset_name == "SHD":
            if self.val_on_test:
                self.data_dict["train_set"] = self.dataset(
                    self.save_path, transform=transform, train=True
                )
                self.data_dict["valid_set"] = self.dataset(
                    self.save_path, transform=transform, train=False
                )
            else:
                og_train_set = self.dataset(
                    self.save_path, transform=transform, train=True
                )
                train_size = int(0.8 * len(og_train_set))  # 80% for training
                val_size = len(og_train_set) - train_size  # 20% for validation
                generator = torch.Generator().manual_seed(42)
                train_dataset, val_dataset = random_split(
                    og_train_set, [train_size, val_size], generator=generator
                )
                self.data_dict["train_set"] = train_dataset
                self.data_dict["valid_set"] = val_dataset
                self.data_dict["test_set"] = self.dataset(
                    self.save_path, transform=transform, train=False
                )
        else:
            raise Exception(f"Dataset {self.dataset_name} is unknown.")

    def apply_augmentation(self, task_type, augmentation_kwargs):
        self.augmented_datasets = []
        if not isinstance(task_type, list):
            if task_type == "audio":
                self.create_augmented_data(
                    "audio_uniform_noise", augmentation_kwargs["num_noise_events"]
                )
                self.create_augmented_data(
                    "drop_event", augmentation_kwargs["drop_event_prob"]
                )
                self.create_augmented_data("time_jitter", augmentation_kwargs["std_t"])
                self.create_augmented_data(
                    "audio_spatial_jitter", augmentation_kwargs["var_x"]
                )
            elif task_type == "visual":
                self.create_augmented_data(
                    "random_crop", augmentation_kwargs["size_share"]
                )
                self.create_augmented_data(
                    "uniform_noise", augmentation_kwargs["num_noise_events"]
                )
                self.create_augmented_data(
                    "drop_event", augmentation_kwargs["drop_event_prob"]
                )
                self.create_augmented_data("time_jitter", augmentation_kwargs["std_t"])
                self.create_augmented_data(
                    "spatial_jitter", augmentation_kwargs["var_x"]
                )
            elif task_type == "0":
                self.create_augmented_data(
                    "random_crop", augmentation_kwargs["size_share"]
                )
            elif task_type == "1":
                self.create_augmented_data(
                    "uniform_noise", augmentation_kwargs["num_noise_events"]
                )
            elif task_type == "2":
                self.create_augmented_data(
                    "drop_event", augmentation_kwargs["drop_event_prob"]
                )
            elif task_type == "3":
                self.create_augmented_data("time_jitter", augmentation_kwargs["std_t"])
            elif task_type == "4":
                self.create_augmented_data(
                    "spatial_jitter", augmentation_kwargs["var_x"]
                )
            elif task_type == "5":
                self.create_augmented_data(
                    "audio_uniform_noise", augmentation_kwargs["num_noise_events"]
                )
            elif task_type == "6":
                self.create_augmented_data(
                    "audio_spatial_jitter", augmentation_kwargs["var_x"]
                )
            else:
                raise Exception(f"task_type {task_type} is unknown.")
        else:
            while len(task_type) > 0:
                if "0" in task_type:
                    self.create_augmented_data(
                        "random_crop", augmentation_kwargs["size_share"]
                    )
                    task_type.remove("0")
                if "1" in task_type:
                    self.create_augmented_data(
                        "uniform_noise", augmentation_kwargs["num_noise_events"]
                    )
                    task_type.remove("1")
                if "2" in task_type:
                    self.create_augmented_data(
                        "drop_event", augmentation_kwargs["drop_event_prob"]
                    )
                    task_type.remove("2")
                if "3" in task_type:
                    self.create_augmented_data(
                        "time_jitter", augmentation_kwargs["std_t"]
                    )
                    task_type.remove("3")
                if "4" in task_type:
                    self.create_augmented_data(
                        "spatial_jitter", augmentation_kwargs["var_x"]
                    )
                    task_type.remove("4")
                if "5" in task_type:
                    self.create_augmented_data(
                        "audio_uniform_noise", augmentation_kwargs["num_noise_events"]
                    )
                    task_type.remove("5")
                if "6" in task_type:
                    self.create_augmented_data(
                        "audio_spatial_jitter", augmentation_kwargs["var_x"]
                    )
                    task_type.remove("6")

    def create_augmented_data(self, augmentation_type, args):
        std_transforms = self.transforms
        if augmentation_type == "random_crop":
            aug_fn = self.aug_random_crop
            del std_transforms[-1]  # ToFrame is redefined during augmentation
        elif augmentation_type == "uniform_noise":
            aug_fn = self.aug_uniform_noise
        elif augmentation_type == "audio_uniform_noise":
            aug_fn = self.aug_audio_uniform_noise
        elif augmentation_type == "drop_event":
            aug_fn = self.aug_drop_event
        elif augmentation_type == "time_jitter":
            aug_fn = self.aug_time_jitter
        elif augmentation_type == "spatial_jitter":
            aug_fn = self.aug_spatial_jitter
        elif augmentation_type == "audio_spatial_jitter":
            aug_fn = self.aug_audio_spatial_jitter
        else:
            raise Exception(f"augmentation type {augmentation_type} is unknown.")
        aug_transforms = aug_fn(args)
        aug_transforms.extend(std_transforms)
        transform = tonic.transforms.Compose(aug_transforms)
        if self.dataset_name == "SHD":
            aug_data = self.dataset(self.save_path, transform=transform, train=True)
        else:
            raise Exception(f"Dataset {self.dataset_name} is unknown.")

        self.augmented_datasets.append(aug_data)

    def aug_random_crop(self, size_share):
        aug_sensor_size = (
            int(size_share * self.sensor_size[0]),
            int(size_share * self.sensor_size[1]),
            self.sensor_size[2],
        )
        aug_transforms = [
            tonic.transforms.RandomCrop(
                sensor_size=self.sensor_size,
                target_size=(aug_sensor_size[0], aug_sensor_size[1]),
            ),
            tonic.transforms.ToFrame(
                sensor_size=aug_sensor_size, n_time_bins=self.nb_bins
            ),
        ]
        return aug_transforms

    def aug_uniform_noise(self, num_noise_events):
        return [
            tonic.transforms.UniformNoise(
                sensor_size=self.sensor_size, n=num_noise_events
            )
        ]

    def aug_audio_uniform_noise(self, num_noise_events):
        return [
            CustomAudioUniformNoiseTransform(
                sensor_size=self.sensor_size, nb_noise_events=num_noise_events
            )
        ]

    @staticmethod
    def aug_drop_event(drop_event_prob):
        return [tonic.transforms.DropEvent(p=drop_event_prob)]

    @staticmethod
    def aug_time_jitter(std_t):
        return [tonic.transforms.TimeJitter(std=std_t, clip_negative=True)]

    def aug_spatial_jitter(self, var_x, var_y=None):
        if var_y is None:
            var_y = var_x
        return [
            tonic.transforms.SpatialJitter(
                sensor_size=self.sensor_size,
                var_x=var_x,
                var_y=var_y,
                clip_outliers=True,
            )
        ]

    def aug_audio_spatial_jitter(self, var_x):
        return [
            CustomAudioSpatialJitterTransform(
                sensor_size=self.sensor_size, var_x=var_x, clip_outliers=True
            )
        ]
