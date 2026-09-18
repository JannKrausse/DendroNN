import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import numpy as np
import string
from pathlib import Path
import shutil
import subprocess
import tempfile


NEUROMORSE_REPOSITORY = "https://github.com/Ben-E-Walters/NeuroMorse.git"


def _require_git_lfs():
    if shutil.which("git") is None:
        raise RuntimeError(
            "The NeuroMorse project is missing and Git is not installed. "
            "Install Git before running a NeuroMorse experiment."
        )
    if shutil.which("git-lfs") is None:
        raise RuntimeError(
            "The NeuroMorse project is missing and Git LFS is not installed. "
            "Install Git LFS and run 'git lfs install' before running a "
            "NeuroMorse experiment."
        )


def _ensure_neuromorse_project():
    project_dir = Path(__file__).resolve().parent / "NeuroMorse"
    if (
        project_dir.is_dir()
        and (project_dir / "README.md").is_file()
        and (project_dir / "data").is_dir()
    ):
        return project_dir

    _require_git_lfs()
    project_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_project_dir = Path(temporary_directory) / "NeuroMorse"
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    NEUROMORSE_REPOSITORY,
                    str(temporary_project_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(temporary_project_dir), "lfs", "pull"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            details = getattr(error, "stderr", "") or str(error)
            raise RuntimeError(
                "The NeuroMorse project is missing and could not be cloned. "
                f"Install Git and ensure network access to {NEUROMORSE_REPOSITORY}. "
                f"Details: {details.strip()}"
            ) from error

        if project_dir.exists():
            raise RuntimeError(
                f"NeuroMorse directory exists but is incomplete: {project_dir}. "
                "Remove it or provide the complete upstream NeuroMorse project."
            )
        temporary_project_dir.rename(project_dir)

    return project_dir


def _ensure_corpus_file():
    project_dir = _ensure_neuromorse_project()
    corpus_path = project_dir / "data" / "corpus.txt"
    if not corpus_path.is_file() or corpus_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"The NeuroMorse project was cloned, but its corpus is missing: {corpus_path}"
        )
    _reject_lfs_pointer(corpus_path, project_dir)
    return corpus_path


def _reject_lfs_pointer(path, project_dir):
    with path.open("rb") as corpus_file:
        first_line = corpus_file.readline()
    if first_line.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"NeuroMorse data is a Git LFS pointer instead of the actual file: "
            f"{path}. Run 'git lfs pull' in {project_dir}."
        )


def create_neuromorse_dataset():
    corpus_path = _ensure_corpus_file()
    # Create a Morse code dataset.
    Morse_Dict = {
        "a": ".-",
        "b": "-...",
        "c": "-.-.",
        "d": "-..",
        "e": ".",
        "f": "..-.",
        "g": "--.",
        "h": "....",
        "i": "..",
        "j": ".---",
        "k": "-.-",
        "l": ".-..",
        "m": "--",
        "n": "-.",
        "o": "---",
        "p": ".--.",
        "q": "--.-",
        "r": ".-.",
        "s": "...",
        "t": "-",
        "u": "..-",
        "v": "...-",
        "w": ".--",
        "x": "-..-",
        "y": "-.--",
        "z": "--..",
        "1": ".----",
        "2": "..---",
        "3": "...--",
        "4": "....-",
        "5": ".....",
        "6": "-....",
        "7": "--...",
        "8": "---..",
        "9": "----.",
        "0": "-----",
        ".": ".-.-.-",
        ",": "--..--",
        "?": "..--..",
        ":": "---...",
        "'": ".----.",
        "-": "-....-",
        "/": "-..-.",
        "(": "-.--.",
        ")": "-.--.-",
        '"': ".-..-.",
        "=": "-...-",
        ";": "-.-.-.",
        "$": "...-..-",
    }

    space = 5  # Time between dots and dashes.
    letter_space = 10  # Time between letters
    word_space = 15  # Time between consecutive words

    SpikeArray = []
    New_Dataset = []
    SpikeDict = {}

    # Convert each Morse character into spike array
    for key in Morse_Dict.keys():
        time = 0
        b = []
        for i, ch in enumerate(Morse_Dict[key]):
            if ch == ".":
                channel = 0
            elif ch == "-":
                channel = 1
            b.append((time, channel, 1))
            time = time + space + 1
        SpikeArray = np.array(b, dtype=[("t", "<f4"), ("x", "<f4"), ("p", "<f4")])
        New_Dataset.append((SpikeArray, key))
        SpikeDict[key] = SpikeArray

    # Create Morse Spike Dictionary
    Morse_Spike_Dict = {}
    for key in Morse_Dict.keys():
        time = 0
        b = []
        for i, ch in enumerate(Morse_Dict[key]):
            if ch == ".":
                channel = 0
            elif ch == "-":
                channel = 1
            b.append((time, channel, 1))
            time = time + space + 1
        SpikeArray = np.array(b, dtype=[("t", "<f4"), ("x", "<f4"), ("p", "<f4")])
        Morse_Spike_Dict[key] = SpikeArray

    # List of top 50 words
    Top50List = [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
    ]
    WordDataset = []

    for idx, word in enumerate(Top50List):
        time = 0
        list = []
        for i, ch in enumerate(word):
            Spikes = np.copy(SpikeDict[ch])
            Spikes["t"] += time
            time = Spikes[-1][0] + (
                1 + letter_space
            )  ################################## ACCORDING TO MY ANALYSIS THIS DIFFERS FROM THE CONSTRUCTION OF THE TEST DATA!!!################################## ACCORDING TO MY ANALYSIS THIS DIFFERS FROM THE CONSTRUCTION OF THE TEST DATA!!!
            # time = Spikes[-1][0]+letter_space  # MY CORRECTION, SEE REASON ABOVE
            list.append(Spikes)

        WholeArray = np.concatenate(list)
        WordDataset.append((WholeArray, word))

    TrainSpikeDataset = []
    training_labels = []
    num_channels = 2
    # Map each word to its index
    word_to_index = {word: idx for idx, word in enumerate(Top50List)}

    for i in range(50):
        data, label = WordDataset[i]
        training_labels.append(word_to_index.get(label))
        data_neuro = torch.zeros((int(data[-1][0]) + 1 + word_space, num_channels))
        for idx in data:
            # idx[0] = spike time
            # idx[1] = 0 if dot 1 if dash
            data_neuro[int(idx[0]), int(idx[1])] = 1
        TrainSpikeDataset.append(data_neuro)

    # Load a little bit of the corpus (test data)
    with corpus_path.open("r", encoding="utf8") as f:
        corpus = f.read().lower().split()

    # Select a random subset of words from the corpus, bit slow to load the whole thing.
    subset_size = 100000  # Adjust this number as needed
    # subset_indices = random.sample(range(len(corpus)), subset_size)
    test_subset = corpus[0:subset_size]

    # Create a translation table
    translator = str.maketrans("", "", string.punctuation)

    # Remove punctuation from each word
    cleaned_test_subset = [word.translate(translator) for word in test_subset]

    # Remove any empty strings resulting from removing punctuation-only words
    cleaned_test_subset = [word for word in cleaned_test_subset if word]

    # Constants (same as in the training script)
    num_channels = 2  # Dots and dashes
    word_space = 15  # Spacing between words
    letter_space = 10  # Spacing between letters

    # Map each word to its index or OOV label
    word_to_index = {word: idx for idx, word in enumerate(Top50List)}
    OOV_label = 50  # Label for OOV words

    # Create spike trains for each word in the test subset
    TestSpikeDataset = []
    test_labels = []

    for word in cleaned_test_subset:
        # Initialize spike train data and label
        data = []
        label = word_to_index.get(word, OOV_label)  # Assign label or OOV label
        time = 0

        # Generate spike train for the word
        for ch in word:
            if ch in Morse_Spike_Dict:
                # Retrieve the spike train for the character
                char_spikes = np.copy(Morse_Spike_Dict[ch])

                # Adjust the spike times to account for the current time
                char_spikes["t"] += time

                # Convert to the desired structured array format
                char_spikes_t_x = np.zeros(
                    len(char_spikes), dtype=[("t", "<f4"), ("x", "<f4")]
                )
                char_spikes_t_x["t"] = char_spikes["t"]
                char_spikes_t_x["x"] = char_spikes["x"]

                # Append the character's spike train to the word's data
                data.extend(char_spikes_t_x)

                # Update time for the next character
                # time = char_spikes['t'][-1] + letter_space  ################################## ACCORDING TO MY ANALYSIS THIS DIFFERS FROM THE CONSTRUCTION OF THE TRAINING DATA!!!
                time = char_spikes["t"][-1] + (
                    1 + letter_space
                )  # MY CORRECTION, SEE REASON ABOVE
            else:
                # Skip characters not in Morse_Spike_Dict
                continue

        # Add spacing for the next word
        if data:
            # Append the word's spike train to the dataset
            data = np.array(data, dtype=[("t", "<f4"), ("x", "<f4")])

            # Create a binary spike train tensor
            max_time = int(data[-1]["t"]) + 1 + word_space  # Add word spacing
            data_neuro = torch.zeros((max_time, num_channels))  # Time x Channels

            # Populate the spike train tensor
            for idx in data:
                data_neuro[int(np.floor(idx["t"])), int(idx["x"])] = 1

            # Append the processed word's spike train and label
            TestSpikeDataset.append(data_neuro)
            test_labels.append(label)

    # Determine the maximum length of spike trains
    max_length = max(tensor.shape[0] for tensor in TrainSpikeDataset)

    # Pad all tensors to max_length
    for i in range(len(TrainSpikeDataset)):
        padding = max_length - TrainSpikeDataset[i].shape[0]
        TrainSpikeDataset[i] = F.pad(
            TrainSpikeDataset[i], (0, 0, 0, padding), mode="constant", value=0
        )  # Pad at the end

    max_length = max(tensor.shape[0] for tensor in TestSpikeDataset)

    # Pad all tensors to max_length
    for i in range(len(TestSpikeDataset)):
        padding = max_length - TestSpikeDataset[i].shape[0]
        TestSpikeDataset[i] = F.pad(
            TestSpikeDataset[i], (0, 0, 0, padding), mode="constant", value=0
        )  # Pad at the end

    ############################################ MY ADDITION: basically code for test data, but for the validation.txt
    # Load validation.txt
    validation_path = corpus_path.parent / "Validation" / "validation.txt"
    _reject_lfs_pointer(validation_path, corpus_path.parent.parent)
    with validation_path.open("r", encoding="utf8") as f:
        validation_subset = f.read().lower().split()

    # Remove punctuation from each word
    cleaned_validation_subset = [
        word.translate(translator) for word in validation_subset
    ]

    # Remove any empty strings resulting from removing punctuation-only words
    cleaned_validation_subset = [word for word in cleaned_validation_subset if word]

    # Create spike trains for each word in the validation subset
    ValidationSpikeDataset = []
    validation_labels = []

    for word in cleaned_validation_subset:
        # Initialize spike train data and label
        data = []
        label = word_to_index.get(word, OOV_label)  # Assign label or OOV label
        time = 0

        # Generate spike train for the word
        for ch in word:
            if ch in Morse_Spike_Dict:
                # Retrieve the spike train for the character
                char_spikes = np.copy(Morse_Spike_Dict[ch])

                # Adjust the spike times to account for the current time
                char_spikes["t"] += time

                # Convert to the desired structured array format
                char_spikes_t_x = np.zeros(
                    len(char_spikes), dtype=[("t", "<f4"), ("x", "<f4")]
                )
                char_spikes_t_x["t"] = char_spikes["t"]
                char_spikes_t_x["x"] = char_spikes["x"]

                # Append the character's spike train to the word's data
                data.extend(char_spikes_t_x)

                # Update time for the next character
                # time = char_spikes['t'][-1] + letter_space  ################################## ACCORDING TO MY ANALYSIS THIS DIFFERS FROM THE CONSTRUCTION OF THE TRAINING DATA!!!
                time = char_spikes["t"][-1] + (
                    1 + letter_space
                )  # MY CORRECTION, SEE REASON ABOVE
            else:
                # Skip characters not in Morse_Spike_Dict
                continue

        # Add spacing for the next word
        if data:
            # Append the word's spike train to the dataset
            data = np.array(data, dtype=[("t", "<f4"), ("x", "<f4")])

            # Create a binary spike train tensor
            max_time = int(data[-1]["t"]) + 1 + word_space  # Add word spacing
            data_neuro = torch.zeros((max_time, num_channels))  # Time x Channels

            # Populate the spike train tensor
            for idx in data:
                data_neuro[int(np.floor(idx["t"])), int(idx["x"])] = 1

            # Append the processed word's spike train and label
            ValidationSpikeDataset.append(data_neuro)
            validation_labels.append(label)

    # Determine the maximum length of spike trains
    max_length = max(tensor.shape[0] for tensor in ValidationSpikeDataset)

    # Pad all tensors to max_length
    for i in range(len(ValidationSpikeDataset)):
        padding = max_length - ValidationSpikeDataset[i].shape[0]
        ValidationSpikeDataset[i] = F.pad(
            ValidationSpikeDataset[i], (0, 0, 0, padding), mode="constant", value=0
        )  # Pad at the end

    # Stack all spike train tensors into a single tensor
    inputs = torch.stack(
        TrainSpikeDataset
    )  # Shape: [num_words, max_length, num_channels]
    test_inputs = torch.stack(TestSpikeDataset)
    validation_inputs = torch.stack(ValidationSpikeDataset)

    # Convert labels to a torch tensor
    labels = torch.tensor(training_labels)  # Shape: [num_words]
    test_labels = torch.tensor(test_labels)
    validation_labels = torch.tensor(validation_labels)

    # Create a TensorDataset
    dataset = TensorDataset(inputs, labels)
    test_dataset = TensorDataset(test_inputs, test_labels)
    validation_dataset = TensorDataset(validation_inputs, validation_labels)

    return dataset, test_dataset, validation_dataset
