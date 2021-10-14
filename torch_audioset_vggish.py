# %% [markdown]
# # Imports and constants

# %%
import copy
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import resampy
import seaborn as sns
import sklearn
import soundfile as sf

# from sklearn.model_selection import KFold
import time
import torch
import torchaudio
from tqdm import tqdm
from typing import Union

import torch_audioset
from torch_audioset.vggish.model import get_vggish
import torch_audioset.params
from torch_audioset.data.torch_input_processing import VGGishLogMelSpectrogram
from torch_audioset.data.tflow_input_processing.vggish_utils import mel_features

# from torch_audioset.data.tflow_input_processing.vggish_utils import vggish_input

DATASET_DIR: str = "./ESC-50/"
AUDIO_DIR: str = DATASET_DIR + "audio/"
META_DIR: str = DATASET_DIR + "meta/"
META_FILE: str = META_DIR + "esc50.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vggish_params = torch_audioset.params.VGGishParams()


# %%


def wav_read(wav_file):
    wav_data, sr = sf.read(wav_file, dtype="int16")
    return wav_data, sr


def waveform_to_examples(data, sample_rate):
    """Convert audio waveform into an array of examples for VGGish.

    Args:
      data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
      sample_rate: Sample rate of data.

    Returns:
      3-D np.array of shape [num_examples, num_frames, num_bands] which represents
      a sequence of examples, each of which contains a patch of log mel
      spectrogram, covering num_frames frames of audio and num_bands mel frequency
      bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
    """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ,
    )

    # Frame features into examples.
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(
        round(vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate)
    )
    example_hop_length = int(
        round(vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate)
    )
    log_mel_examples = mel_features.frame(
        log_mel, window_length=example_window_length, hop_length=example_hop_length
    )
    return log_mel_examples


def wavfile_to_examples(wav_file):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.

    Args:
      wav_file: String path to a file, or a file-like object. The file
      is assumed to contain WAV audio data with signed 16-bit PCM samples.

    Returns:
      See waveform_to_examples.
    """
    wav_data, sr = wav_read(wav_file)
    assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return waveform_to_examples(samples, sr)


# %%
# Define ESC-50 PyTorch dataset


class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotations_file: Union[str, pd.DataFrame],
        audio_dir: str,
        transform=None,
        target_transform=None,
    ):
        if isinstance(annotations_file, str):
            self.metadata = pd.read_csv(annotations_file)
        else:
            self.metadata = pd.DataFrame(annotations_file, copy=True)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.metadata.iloc[idx]["filename"])
        waveform, _ = torchaudio.load(audio_path)
        label = self.metadata.iloc[idx]["target"]
        if self.transform:
            # waveform = self.transform(waveform)
            waveform = self.transform(audio_path)
        if self.target_transform:
            label = self.target_transform(label)
        return waveform, label


# %%
# Define general functions


def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            dataset_size = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs[0, :, None, :, :].float().to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                for input in inputs:
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    input = input[None, :, :].float().to(device)
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(input)
                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    dataset_size += 1
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)
                # consider correct detection if any frame was corretly classified
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


# %%
# Load dataset metadata
files: pd.DataFrame = pd.read_csv(META_FILE)

# %%
# Load sample audio file
y, sr = librosa.core.load(AUDIO_DIR + files.sample(n=1)["filename"].values[0], sr=44100)
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, x_axis="time", y_axis="log", ax=ax)
fig.colorbar(img, ax=ax, format="%+2.f dB")

# %%
# Load DataLoaders
# audio_dataset = ESC50Dataset(META_FILE, AUDIO_DIR, transform=VGGishLogMelSpectrogram())
audio_dataset = ESC50Dataset(META_FILE, AUDIO_DIR, transform=wavfile_to_examples)
data_loaders = [
    {
        "train": torch.utils.data.DataLoader(
            audio_dataset,
            batch_size=1,
            sampler=torch.utils.data.SubsetRandomSampler(
                files.loc[(files["fold"] != i)].index
            ),
        ),
        "val": torch.utils.data.DataLoader(
            audio_dataset,
            batch_size=1,
            sampler=torch.utils.data.SubsetRandomSampler(
                files.loc[(files["fold"] == i)].index
            ),
        ),
    }
    for i in files["fold"].unique()
]

dataset_sizes = [
    {
        "train": len(files[files["fold"] != i]),
        "val": len(files[files["fold"] == i]),
    }
    for i in files["fold"].unique()
]

# %%
# Load VGGish model
model = get_vggish(with_classifier=True, pretrained=True)

# freeze all layers
for param in model.features.parameters():
    param.requires_grad = False
for param in model.embeddings.parameters():
    param.requires_grad = False

# Add output layer
model.classifier[2] = torch.nn.Linear(100, 50, bias=True)
model = model.to(device)

for param in model.classifier.parameters():
    param.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
# optimizer = torch.optim.RMSprop(model.classifier.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=1)
# exp_lr_scheduler = None

# %%
# Train models
try:
    models = pickle.load(open("./models/models.pkl", "rb"))
except FileNotFoundError:
    models = []
    for i in tqdm(range(len(data_loaders))):
        models.append(
            train_model(
                model,
                data_loaders[i],
                dataset_sizes[i],
                criterion,
                optimizer,
                exp_lr_scheduler,
            )
        )
    pickle.dump(models, open("./models/models.pkl", "wb"))

# %%
# Train best model on all folds
try:
    best_model = pickle.load(open("./models/best_model.pkl", "rb"))
except FileNotFoundError:
    best_model_index = np.argmax([m[1].item() for m in models])
    best_model = models[best_model_index]

    for i in tqdm(range(len(data_loaders))):
        best_model_tmp = train_model(
            best_model[0],
            data_loaders[i],
            dataset_sizes[i],
            criterion,
            optimizer,
            exp_lr_scheduler,
        )
        if best_model_tmp[1].item() > best_model[1].item():
            best_model = best_model_tmp
    pickle.dump(best_model, open("./models/best_model.pkl", "wb"))

# %%

# %%

# %%
