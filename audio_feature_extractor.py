"""Feature extractor script for the ESC-50 audio dataset,  using the librosa library."""

# %% [markdown]

# # Extração de *Features* de dados de áudio
# Nesse documento, se encontram as *features* extraídas de um subconjunto dos
# dados de áudio.
#
# A maioria das *features* selecionadas é proveniente do spectrograma de cada
# exemplo do conjunto de dados utilizado.
#
# Principais referências utilizadas:
#
# - https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.332.989&rep=rep1&type=pdf  # noqa: E501
# - https://librosa.org/doc/main/feature.html  # noqa: E501
# - https://towardsdatascience.com/how-i-understood-what-features-to-consider-while-training-audio-files-eedfb6e9002b  # noqa: E501

# %% [markdown]
# # Imports

# %%
import librosa
import librosa.display
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import re
import warnings

# %% [markdown]
# # Function definitions


# %%


def extract_feature_means(audio_file_path: str, verbose: bool = True) -> pd.DataFrame:
    """Extract audio features of a given file."""
    if verbose:
        print("File:", audio_file_path)
    number_of_mfcc = 20
    n_fft = 2048  # FFT window size
    hop_length = 512  # number audio of frames between STFT columns

    if verbose:
        print("1.Importing file with librosa...")
    try:
        y, sr = librosa.load(audio_file_path)
    except Exception as e:
        print(e)
        return None
    # Trim leading and trailing silence from an audio signal
    signal, _ = librosa.effects.trim(y)

    if verbose:
        print("2.Fourier transform...")
    # Short-time Fourier transform (STFT)
    d_audio = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))

    if verbose:
        print("3.Spectrogram...")
    # Convert an amplitude spectrogram to Decibels-scaled spectrogram.
    db_audio = librosa.amplitude_to_db(d_audio, ref=np.max)

    if verbose:
        print("4.Mel spectograms...")
    s_audio = librosa.feature.melspectrogram(signal, sr=sr)
    s_db_audio = librosa.amplitude_to_db(s_audio, ref=np.max)

    if verbose:
        print("6.Harmonics and perceptrual...")
    y_harm, y_perc = librosa.effects.hpss(signal)

    if verbose:
        print("7.Spectral centroid...")
    # Calculate the Spectral Centroids
    spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sr)[0]
    spectral_centroids_delta = librosa.feature.delta(spectral_centroids)
    spectral_centroids_accelerate = librosa.feature.delta(spectral_centroids, order=2)

    if verbose:
        print("8.Chroma features...")
    chromagram = librosa.feature.chroma_stft(signal, sr=sr, hop_length=hop_length)

    if verbose:
        print("9.Tempo BPM...")
    tempo_y, _ = librosa.beat.beat_track(signal, sr=sr)

    if verbose:
        print("10.Spectral rolloff...")
    # Spectral RollOff Vector
    spectral_rolloff = librosa.feature.spectral_rolloff(signal, sr=sr)[0]
    # spectral flux
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
    # Spectral Bandwidth
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(signal, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(signal, sr=sr, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(signal, sr=sr, p=4)[0]

    audio_features = {
        "file_name": audio_file_path,
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(signal)[0]),
        "zero_crossings": np.sum(librosa.zero_crossings(signal, pad=False)),
        "spectrogram": np.mean(db_audio[0]),
        "mel_spectrogram": np.mean(s_db_audio[0]),
        "harmonics": np.mean(y_harm),
        "perceptual_shock_wave": np.mean(y_perc),
        "spectral_centroids": np.mean(spectral_centroids),
        "spectral_centroids_delta": np.mean(spectral_centroids_delta),
        "spectral_centroids_accelerate": np.mean(spectral_centroids_accelerate),
        "chroma1": np.mean(chromagram[0]),
        "chroma2": np.mean(chromagram[1]),
        "chroma3": np.mean(chromagram[2]),
        "chroma4": np.mean(chromagram[3]),
        "chroma5": np.mean(chromagram[4]),
        "chroma6": np.mean(chromagram[5]),
        "chroma7": np.mean(chromagram[6]),
        "chroma8": np.mean(chromagram[7]),
        "chroma9": np.mean(chromagram[8]),
        "chroma10": np.mean(chromagram[9]),
        "chroma11": np.mean(chromagram[10]),
        "chroma12": np.mean(chromagram[11]),
        "tempo_bpm": tempo_y,
        "spectral_rolloff": np.mean(spectral_rolloff),
        "spectral_flux": np.mean(onset_env),
        "spectral_bandwidth_2": np.mean(spectral_bandwidth_2),
        "spectral_bandwidth_3": np.mean(spectral_bandwidth_3),
        "spectral_bandwidth_4": np.mean(spectral_bandwidth_4),
    }

    # extract mfcc feature
    mfcc_df = extract_mfcc_feature_means(
        audio_file_path, signal, sample_rate=sr, number_of_mfcc=number_of_mfcc
    )

    df = pd.DataFrame.from_records(data=[audio_features])

    df = pd.merge(df, mfcc_df, on="file_name")

    if verbose:
        print("DONE:", audio_file_path)
    return df


def extract_mfcc_feature_means(
    audio_file_name: str, signal: np.ndarray, sample_rate: int, number_of_mfcc: int
) -> pd.DataFrame:
    """Extract MFCCs from a given audio file."""
    mfcc_alt = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=number_of_mfcc)
    delta = librosa.feature.delta(mfcc_alt)
    accelerate = librosa.feature.delta(mfcc_alt, order=2)

    mfcc_features = {
        "file_name": audio_file_name,
    }

    for i in range(0, number_of_mfcc):
        # mfcc coefficient
        key_name = "".join(["mfcc", str(i)])
        mfcc_value = np.mean(mfcc_alt[i])
        mfcc_features.update({key_name: mfcc_value})

        # mfcc delta coefficient
        key_name = "".join(["mfcc_delta_", str(i)])
        mfcc_value = np.mean(delta[i])
        mfcc_features.update({key_name: mfcc_value})

        # mfcc accelerate coefficient
        key_name = "".join(["mfcc_accelerate_", str(i)])
        mfcc_value = np.mean(accelerate[i])
        mfcc_features.update({key_name: mfcc_value})

    df = pd.DataFrame.from_records(data=[mfcc_features])
    return df


# %% [markdown]
# # Extracting raw audio from .mp4 files
# Using librosa

# %%
DATA_DIR: str = "./ESC-50/audio"
EXTENSION: str = ".wav"
fnames: list = sorted(
    [
        os.path.join(root, file)
        for root, _, files in os.walk(DATA_DIR)
        for file in files
        if file.endswith(EXTENSION)
    ]
)
pattern: str = r".*\.wav"
matches: list = [re.fullmatch(pattern, fname) for fname in fnames]
if len(matches) != len(fnames):
    raise ValueError("check fname patterns")
rows = np.array(
    [
        [
            fnames[i],
        ]
        for i in range(len(matches))
    ]
)
columns = ["fname"]
print("Number of identified audio files:", len(fnames))

# %% [markdown]
# ## Creating DataFrame with filenames (audios dataset)
# ### Loading filenames into DataFrame

# %%
audios = pd.DataFrame(data=rows, columns=columns)

# %% [markdown]
# ### Extracting features (parallel)

# %%
print("Processing", len(audios), "audios...")
result = []
n_processes = 1
iterable = list(audios["fname"])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with mp.Pool(processes=n_processes) as pool:
        result = pool.map(
            extract_feature_means,
            iterable=iterable,
            chunksize=len(iterable) // n_processes,
        )
        pool.close()
        pool.join()

# %%
print("Done processing audios. Concatenating and writing to output file...")
for idx, i in enumerate(result):
    if i is None:
        del result[idx]
audios_means = pd.concat(result)
audios_means.loc[:, "file_name"] = audios_means.loc[:, "file_name"].transform(
    lambda x: x[15:]
)
output_path = "./features_means.csv"
audios_means.to_csv(output_path, index=False)

# %% [markdown]
# ## Visualizing DataFrame with features

# %%
