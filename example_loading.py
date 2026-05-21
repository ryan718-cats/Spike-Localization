import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pytorch_lightning as pl

from ieeg_load_preprocess import (
    CHANNELS_TO_INCLUDE,
    CURRENT_CHANNEL_ORDER,
    NEW_CHANNEL_ORDER,
    TARGET_FS as FQ,
    get_ieeg_data,
    preprocess_scalp_segment,
)
from sleeplib.Resnet_15.model import FineTuning
from sleeplib.config import Config
from sleeplib.transforms import extremes_remover
from sleeplib.montages import con_combine_montage

BASE_DIR = r"C:\Users\ryanc\Downloads\Conrad_Lab\progress_sheet"
SLEEPLIB_PARENT = BASE_DIR
CKPT_PATH = r"C:\Users\ryanc\Downloads\Conrad_Lab\progress_sheet\models\1s-round11-hardmine-chan_weights-v1.ckpt"

# Credentials from environment only — do not hardcode passwords in this file.
IEEG_USERNAME = os.environ.get("IEEG_USERNAME", "")
IEEG_PASSWORD = os.environ.get("IEEG_PASSWORD", "")

from spikenet_timing import (
    SPIKENET_DELAY_SEC,
    SPIKENET_STEP_SAMPLES as STEP_SAMPLES,
    SPIKENET_STEP_SEC as STEP_SEC,
)

WINDOW_SIZE_SEC = 1

SPIKE_THRESH = 0.43

sys.path.insert(0, SLEEPLIB_PARENT)

class ContinousToSnippetDataset(Dataset):
    def __init__(self, signal_data, montage=None, transform=None, Fq=128, window_size=1, step=8):
        signal = np.where(np.isnan(signal_data), 0, signal_data).astype(np.float32)
        signal = torch.FloatTensor(signal)
        self.snippets = signal.unfold(dimension=1, size=window_size * Fq, step=step).permute(1, 0, 2)
        self.transform = transform
        self.montage = montage
    def __len__(self):
        return self.snippets.shape[0]
    def _preprocess(self, x):
        if self.montage is not None:
            x = self.montage(x)
        if self.transform is not None:
            x = self.transform(x)
        x = x / (np.quantile(np.abs(x), q=0.95, axis=-1, keepdims=True) + 1e-8)
        return torch.FloatTensor(np.array(x, copy=True))
    def __getitem__(self, idx):
        x = self.snippets[idx, :, :]
        x = self._preprocess(x)
        return x, 0

channels_to_include = CHANNELS_TO_INCLUDE
current_channel_order = CURRENT_CHANNEL_ORDER
new_channel_order = NEW_CHANNEL_ORDER

def format_hhmmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"

config = Config()
transform_test = transforms.Compose([extremes_remover(signal_max=2000, signal_min=20)])
montage_fn = con_combine_montage()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = FineTuning.load_from_checkpoint(
    CKPT_PATH,
    lr=config.LR,
    head_dropout=config.HEAD_DROPOUT,
    n_channels=config.N_CHANNELS,
    n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH,
    map_location=torch.device(device),
)

trainer = pl.Trainer(
    fast_dev_run=False,
    enable_progress_bar=True,
    accelerator="gpu" if device == "cuda" else "cpu",
    devices=1,
    strategy="auto",
)


from itertools import combinations
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


def predict_sn2_trace(reordered_data: np.ndarray) -> np.ndarray:
    """Run SpikeNet2 on preprocessed/reordered segment; return flat SN2 scores."""
    dataset = ContinousToSnippetDataset(
        signal_data=reordered_data.T,
        montage=montage_fn,
        transform=transform_test,
        window_size=WINDOW_SIZE_SEC,
        step=STEP_SAMPLES,
        Fq=FQ,
    )
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    preds = trainer.predict(model, loader)
    return np.concatenate(preds).astype(float).flatten()


def predict_sn2_for_ieeg_window(
    ieeg_file: str,
    start_sec: float,
    end_sec: float,
) -> tuple[np.ndarray, float]:
    """
    Download [start_sec, end_sec], lab-preprocess, run SpikeNet2.

    Returns (sn2_scores, chunk_start_sec) for use with spikenet_timing.prob_at_saved_timestamp.
    """
    chunk_start = float(start_sec)
    df, fs = get_iEEG_data(
        username=IEEG_USERNAME,
        password=IEEG_PASSWORD,
        iEEG_filename=ieeg_file,
        start_time_usec=int(chunk_start * 1e6),
        stop_time_usec=int(end_sec * 1e6),
        select_electrodes=channels_to_include,
    )
    reordered, _, _ = preprocess_scalp_segment(
        df.values, list(df.columns), fs, target_fs=FQ
    )
    return predict_sn2_trace(reordered), chunk_start


def localization(ieeg_file, start_sec):
    ieeg_file_name = ieeg_file
    center_sec = float(start_sec)
    start_sec = center_sec - 0.5
    end_sec = center_sec + 0.5

    try:
        df, fs = get_iEEG_data(
            username=IEEG_USERNAME,
            password=IEEG_PASSWORD,
            iEEG_filename=ieeg_file_name,
            start_time_usec=start_sec * 1e6,
            stop_time_usec=end_sec * 1e6,
            select_electrodes=channels_to_include,
        )
    except Exception as e:
        print(f"Skipping {ieeg_file_name}: could not load from iEEG ({e})")
        return None

    reordered_data, _, _ = preprocess_scalp_segment(
        df.values, list(df.columns), fs, target_fs=FQ
    )
    base_reordered_data = reordered_data.copy()

    brain_region = {
        "left_frontal": ["Fp1", "F3", "F7"],
        "left_central": ["T3", "C3"],
        "left_parietal": ["T5", "P3", "O1"],
        "left_temporal": ["F7", "T3", "T5"],
        "right_frontal": ["Fp2", "F4", "F8"],
        "right_central": ["T4", "C4"],
        "right_parietal": ["T6", "P4", "O2"],
        "right_temporal": ["F8", "T4", "T6"],
        # "left_parasagittal": ["Fp1", "F3", "C3", "P3", "O1"],
        # "right_parasagittal": ["Fp2", "F4", "C4", "P4", "O2"],
        # "left_temporal": ["Fp1", "F7", "T3", "T5", "O1"],
        # "right_temporal": ["Fp2", "F8", "T4", "T6", "O2"],
    }

    def get_max_prob(temp_data):
        SN2 = predict_sn2_trace(temp_data)
        return float(np.max(SN2)) if SN2.size else 0.0

    def generate_eeg_channel(n_samples, fs=128, beta=1.0, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        noise = rng.standard_normal(n_samples)

        f = np.fft.rfftfreq(n_samples, d=1 / fs)
        spectrum = np.fft.rfft(noise)

        f[0] = 1e-6
        spectrum = spectrum / (f ** (beta / 2))

        eeg = np.fft.irfft(spectrum, n=n_samples)

        low = 0.5 / (fs / 2)
        high = 40 / (fs / 2)
        from scipy.signal import butter, filtfilt

        b, a = butter(4, [low, high], btype="band")
        eeg = filtfilt(b, a, eeg)

        eeg = eeg - np.mean(eeg)
        std = np.std(eeg)
        if std > 0:
            eeg = eeg / std

        return eeg

    def replace_channels_with_fake(temp_data, channels_to_modify, fs=128, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        idxs = [new_channel_order.index(ch) for ch in channels_to_modify if ch in new_channel_order]
        n_samples = temp_data.shape[0]

        for idx in idxs:
            temp_data[:, idx] = generate_eeg_channel(
                n_samples=n_samples,
                fs=fs,
                rng=rng,
            )

        return temp_data

    region_scores = {}
    rng = np.random.default_rng()

    for region_name, channels_in_region in brain_region.items():
        temp = base_reordered_data.copy()
        temp = replace_channels_with_fake(
            temp_data=temp,
            channels_to_modify=channels_in_region,
            fs=FQ,
            rng=rng,
        )
        region_scores[region_name] = get_max_prob(temp)

    if not region_scores:
        return {
            "good_regions": [],
            "good_channels": [],
            "lowest_max_prob": None,
        }

    lowest_prob = min(region_scores.values())
    good_regions = [region for region, prob in region_scores.items() if prob == lowest_prob]

    good_channels = []
    for region in good_regions:
        good_channels.extend(brain_region[region])
    good_channels = list(dict.fromkeys(good_channels))

    return {
        "good_regions": good_regions,
        "good_channels": good_channels,
        "lowest_max_prob": lowest_prob,
        "region_probabilities": region_scores
    }


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def normalize_localization_output(out):
    if out is None:
        return [], []

    if isinstance(out, dict):
        regions = out.get("good_regions", [])
        channels = out.get("good_channels", [])
        return regions, channels

    if isinstance(out, tuple):
        if len(out) == 2:
            regions, channels = out
            return regions, channels

    return [], []


results = {}

df = pd.read_csv(r"C:\Users\ryanc\Downloads\selections.csv")

for _, row in df.iterrows():
    filename_full = row["Filename"]
    response = row["Response"]

    if not isinstance(filename_full, str) or "_spike_at_" not in filename_full:
        continue

    base = filename_full.replace(".edf", "")
    file_part, time_part = base.split("_spike_at_")

    ieeg_file = file_part
    spike_time = float(time_part)

    out = localization(ieeg_file, spike_time)

    if out is None:
        continue

    regions, channels = normalize_localization_output(out)

    results[ieeg_file + f"_spike_at_{spike_time}"] = {
        "regions": ensure_list(regions),
        "channels": ensure_list(channels),
        "response": response
    }