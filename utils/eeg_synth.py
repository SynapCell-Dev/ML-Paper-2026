"""Synthetic EEG signal generation for educational visualizations."""

import numpy as np
from scipy import signal as sig


# Canonical EEG frequency bands
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 100),
}

BAND_COLORS = {
    "delta": "#ef4444",
    "theta": "#f59e0b",
    "alpha": "#10b981",
    "beta": "#06b6d4",
    "gamma": "#8b5cf6",
}


def generate_eeg(
    duration_sec: float = 5.0,
    fs: int = 256,
    band_amplitudes: dict | None = None,
    noise_level: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a single-channel synthetic EEG signal.

    Returns a 1-D array of length ``int(duration_sec * fs)``.
    """
    rng = np.random.default_rng(seed)
    if band_amplitudes is None:
        band_amplitudes = {"delta": 1.0, "theta": 0.7, "alpha": 1.0, "beta": 0.4, "gamma": 0.15}

    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    signal_out = np.zeros(n_samples)

    for band, amp in band_amplitudes.items():
        low, high = BANDS[band]
        freq = rng.uniform(low, high)
        phase = rng.uniform(0, 2 * np.pi)
        signal_out += amp * np.sin(2 * np.pi * freq * t + phase)
        # Add a harmonic for realism
        freq2 = rng.uniform(low, high)
        signal_out += amp * 0.3 * np.sin(2 * np.pi * freq2 * t + rng.uniform(0, 2 * np.pi))

    # Pink noise (1/f)
    if noise_level > 0:
        white = rng.standard_normal(n_samples)
        fft_w = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
        freqs[0] = 1.0  # avoid division by zero
        fft_w /= np.sqrt(freqs)
        pink = np.fft.irfft(fft_w, n=n_samples)
        signal_out += noise_level * pink

    return signal_out


def generate_band_signals(
    duration_sec: float = 5.0,
    fs: int = 256,
    band_amplitudes: dict | None = None,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Return individual band components as a dict of arrays."""
    rng = np.random.default_rng(seed)
    if band_amplitudes is None:
        band_amplitudes = {"delta": 1.0, "theta": 0.7, "alpha": 1.0, "beta": 0.4, "gamma": 0.15}

    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    bands = {}
    for band, amp in band_amplitudes.items():
        low, high = BANDS[band]
        freq = rng.uniform(low, high)
        phase = rng.uniform(0, 2 * np.pi)
        bands[band] = amp * np.sin(2 * np.pi * freq * t + phase)
    return bands


def generate_multichannel_eeg(
    n_channels: int = 6,
    duration_sec: float = 5.0,
    fs: int = 256,
    noisy_channels: list[int] | None = None,
    noise_boost: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate multi-channel EEG (shape: n_channels x n_samples).

    Channels listed in *noisy_channels* receive extra high-frequency noise.
    """
    if noisy_channels is None:
        noisy_channels = []
    rng = np.random.default_rng(seed)
    signals = np.zeros((n_channels, int(duration_sec * fs)))
    for ch in range(n_channels):
        signals[ch] = generate_eeg(duration_sec, fs, seed=seed + ch)
        if ch in noisy_channels:
            n = signals.shape[1]
            signals[ch] += noise_boost * rng.standard_normal(n)
    return signals


def generate_seizure_signal(
    duration_sec: float = 60.0,
    fs: int = 256,
    seizure_events: list[tuple[float, float]] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a signal with embedded seizure events.

    Returns (signal, labels) where labels is 0/1.
    """
    if seizure_events is None:
        seizure_events = [(15.0, 25.0), (40.0, 48.0)]

    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs

    # Background: low-amplitude normal EEG
    background = generate_eeg(duration_sec, fs, noise_level=0.2, seed=seed)

    labels = np.zeros(n_samples, dtype=int)
    seizure_component = np.zeros(n_samples)

    for start, end in seizure_events:
        i_start = int(start * fs)
        i_end = min(int(end * fs), n_samples)
        if i_start >= n_samples or i_end <= i_start:
            continue
        labels[i_start:i_end] = 1

        # Seizure: high amplitude, fast frequency, evolving
        dur = end - start
        t_local = np.arange(i_end - i_start) / fs
        envelope = np.sin(np.pi * t_local / dur)  # waxing-waning
        freq_sweep = np.linspace(8, 25, i_end - i_start)
        phase_acc = np.cumsum(2 * np.pi * freq_sweep / fs)
        seizure_component[i_start:i_end] = 3.0 * envelope * np.sin(phase_acc)
        # Add high-frequency burst
        seizure_component[i_start:i_end] += 1.5 * envelope * np.sin(
            2 * np.pi * 40 * t_local + rng.uniform(0, 2 * np.pi)
        )

    signal_out = background + seizure_component
    return signal_out, labels


def compute_stft(
    signal_arr: np.ndarray,
    fs: int = 256,
    nperseg: int = 128,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute STFT. Returns (frequencies, times, Zxx)."""
    if noverlap is None:
        noverlap = nperseg // 2
    frequencies, times, Zxx = sig.stft(signal_arr, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return frequencies, times, np.abs(Zxx)


def apply_gaussian_mask(
    Zxx: np.ndarray,
    center_f: int | None = None,
    center_t: int | None = None,
    sigma_f: float = 5.0,
    sigma_t: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a Gaussian-smoothed mask to an STFT magnitude array.

    Returns (masked_Zxx, mask_matrix) where mask values are in [0, 1].
    0 = fully masked, 1 = fully visible.
    """
    n_freq, n_time = Zxx.shape
    if center_f is None:
        center_f = n_freq // 2
    if center_t is None:
        center_t = n_time // 2

    f_idx = np.arange(n_freq)
    t_idx = np.arange(n_time)
    ff, tt = np.meshgrid(f_idx, t_idx, indexing="ij")

    gauss = np.exp(-((ff - center_f) ** 2) / (2 * sigma_f**2) - ((tt - center_t) ** 2) / (2 * sigma_t**2))
    mask = 1.0 - gauss  # 1 = keep, 0 = masked
    masked = Zxx * mask
    return masked, mask


def apply_rectangular_mask(
    Zxx: np.ndarray,
    center_f: int | None = None,
    center_t: int | None = None,
    width_f: int = 10,
    width_t: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a rectangular mask to an STFT magnitude array."""
    n_freq, n_time = Zxx.shape
    if center_f is None:
        center_f = n_freq // 2
    if center_t is None:
        center_t = n_time // 2

    mask = np.ones_like(Zxx)
    f_lo = max(0, center_f - width_f // 2)
    f_hi = min(n_freq, center_f + width_f // 2)
    t_lo = max(0, center_t - width_t // 2)
    t_hi = min(n_time, center_t + width_t // 2)
    mask[f_lo:f_hi, t_lo:t_hi] = 0.0
    return Zxx * mask, mask


def compute_psd(
    signal_arr: np.ndarray,
    fs: int = 256,
    nperseg: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density via Welch's method."""
    frequencies, psd = sig.welch(signal_arr, fs=fs, nperseg=min(nperseg, len(signal_arr)))
    return frequencies, psd
