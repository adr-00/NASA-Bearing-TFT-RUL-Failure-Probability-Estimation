# src/preprocessing.py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from tqdm import tqdm


def compute_time_features(signal: np.ndarray) -> dict:
    """
    Calcula indicadores de salud en el dominio del tiempo.
    
    Args:
        signal: array 1D de muestras de vibración (20480,)
    
    Returns:
        dict con las features calculadas
    """
    rms         = np.sqrt(np.mean(signal ** 2))
    peak        = np.max(np.abs(signal))
    kurtosis    = stats.kurtosis(signal)
    skewness    = stats.skew(signal)
    peak_to_peak = np.max(signal) - np.min(signal)
    crest_factor = peak / (rms + 1e-10)  # +1e-10 para evitar división por cero

    return {
        "rms"          : rms,
        "kurtosis"     : kurtosis,
        "skewness"     : skewness,
        "peak_to_peak" : peak_to_peak,
        "crest_factor" : crest_factor,
    }


def compute_freq_features(signal: np.ndarray, fs: int = 20_480, top_n: int = 5) -> dict:
    """
    Calcula indicadores de salud en el dominio de la frecuencia via FFT.
    
    Args:
        signal: array 1D de muestras de vibración (20480,)
        fs    : frecuencia de muestreo en Hz
        top_n : número de frecuencias dominantes a extraer
    
    Returns:
        dict con frecuencias y amplitudes dominantes
    """
    n      = len(signal)
    freqs  = np.fft.rfftfreq(n, d=1/fs)
    fft_mag = np.abs(np.fft.rfft(signal)) / n  # normalizado

    # Indices de los top_n picos de mayor amplitud
    top_idx = np.argsort(fft_mag)[::-1][:top_n]

    features = {}
    for i, idx in enumerate(top_idx):
        features[f"fft_freq_{i+1}"]  = freqs[idx]
        features[f"fft_amp_{i+1}"]   = fft_mag[idx]

    return features


def extract_features_from_snapshot(snapshot: np.ndarray, fs: int = 20_480, top_n: int = 5) -> dict:
    """
    Extrae todas las features de un snapshot completo.
    Usa solo el canal 0 (ch1) para consistencia entre experimentos.
    
    Args:
        snapshot: array (20480, n_canales)
        fs      : frecuencia de muestreo
        top_n   : frecuencias dominantes FFT a extraer
    
    Returns:
        dict con todas las features del snapshot
    """
    signal = snapshot[:, 0].astype(np.float64)  # canal X siempre existe

    time_feats = compute_time_features(signal)
    freq_feats = compute_freq_features(signal, fs=fs, top_n=top_n)

    return {**time_feats, **freq_feats}


def process_experiment(
    bearing_data : dict,
    exp_name     : str,
    fs           : int = 20_480,
    top_n        : int = 5,
) -> pd.DataFrame:
    """
    Procesa todos los snapshots de todos los rodamientos de un experimento
    y devuelve un DataFrame con una fila por snapshot por rodamiento.
    
    Args:
        bearing_data : dict {"bearing_1": [array, array, ...], ...}
        exp_name     : nombre del experimento (para la columna 'experiment')
        fs           : frecuencia de muestreo
        top_n        : frecuencias dominantes FFT
    
    Returns:
        DataFrame con columnas: experiment, bearing, snapshot, rms, kurtosis, ...
    """
    records = []

    for bearing_name, snapshots in bearing_data.items():
        for snap_idx, snapshot in enumerate(tqdm(snapshots, desc=f"{exp_name} - {bearing_name}")):
            feats = extract_features_from_snapshot(snapshot, fs=fs, top_n=top_n)
            feats["experiment"] = exp_name
            feats["bearing"]    = bearing_name
            feats["snapshot"]   = snap_idx
            records.append(feats)

    df = pd.DataFrame(records)

    # Reordenar columnas para que los metadatos vayan primero
    meta_cols    = ["experiment", "bearing", "snapshot"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feature_cols]

    return df.sort_values(["bearing", "snapshot"]).reset_index(drop=True)