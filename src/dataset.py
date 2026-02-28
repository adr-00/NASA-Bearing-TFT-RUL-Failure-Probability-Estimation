# src/dataset.py
import pandas as pd
import numpy as np
from darts import TimeSeries
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = [
    "rms", "kurtosis", "skewness",
    "peak_to_peak", "crest_factor",
    "fft_freq_1", "fft_freq_2", "fft_freq_3", "fft_freq_4", "fft_freq_5",
    "fft_amp_1",  "fft_amp_2",  "fft_amp_3",  "fft_amp_4",  "fft_amp_5",
]

FAILING_BEARINGS = [
    ("exp1", "bearing_3"),
    ("exp1", "bearing_4"),
    ("exp2", "bearing_1"),
    ("exp3", "bearing_3"),
]

TRAIN_BEARINGS = [
    ("exp1", "bearing_3"),
    ("exp1", "bearing_4"),
    ("exp2", "bearing_1"),
]
VAL_BEARINGS = [
    ("exp3", "bearing_3"),
]

def load_labeled_data(path):
    df = pd.read_csv(path)
    mask = df.apply(
        lambda r: (r["experiment"], r["bearing"]) in FAILING_BEARINGS, axis=1
    )
    return df[mask].copy()

def scale_features(df_train, df_val):
    scaler = MinMaxScaler()
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_train[FEATURE_COLS] = scaler.fit_transform(df_train[FEATURE_COLS])
    df_val[FEATURE_COLS]   = scaler.transform(df_val[FEATURE_COLS])
    return df_train, df_val, scaler

def make_timeseries(df):
    targets   = []
    past_covs = []
    for exp_name, bearing in FAILING_BEARINGS:
        df_b = df[
            (df["experiment"] == exp_name) &
            (df["bearing"]    == bearing)
        ].sort_values("snapshot").reset_index(drop=True)
        if df_b.empty:
            continue

        fp   = int(df_b['failure_point'].iloc[0])
        df_b = df_b[df_b['snapshot'] <= fp].reset_index(drop=True)

        print(f'  {exp_name} - {bearing}: {len(df_b)} snapshots (fp={fp})')
        target = TimeSeries.from_dataframe(
            df_b,
            value_cols         = "rul",
            fill_missing_dates = False,
            freq               = None,
        )
        past_cov = TimeSeries.from_dataframe(
            df_b,
            value_cols         = FEATURE_COLS,
            fill_missing_dates = False,
            freq               = None,
        )
        exp_id     = int(exp_name.replace('exp', ''))
        bearing_id = int(bearing.replace('bearing_', ''))
        static_cov = pd.DataFrame({'exp_id': [exp_id], 'bearing_id': [bearing_id]})
        target   = target.with_static_covariates(static_cov)
        past_cov = past_cov.with_static_covariates(static_cov)
        targets.append(target)
        past_covs.append(past_cov)
    return targets, past_covs

def get_train_val_series(df):
    def filter_bearings(df, bearing_list):
        mask = df.apply(
            lambda r: (r['experiment'], r['bearing']) in bearing_list, axis=1
        )
        return df[mask].copy()
    df_train = filter_bearings(df, TRAIN_BEARINGS)
    df_val   = filter_bearings(df, VAL_BEARINGS)
    df_train, df_val, scaler = scale_features(df_train, df_val)
    df_scaled = pd.concat([df_train, df_val], ignore_index=True)
    all_targets, all_past_covs = make_timeseries(df_scaled)
    n_train         = len(TRAIN_BEARINGS)
    train_targets   = all_targets[:n_train]
    train_past_covs = all_past_covs[:n_train]
    val_targets     = all_targets[n_train:]
    val_past_covs   = all_past_covs[n_train:]
    return train_targets, train_past_covs, val_targets, val_past_covs, scaler
