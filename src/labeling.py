# src/labeling.py
import numpy as np
import pandas as pd


def find_failure_point(
    rms_series  : np.ndarray,
    burn_in     : float = 0.3,
    n_std       : float = 2.0,
) -> int:
    """
    Detecta el snapshot donde comienza la degradación crítica.
    
    Estrategia: calcula la media y std del RMS durante la fase sana
    (primeros burn_in% de snapshots) y busca el primer punto donde
    el RMS supera media + n_std * std de forma sostenida.
    
    Args:
        rms_series : array con el RMS de cada snapshot
        burn_in    : fracción inicial considerada fase sana (0.3 = 30%)
        n_std      : número de desviaciones estándar para el umbral
    
    Returns:
        índice del snapshot de fallo
    """
    n         = len(rms_series)
    burn_end  = int(n * burn_in)

    # Estadísticas de la fase sana
    healthy_rms = rms_series[:burn_end]
    mu          = np.mean(healthy_rms)
    sigma       = np.std(healthy_rms)
    threshold   = mu + n_std * sigma

    # Buscar el primer punto que supera el umbral de forma sostenida
    # Usamos una ventana de 10 snapshots para evitar falsos positivos
    window = 10
    for i in range(burn_end, n - window):
        if np.all(rms_series[i:i + window] > threshold):
            return i

    # Si no encuentra un punto claro, devuelve el máximo
    return int(np.argmax(rms_series))


def compute_rul_labels(
    df          : pd.DataFrame,
    failure_info: dict,
    burn_in     : float = 0.3,
    n_std       : float = 2.0,
) -> pd.DataFrame:
    """
    Añade la columna 'rul' al DataFrame de features.

    Para rodamientos que NO fallan en el experimento, el RUL se fija
    a 1.0 durante todo el experimento (rodamiento superviviente).

    Para rodamientos que SÍ fallan, el RUL va de 1.0 a 0.0.

    Args:
        df          : DataFrame con columnas experiment, bearing, snapshot, rms, ...
        failure_info: dict {"exp1": ["bearing_3", "bearing_4"], ...}
        burn_in     : fracción inicial sana para calcular umbral
        n_std       : desviaciones estándar para el umbral de fallo

    Returns:
        DataFrame con columna 'rul' añadida
    """
    df = df.copy()
    df["rul"]           = 1.0
    df["failure_point"] = -1
    df["is_failing"]    = False

    for exp_name, failing_bearings in failure_info.items():
        df_exp = df[df["experiment"] == exp_name]

        for bearing in df_exp["bearing"].unique():
            mask = (df["experiment"] == exp_name) & (df["bearing"] == bearing)
            rms  = df.loc[mask, "rms"].values
            n    = len(rms)

            if bearing in failing_bearings:
                # Rodamiento que falla — calcular punto de fallo y cuenta regresiva
                fp = find_failure_point(rms, burn_in=burn_in, n_std=n_std)

                # RUL = cuenta regresiva lineal de 1.0 a 0.0 hasta el punto de fallo
                rul = np.ones(n)
                rul[:fp + 1] = np.linspace(1.0, 0.0, fp + 1)
                rul[fp + 1:] = 0.0  # después del fallo se queda en 0

                df.loc[mask, "rul"]           = rul
                df.loc[mask, "failure_point"] = fp
                df.loc[mask, "is_failing"]    = True
            else:
                # Rodamiento superviviente — RUL fijo a 1.0
                df.loc[mask, "rul"]           = 1.0
                df.loc[mask, "failure_point"] = -1
                df.loc[mask, "is_failing"]    = False

    return df