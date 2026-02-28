# src/config.py
from pathlib import Path

# ─── Rutas base ───────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_RAW     = ROOT_DIR / "data" / "raw"
DATA_PROC    = ROOT_DIR / "data" / "processed"
MODELS_DIR   = ROOT_DIR / "models"
OUTPUTS_DIR  = ROOT_DIR / "outputs"

FAILURE_INFO = {
    "exp1": ["bearing_3", "bearing_4"],
    "exp2": ["bearing_1"],
    "exp3": ["bearing_3"],
}

# Parámetros del TFT
INPUT_CHUNK_LENGTH  = 50
OUTPUT_CHUNK_LENGTH = 10
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

TFT_PARAMS = {
    "input_chunk_length"  : INPUT_CHUNK_LENGTH,
    "output_chunk_length" : OUTPUT_CHUNK_LENGTH,
    "hidden_size"         : 64,
    "lstm_layers"         : 2,
    "num_attention_heads" : 4,
    "dropout"             : 0.1,
    "batch_size"          : 32,
    "n_epochs"            : 100,
    "random_state"        : 42,
}
