import torch
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping
from src.config import TFT_PARAMS, QUANTILES, MODELS_DIR

def build_model():
    early_stop = EarlyStopping(
        monitor   = "val_loss",
        patience  = 10,
        min_delta = 1e-4,
        mode      = "min",
    )
    model = TFTModel(
        input_chunk_length    = TFT_PARAMS['input_chunk_length'],
        output_chunk_length   = TFT_PARAMS['output_chunk_length'],
        hidden_size           = TFT_PARAMS['hidden_size'],
        lstm_layers           = TFT_PARAMS['lstm_layers'],
        num_attention_heads   = TFT_PARAMS['num_attention_heads'],
        dropout               = TFT_PARAMS['dropout'],
        batch_size            = TFT_PARAMS['batch_size'],
        n_epochs              = TFT_PARAMS['n_epochs'],
        likelihood            = QuantileRegression(QUANTILES),
        random_state          = TFT_PARAMS['random_state'],
        use_static_covariates = True,
        add_relative_index    = True,
        pl_trainer_kwargs     = {
            'accelerator'        : 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices'            : 1,
            'callbacks'          : [early_stop],
            'enable_progress_bar': True,
        },
        model_name       = 'tft_rul_bearing',
        save_checkpoints = True,
        work_dir         = str(MODELS_DIR),
        force_reset      = True,
    )
    return model

def train_model(model, train_targets, train_past_covs, val_targets, val_past_covs):
    model.fit(
        series              = train_targets,
        past_covariates     = train_past_covs,
        val_series          = val_targets,
        val_past_covariates = val_past_covs,
        verbose             = True,
    )
    return model