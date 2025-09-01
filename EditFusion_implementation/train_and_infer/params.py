from .utils.util import ObjectDict

model_params = {
    "input_size": 7 + 768,
    "output_size": 1,
    "hidden_size": 256,
    "num_layers": 2,
    "bidirectional": True,
    "dropout": 0.3,
    "max_es_len": 20,
}

training_param = ObjectDict(
    {
        "TRAIN_SPLIT": 0.7,
        "VAL_SPLIT": 0.15,
        "cc_embedding_lr": 1e-6,
        "lstm_lr": 1e-4,
        "fc_lr": 1e-4,
        "attention_lr": 1e-4,
        "epochs": 30,
        "batch_size": 16,
        "scheduler_step_size": 1,
        "scheduler_gamma": 0.9,
        "pos_weight": 0.58,
        "extra_block_loss_weight": 1,
    }
)
