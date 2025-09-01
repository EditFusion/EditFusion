from .utils.util import ObjectDict

# 模型参数
model_params = {
    "input_size": 7 + 768,  # 构造特征数量，就是输入 LSTM 的特征数量
    # "input_size": 7 + 768 + 768,  # 构造特征数量，就是输入 LSTM 的特征数量
    # "input_size": 768,  # 构造特征数量，就是输入 LSTM 的特征数量
    # 'input_size': 7,      # 构造特征数量，就是输入 LSTM 的特征数量
    "output_size": 1,
    "hidden_size": 256,
    "num_layers": 2,
    "bidirectional": True,
    "dropout": 0.3,
    "max_es_len": 20,
}

# 训练参数
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
        "scheduler_step_size": 1,  # 每隔多少个 epoch 调整一次学习率
        "scheduler_gamma": 0.9,  # 学习率衰减的乘数因子
        "pos_weight": 0.58,
        # "pos_weight": 0.4017,
        "extra_block_loss_weight": 1,
    }
)
