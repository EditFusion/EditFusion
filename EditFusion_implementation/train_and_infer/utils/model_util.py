import torch
from torch import nn


def load_model_param(model, model_path):
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        # 获取可用的GPU数量
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        print(f"Using {n_gpu} GPUs!")
        if n_gpu > 1:
            # 使用DataParallel在多个GPU上复制模型
            model = nn.DataParallel(model)
    else:
        print("Using cpu")
        device = torch.device(
            "cpu"
            )

    model.to(device)
    model.load_state_dict(new_state_dict)
    return model, device
