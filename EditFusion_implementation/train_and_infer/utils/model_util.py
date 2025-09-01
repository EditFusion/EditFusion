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

    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        print(f"Using {n_gpu} GPUs!")
    else:
        print("Using cpu")
        device = torch.device("cpu")

    model.to(device)
    # Load state_dict before wrapping with DataParallel
    model.load_state_dict(new_state_dict)

    if torch.cuda.is_available() and n_gpu > 1:
        model = nn.DataParallel(model)

    return model, device
