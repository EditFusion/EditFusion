import pandas as pd
from .model.LSTM_model import LSTMClassifier
from .params import model_params
import torch
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Subset
from .utils.model_util import load_model_param

# 加载状态字典
# 首先需要重新创建与原始模型相同架构的模型实例
model = LSTMClassifier(**model_params)
script_path = Path(os.path.dirname(os.path.abspath(__file__)))
model_path = script_path / "data" / "tmp.pth"

model, device = load_model_param(model, model_path)
model.eval()  # 切换到评估模式

# 加载数据集
dataset_path = script_path / "data" / "CC嵌入数据集_tokenlen32.csv"
dataset = model.CCEmbedding_class.get_dataset(dataset_path)

# ###### debug #######
# dataset = Subset(dataset, range(200))
# ###### debug #######

data_loader = DataLoader(
    dataset, batch_size=4, shuffle=False, collate_fn=model.CCEmbedding_class.collate_fn
)


# todo 这个函数来自 LSTMTrainer，代码冗余，想办法改为一处
def mask_output(outputs, labels, lengths):
    # outputs shape 为 (batch_size, 20, 1)
    # 我们只需要 outputs 的前 len(lengths) 个输出，因为后面的是填充的
    outputs = outputs[
        :, : max(lengths), :
    ]  # outputs shape 为 (batch_size, max(lengths), 1)

    # 计算损失
    # 输出和标签的形状需要一致，所以我们需要对标签进行扩展
    outputs = outputs.squeeze(2)

    # 使用 mask，batch_size * max_length 的 mask，把 outputs 对应在 lengths 后的部分遮蔽为 0
    mask = torch.arange(outputs.size(1)).expand(
        len(lengths), outputs.size(1)
    ) < lengths.unsqueeze(1)
    mask = mask.to(device)
    outputs_selected = outputs.masked_select(mask)
    labels_selected = labels.masked_select(mask)
    return outputs_selected, labels_selected, mask


def accuracy_on_dataset(data_loader):
    # inference
    with torch.inference_mode():
        correct_num = 0
        total_num = 0
        kind_counter = defaultdict(int)
        kind_correct_counter = defaultdict(int)
        model.eval()
        for loaded_feats, labels, lengths, resolution_kinds in tqdm(
            data_loader, dynamic_ncols=True, desc=f"counting chunk accuracy"
        ):
            # 将数据移动到 GPU 上
            labels = labels.to(device)
            if isinstance(loaded_feats, tuple):
                loaded_feats = tuple(f.to(device) for f in loaded_feats)

            # todo 这里代码写的对不对（形状是不是对上了）还需要验证
            outputs = model(
                loaded_feats, lengths
            )  # 在这个阶段先 pack_padded_sequence，然后再 pad_packed_sequence

            outputs_selected, labels_selected, _ = mask_output(outputs, labels, lengths)

            curr_batch_size = labels.shape[0]
            # 如果每个样本的 outputs 和 labels 相同，说明预测正确
            for i in range(curr_batch_size):
                kind_counter[resolution_kinds[i]] += 1
                if torch.equal(
                    outputs[i], labels[i]
                ):  # 只比较标签是否一致，因为空行已经在回溯建立数据集时过滤掉了
                    correct_num += 1
                    kind_correct_counter[resolution_kinds[i]] += 1
            total_num += curr_batch_size
        print(f"Accuracy: {round(correct_num / total_num * 100, 2)}%")

        for kind in kind_counter.keys():
            print(
                f"Accuracy on {kind}: {round(kind_correct_counter[kind] / kind_counter[kind] * 100, 2)}%, {kind_correct_counter[kind]}/{kind_counter[kind]}"
            )


accuracy_on_dataset(data_loader)
