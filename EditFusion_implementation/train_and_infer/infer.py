# 提供模型调用接口，为 flask API 提供服务
import os
import torch
import pandas as pd
from pathlib import Path
from typing import List
from train_and_infer.utils.conflict_utils import Conflict

from train_and_infer.utils.es_generator import compute, SequenceDiff
from .model.LSTM_model import LSTMClassifier
from .params import model_params
from .utils.model_util import load_model_param

# 初始化全局变量 单例模式存储模型实例
model_singleton = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用 {device} 进行推理")


def load_model():
    """
    加载模型，并确保只加载一次。
    """
    global model_singleton
    # 检查模型实例是否已经创建
    if model_singleton is None:
        # 创建模型实例
        model = LSTMClassifier(**model_params)
        script_path = Path(os.path.dirname(os.path.abspath(__file__)))
        model_path = script_path / "data" / "LSTM_model.pth"
        model, device = load_model_param(model, model_path)

        # 将创建的模型实例存储在全局变量中
        model_singleton = model
    return model_singleton


class EditScriptWithFrom:
    def __init__(self, sd: SequenceDiff, _from: str) -> None:
        self.es = sd
        self.from_id = _from


def get_predicted_result(base: List[str], ours: List[str], theirs: List[str]):
    """
    用于预测两个字符串的编辑脚本
    Args:
        base: 原始字符串
        ours: 第一个修改后的字符串
        theirs: 第二个修改后的字符串
    Returns:
        生成的结果
    """
    # 确保在使用前加载模型
    model = load_model()

    conflict = Conflict(ours, theirs, base)
    # 1. 从三个版本内容生成编辑脚本
    ess_ours = compute(base, ours)
    ess_theirs = compute(base, theirs)
    ess_with_from = [EditScriptWithFrom(sd, "ours") for sd in ess_ours] + [
        EditScriptWithFrom(sd, "theirs") for sd in ess_theirs
    ]
    ess_with_from.sort(key=lambda es: es.es.seq1Range)

    # 2. 针对一个冲突块，对其中每个编辑脚本抽取特征
    def extract_features(
        ess_with_from: List[EditScriptWithFrom], conflict: Conflict
    ) -> torch.Tensor:
        """
        从编辑脚本中抽取特征（todo 这不是只有位置特征吗）
        Returns:
            特征向量
        """
        feats = []
        for es_with_from in ess_with_from:
            es = es_with_from.es
            # 按顺序抽取特征
            origin_start = es.seq1Range.start
            origin_end = es.seq1Range.end
            modified_start = es.seq2Range.start
            modified_end = es.seq2Range.end
            origin_length = origin_end - origin_start
            modified_length = modified_end - modified_start
            length_diff = modified_length - origin_length
            feats.append(
                [
                    origin_start,
                    origin_end,
                    modified_start,
                    modified_end,
                    origin_length,
                    modified_length,
                    length_diff,
                ]
            )

        # 转化为 float tensor
        return torch.tensor(feats).float()

    feats = extract_features(ess_with_from, conflict)
    # 3. 将特征输入模型，得到预测结果
    feats = feats.to(device)

    # todo 目前没有考虑 batch_size，每次只输入一个样本，可优化
    # 输入 [batch_size, seq_len, feature_dim]
    feats = feats.unsqueeze(0)
    lengths = torch.tensor([len(ess_with_from)]).to(device)

    # ! ??? 不是，这也没 token id 啊

    outputs = model(feats, lengths)
    outputs = outputs.squeeze(2)  # [batch_size, seq_len]
    outputs = outputs.round().int()  # 转化为 0/1

    # 4. 后处理，将预测结果生成实际合成的代码
    def generate_resolution(
        ess_with_from: List[EditScriptWithFrom], labels: List[int], conflict: Conflict
    ) -> List[str]:
        """
        从编辑脚本和预测结果生成合成代码
        Args:
            ess: 编辑脚本       sorted
            labels: 预测结果    对应 ess 的顺序
            conflict: 冲突块
        Returns:
            合成代码
        """
        resolution = []
        base_to_add = 0

        for es_with_from, label in zip(ess_with_from, labels):
            if label == 0:
                continue
            else:
                # 加入这个编辑脚本的修改

                # 将之前的 base 添加到 resolution 中
                if es_with_from.es.seq1Range.start < base_to_add:
                    # todo 接受内容相同的编辑 和 concat 区分
                    # todo 对于 相同位置的 token 合并等操作
                    # todo：也可以考虑采取预测值更大的方案
                    raise Exception("暂不提供解决方案，请手动解决")
                resolution += conflict.base[
                    base_to_add : es_with_from.es.seq1Range.start
                ]

                # 将修改后的内容添加到 resolution 中
                modified_content = (
                    conflict.ours if es_with_from.from_id == "ours" else conflict.theirs
                )
                resolution += modified_content[
                    es_with_from.es.seq2Range.start : es_with_from.es.seq2Range.end
                ]
                base_to_add = es_with_from.es.seq1Range.end
        # 将最后的 base 添加到 resolution 中
        resolution += conflict.base[base_to_add:]
        return resolution

    return generate_resolution(ess_with_from, outputs.tolist()[0], conflict)


if __name__ == "__main__":
    base = ["a", "b", "c", "d", "e"]
    ours = ["a", "b", "c", "d", "111", "e"]
    theirs = ["333", "a", "222", "c", "d", "e", "444"]
    print(get_predicted_result(base, ours, theirs))
