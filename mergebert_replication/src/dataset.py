# src/dataset.py
# -*- coding: utf-8 -*-
"""
此文件定义了用于加载和预处理 MergeBERT 数据集的 PyTorch Dataset 和 DataLoader。
它负责从JSON文件中读取预处理好的冲突块，并将其转换为模型所需的张量格式。
"""

import torch
import os
import json
from typing import List, Dict, Tuple

from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from diff_match_patch import diff_match_patch

# 从配置文件导入所需的常量
from .config import EDIT_TYPE_MAP, MAX_LEN, LABEL_MAP

# 定义编辑操作的符号，用于生成编辑序列
EDIT_SYMBOLS = {'EQUAL': '=', 'INSERT': '+', 'DELETE': '-'}

def generate_aligned_sequences(
    tokens_1: List[str], tokens_2: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    使用 diff-match-patch 库对两个词元序列进行对齐。
    这是 MergeBERT 的核心数据处理步骤之一。

    Args:
        tokens_1 (List[str]): 第一个词元序列 (例如，来自 base 版本)。
        tokens_2 (List[str]): 第二个词元序列 (例如，来自 a/b 版本)。

    Returns:
        Tuple[List[str], List[str], List[str]]: 返回三个列表：
        - aligned_1: 对齐后的第一个序列，用 '<pad>' 填充缺失部分。
        - aligned_2: 对齐后的第二个序列，用 '<pad>' 填充缺失部分。
        - edit_sequence: 描述从 aligned_1 到 aligned_2 转换的编辑序列。
    """
    dmp = diff_match_patch()
    # 使用空格连接词元，以便 dmp 库可以将其作为文本处理
    diffs = dmp.diff_main(' '.join(tokens_1), ' '.join(tokens_2))
    
    aligned_1, aligned_2, edit_sequence = [], [], []
    
    for op, text in diffs:
        tokens = text.strip().split(' ')
        if not tokens or tokens == ['']:
            continue
            
        if op == dmp.DIFF_EQUAL:
            # 如果两个序列的这部分相等，则直接添加到对齐序列中
            aligned_1.extend(tokens)
            aligned_2.extend(tokens)
            edit_sequence.extend([EDIT_SYMBOLS['EQUAL']] * len(tokens))
        elif op == dmp.DIFF_DELETE:
            # 如果这部分在序列1中存在，但在序列2中被删除
            aligned_1.extend(tokens)
            aligned_2.extend(['<pad>'] * len(tokens)) # 在序列2的相应位置填充
            edit_sequence.extend([EDIT_SYMBOLS['DELETE']] * len(tokens))
        elif op == dmp.DIFF_INSERT:
            # 如果这部分在序列2中存在，但在序列1中是新增的
            aligned_1.extend(['<pad>'] * len(tokens)) # 在序列1的相应位置填充
            aligned_2.extend(tokens)
            edit_sequence.extend([EDIT_SYMBOLS['INSERT']] * len(tokens))

    # 这是一个简化的替换操作('~')检测逻辑
    # 它将一个删除后紧跟一个插入的操作视为一次替换
    i = 0
    while i < len(edit_sequence) - 1:
        if edit_sequence[i] == EDIT_SYMBOLS['DELETE'] and edit_sequence[i+1] == EDIT_SYMBOLS['INSERT']:
            edit_sequence[i] = '~'
            edit_sequence.pop(i+1)
            # 简化处理：假设替换是1对1的，因此删除对齐序列中多余的<pad>和token
            # 一个更鲁棒的实现需要处理不等长的替换
            if i < len(aligned_1) and i + 1 < len(aligned_1):
                del aligned_1[i+1]
            if i < len(aligned_2):
                del aligned_2[i]
        else:
            i += 1
            
    return aligned_1, aligned_2, edit_sequence

class MergeBERTDataset(Dataset):
    """
    自定义的 PyTorch 数据集类，用于加载和处理 MergeBERT 数据。
    """
    def __init__(self, data_dir: str, tokenizer: RobertaTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        print(f"Initializing dataset from: {data_dir}")
        
        # 遍历数据目录下的所有 .json 文件
        for filename in os.listdir(data_dir):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(data_dir, filename)
            print(f"Processing file: {filename}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 每个JSON文件是一个包含多个冲突文件实例的列表
                for conflict_file in data:
                    # 遍历每个文件中的所有冲突块
                    for chunk in conflict_file.get('conflict_chunks', []):
                        # --- 核心处理逻辑开始 ---
                        a_content = chunk.get('a_content', '')
                        b_content = chunk.get('b_content', '')
                        o_content = chunk.get('o_content', '')
                        label_str = chunk.get('mergebert_label', 'OTHER')

                        # 将字符串标签通过 LABEL_MAP 转换为整数ID
                        # 如果标签不在MAP中，则默认为 'OTHER' 对应的ID
                        label = LABEL_MAP.get(label_str, LABEL_MAP['OTHER'])

                        # 将代码内容分词
                        tokens_a = self.tokenizer.tokenize(a_content)
                        tokens_b = self.tokenizer.tokenize(b_content)
                        tokens_o = self.tokenizer.tokenize(o_content)

                        # 生成对齐序列和编辑序列
                        o_a, a_o, delta_ao_str = generate_aligned_sequences(tokens_o, tokens_a)
                        o_b, b_o, delta_bo_str = generate_aligned_sequences(tokens_o, tokens_b)

                        # 准备模型的四路输入
                        sequences = {'ao': a_o, 'oa': o_a, 'bo': b_o, 'ob': o_b}
                        # 注意：a-o 和 o-a 的编辑序列是相同的，b-o 和 o-b 也是
                        deltas = {'ao': delta_ao_str, 'oa': delta_ao_str, 'bo': delta_bo_str, 'ob': delta_bo_str}
                        
                        instance_tensors = {}
                        # 对每一路输入进行编码和张量化
                        for key in sequences:
                            tokens = sequences[key][:self.max_len]
                            delta_symbols = deltas[key][:self.max_len]

                            # 使用 tokenizer 对词元进行编码，并进行填充和截断
                            encoded = self.tokenizer.encode_plus(
                                tokens,
                                is_split_into_words=True,
                                max_length=self.max_len,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt'
                            )
                            instance_tensors[f'input_ids_{key}'] = encoded['input_ids'].squeeze(0)
                            instance_tensors[f'attention_mask_{key}'] = encoded['attention_mask'].squeeze(0)

                            # 将编辑序列符号转换为ID
                            edit_type_ids = [EDIT_TYPE_MAP.get(s, EDIT_TYPE_MAP[' ']) for s in delta_symbols]
                            padding_length = self.max_len - len(edit_type_ids)
                            edit_type_ids.extend([EDIT_TYPE_MAP[' ']] * padding_length)
                            instance_tensors[f'edit_type_ids_{key}'] = torch.tensor(edit_type_ids, dtype=torch.long)

                        # 添加真实的标签ID
                        instance_tensors['label'] = torch.tensor(label, dtype=torch.long)
                        self.samples.append(instance_tensors)

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    def __len__(self):
        """返回数据集中的样本总数"""
        return len(self.samples)

    def __getitem__(self, idx):
        """根据索引获取单个样本"""
        return self.samples[idx]

def create_data_loader(data_dir, tokenizer, max_len, batch_size, shuffle=True):
    """
    创建一个 DataLoader 实例。

    Args:
        data_dir (str): 数据集目录。
        tokenizer (RobertaTokenizer): 分词器实例。
        max_len (int): 序列最大长度。
        batch_size (int): 批处理大小。
        shuffle (bool): 是否在每个 epoch 开始时打乱数据。

    Returns:
        DataLoader: 配置好的 PyTorch DataLoader。
    """
    dataset = MergeBERTDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)