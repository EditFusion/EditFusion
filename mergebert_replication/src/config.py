# src/config.py
# -*- coding: utf-8 -*-
"""
该文件集中管理项目的所有配置和超参数，方便统一修改和维护。
"""

# -------------------
# 模型相关配置
# -------------------
# 使用的预训练模型名称，来自 Hugging Face Model Hub
MODEL_NAME = 'huggingface/CodeBERTa-small-v1'

# 标签映射：将字符串标签转换为数字索引
# 根据您的数据样本和描述，我们定义以下标签
# 如果您的数据中有其他标签，请在这里添加
LABEL_MAP = {
    'A': 0,
    'B': 1,
    'O': 2,
    'AB': 3,
    'mixline': 4,
    'newline': 5,
    'OTHER': 6, # 用于处理如 'mixline', 'newline' 等被合并的类别
}

# 分类的总类别数
NUM_LABELS = len(LABEL_MAP)

# -------------------
# 训练相关超参数
# -------------------
EPOCHS = 3
BATCH_SIZE = 16 # 可根据您的GPU显存大小进行调整
MAX_LEN = 512   # 输入序列的最大长度，超过部分将被截断
LEARNING_RATE = 5e-5

# -------------------
# 数据路径配置
# -------------------
# !!关键：这里指向您提供的预处理好的数据集目录
TRAIN_DIR = "/home/foril/projects/EditFusion/EditFusion_implementation/train_and_infer/data/gathered_data/mergebert_all_lang_with_no_newline/"
TEST_DIR = "data/test" # 测试集目录（当前未使用，作为预留）

# -------------------
# MergeBERT 特定配置
# -------------------
# 编辑类型符号到数字索引的映射
# '=': Equal, '+': Insert, '-': Delete, '~': Replace, ' ': Padding
EDIT_TYPE_MAP = {'=': 0, '+': 1, '-': 2, '~': 3, ' ': 4}
