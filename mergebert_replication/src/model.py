# src/model.py
# -*- coding: utf-8 -*-
"""
此文件定义了 MergeBERT 模型的完整 PyTorch 实现。
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

# 从统一的配置文件中导入
from .config import EDIT_TYPE_MAP, MODEL_NAME

class MergeBertEmbeddings(nn.Module):
    """
    自定义嵌入层，这是 MergeBERT 的核心创新之一。
    它将三种不同的嵌入向量相加，为模型提供丰富的输入信息：
    1. 词元嵌入 (Token Embeddings): 词的语义含义。
    2. 位置嵌入 (Position Embeddings): 词在序列中的位置。
    3. 编辑类型嵌入 (Edit Type Embeddings): 词元是通过哪种编辑操作（增、删、改、等）产生的。
    """
    def __init__(self, config):
        super().__init__()
        # 标准词元嵌入，其权重可以从预训练的 CodeBERT 加载
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 标准位置嵌入
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 编辑类型嵌入，这是一个小型的嵌入层，词汇表大小为编辑类型的数量（5种）
        self.edit_type_embeddings = nn.Embedding(len(EDIT_TYPE_MAP), config.hidden_size)

        # 对合并后的嵌入向量进行层归一化和 dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 注册一个持久的 buffer，用于存储位置ID，这样它就不会被视为模型参数
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, edit_type_ids):
        # 获取输入序列的长度
        seq_length = input_ids.size(1)
        # 根据序列长度获取相应的位置ID
        position_ids = self.position_ids[:, :seq_length]
        
        # 分别获取三种嵌入向量
        token_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        edit_embeds = self.edit_type_embeddings(edit_type_ids)
        
        # 核心：将三种嵌入向量逐元素相加
        embeddings = token_embeds + position_embeds + edit_embeds
        # 应用 LayerNorm 和 Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MergeBERT(nn.Module):
    """
    MergeBERT 模型的主体架构。
    """
    def __init__(self, num_labels: int):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(MODEL_NAME)
        
        # 加载预训练的 Transformer 编码器 (例如 CodeBERT)
        # 这个编码器的权重是共享的，将并行处理四路输入
        self.encoder = RobertaModel.from_pretrained(MODEL_NAME)
        
        # **关键**：用我们自定义的 MergeBertEmbeddings 替换掉原始模型的嵌入层
        self.encoder.embeddings = MergeBertEmbeddings(self.config)

        # 定义一个可学习的参数，用于对四路编码器的输出进行加权求和
        # 初始值为1，后续通过 softmax 转换为权重
        self.aggregation_weights = nn.Parameter(torch.ones(4))

        # 标准的线性分类层，用于将聚合后的向量映射到最终的标签类别上
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, **inputs):
        # 定义四路输入的名称
        streams = ['ao', 'oa', 'bo', 'ob']
        cls_outputs = []

        # 并行处理四路输入
        for stream in streams:
            input_ids = inputs[f'input_ids_{stream}']
            attention_mask = inputs[f'attention_mask_{stream}']
            edit_type_ids = inputs[f'edit_type_ids_{stream}']

            # 1. 通过我们自定义的嵌入层，得到融合了三种信息的嵌入向量
            embedding_output = self.encoder.embeddings(input_ids, edit_type_ids)
            
            # 2. 将嵌入向量送入 Transformer 编码器的主体部分
            encoder_outputs = self.encoder.encoder(
                embedding_output,
                attention_mask=attention_mask
            )
            # 编码器的输出是一个元组，我们取第一个元素，即所有词元的隐藏状态
            sequence_output = encoder_outputs[0]
            
            # 3. 提取序列的第一个词元 ([CLS] 或 <s>) 的隐藏状态，它代表了整个序列的语义
            cls_output = sequence_output[:, 0, :]
            cls_outputs.append(cls_output)
            
        # 4. 加权聚合四路输出的 [CLS] 向量
        #    首先，使用 softmax 将可学习的权重转换为一个概率分布（和为1）
        softmax_weights = nn.functional.softmax(self.aggregation_weights, dim=0)
        
        #    将四路的 [CLS] 向量堆叠起来
        aggregated_output = torch.stack(cls_outputs, dim=1) # -> 形状: (batch_size, 4, hidden_size)
        
        #    利用广播机制进行加权求和
        weighted_sum = torch.sum(aggregated_output * softmax_weights.view(1, 4, 1), dim=1)

        # 5. 将聚合后的向量送入分类器，得到最终的 logits
        logits = self.classifier(weighted_sum)
        return logits
