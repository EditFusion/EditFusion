# src/train.py
# -*- coding: utf-8 -*-
"""
此文件是模型的主训练脚本。
它负责初始化模型、分词器、数据加载器、优化器等，并执行完整的训练和验证流程。
"""

import torch
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer
from tqdm import tqdm
import os

# 从项目其他模块导入所需的类和配置
from .model import MergeBERT
from .dataset import create_data_loader
from .config import (
    MODEL_NAME,
    NUM_LABELS,
    EPOCHS,
    BATCH_SIZE,
    MAX_LEN,
    LEARNING_RATE,
    TRAIN_DIR,
)

def train():
    """执行完整的模型训练流程"""
    # 1. 初始化设备 (GPU或CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 初始化分词器
    # 分词器负责将文本代码转换为模型可以理解的数字ID
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # 3. 准备数据
    # 检查训练数据目录是否存在
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training data directory not found at '{TRAIN_DIR}'")
        print("Please ensure the path in src/config.py is correct.")
        return

    # 创建数据加载器 (DataLoader)
    print("Creating data loader...")
    train_dataloader = create_data_loader(TRAIN_DIR, tokenizer, MAX_LEN, BATCH_SIZE)
    
    # 检查数据是否成功加载
    if len(train_dataloader.dataset) == 0:
        print("Error: No data loaded. Please check the training data directory and file formats.")
        return
    print(f"Dataset loaded with {len(train_dataloader.dataset)} samples.")

    # 4. 初始化模型
    # 将模型移动到指定设备 (GPU/CPU)
    print("Initializing MergeBERT model...")
    model = MergeBERT(num_labels=NUM_LABELS).to(device)
    
    # 5. 初始化优化器和学习率调度器
    # AdamW 是一个常用的针对 Transformer 模型的优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    # 学习率调度器，用于在训练过程中动态调整学习率，有助于模型更好地收敛
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # 定义损失函数，对于多分类任务，通常使用交叉熵损失
    loss_fn = torch.nn.CrossEntropyLoss()

    # 6. 开始训练循环
    for epoch in range(EPOCHS):
        print(f'\n======== Epoch {epoch + 1} / {EPOCHS} ========')
        model.train() # 将模型设置为训练模式
        total_loss = 0
        
        # 使用tqdm显示进度条，方便地遍历所有批次的数据
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad() # 清除上一批次的梯度
            
            # 将批次中的所有张量移动到指定设备
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop('label') # 从批次中分离出标签
            except Exception as e:
                print(f"Error moving batch to device or popping label: {e}")
                continue # 跳过这个有问题的批次
            
            # 模型前向传播，得到预测的 logits
            logits = model(**batch)
            
            # 计算损失
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            # 反向传播，计算梯度
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新模型参数
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss:.4f}")

    # 7. 保存训练好的模型
    # 创建模型保存目录（如果不存在）
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    save_path = f"saved_models/mergebert_epoch_{EPOCHS}.bin"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved to {save_path}")

if __name__ == '__main__':
    train()
