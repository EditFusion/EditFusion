from accelerate import Accelerator
from torch.utils.data import DataLoader, random_split, Subset
from ..params import training_param
import torch.optim as optim
from collections import defaultdict
from tqdm import tqdm
from torch import nn
import torch
import os
import pandas as pd
import torch.distributed as dist


def merge_defaultdict_int(dicts_list):
    """合并一批 defaultdict(int)。"""
    merged = defaultdict(int)
    for d in dicts_list:
        for k, v in d.items():
            merged[k] += v
    return merged


def gather_dict_from_all_procs(local_dict, only_main_process=True):
    """
    使用 dist.all_gather_object 收集所有进程上的字典 (defaultdict(int))，
    返回合并后的字典。仅在主进程上返回合并结果，其它进程上返回 None。

    参数：
      local_dict: 当前进程上的 defaultdict(int)。
      only_main_process: 是否只在 main process 上返回合并后的字典，其他进程返回 None。
    """
    # 如果当前只跑单卡，或者没有初始化dist，则直接返回 local_dict
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return local_dict

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 用 None 占位，长度为 world_size，用来存放从所有进程收集到的字典
    dict_list = [None for _ in range(world_size)]

    # all_gather_object：会把 local_dict 从每个 rank 收集到每个 rank 的 dict_list
    dist.all_gather_object(dict_list, local_dict)

    # 现在在每个 rank 上，dict_list 都包含了所有进程的字典
    # 如果只想在主进程合并，就在 rank=0 执行合并
    if only_main_process and rank != 0:
        return None
    else:
        # print(f"rank {rank} dict_list: {dict_list}")
        merged_dict = merge_defaultdict_int(dict_list)
        return merged_dict


class GRUTrainer:
    def __init__(
        self, dataset_path, model, accelerator, debug: bool = False, log_file=None
    ):
        if debug:
            self.log_file = None
        else:
            # 若 log_file 传入但不存在，创建
            if log_file and not os.path.exists(
                log_file if isinstance(log_file, str) else log_file.name
            ):
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                self.log_file = open(log_file, "w")
            else:
                self.log_file = log_file

        batch_size = training_param.batch_size

        self.accelerator = accelerator
        # 获取设备信息
        device = self.accelerator.device
        print(f"Using device: {device}")

        self.dataset_path = dataset_path
        self.model = model
        self.train_split = training_param.TRAIN_SPLIT
        self.val_split = training_param.VAL_SPLIT

        if "pos_weight" in training_param:
            pos_weight = torch.tensor([training_param.pos_weight], device=device)
        else:  # 默认为 1
            pos_weight = torch.tensor([1.0], device=device)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )  # 使用 BCEWithLogitsLoss，不需要额外的 sigmoid 层

        # 分层学习率
        pretrained_params = model.CCEmbedding_class.cc_embedding.parameters()
        # 如果 model.CCEmbedding_class 存在 attention 属性，则 attention 参数也要加入优化器
        if hasattr(model.CCEmbedding_class, "attention"):
            attention_params = model.CCEmbedding_class.attention.parameters()
        else:
            attention_params = []
        if hasattr(model.CCEmbedding_class, "origin_embedding"):
            origin_embedding_params = model.CCEmbedding_class.origin_embedding.parameters()
        else:
            origin_embedding_params = []
        gru_params = model.gru.parameters()  # Changed from lstm to gru
        fc_params = model.fc.parameters()
        self.optimizer = optim.Adam(
            [
                {"params": pretrained_params, "lr": training_param.cc_embedding_lr},
                {"params": origin_embedding_params, "lr": training_param.cc_embedding_lr},
                {"params": attention_params, "lr": training_param.attention_lr},
                {"params": gru_params, "lr": training_param.lstm_lr},  # Using lstm_lr for gru
                {"params": fc_params, "lr": training_param.fc_lr},
            ]
        )
        # 定义学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=training_param.scheduler_step_size,
            gamma=training_param.scheduler_gamma,
        )

        # 创建 DataSet
        GRU_dataset = self.model.CCEmbedding_class.get_dataset(self.dataset_path)
        self.accelerator.print(f"Dataset size: {len(GRU_dataset)}")

        if debug:
            GRU_dataset = Subset(GRU_dataset, range(50))

        # 计算训练集/验证集/测试集的大小
        total_size = len(GRU_dataset)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size

        # 随机划分数据集
        train_dataset, val_dataset, test_dataset = random_split(
            GRU_dataset, [train_size, val_size, test_size]
        )

        # 创建 DataLoader
        train_loader, val_loader, test_loader = [
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(dataset is train_dataset),
                collate_fn=self.model.CCEmbedding_class.collate_fn,
            )
            for dataset in [train_dataset, val_dataset, test_dataset]
        ]

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            train_loader,
            val_loader,
            test_loader,
            self.scheduler,
        )

    def forward_in_stage(
        self,
        data_loader: DataLoader,
        stage: str,
        epoch: int,
        epochs: int,
        log_file=None,
    ):
        """
        在训练/验证/测试阶段前向传播，计算损失和准确率。
        每次调用是一个 epoch 的训练/验证/测试。
        返回：
            loss: 该 epoch 的平均损失。
            accuracy: 该 epoch 的准确率。
        """

        def mask_output(outputs, labels, lengths):
            """
            对 outputs 和 labels 根据 lengths 创建掩码，并选择有效的输出和标签。

            参数:
                outputs (torch.Tensor): 模型输出，形状为 (batch_size, seq_len, 1)。
                labels (torch.Tensor): 标签，形状为 (batch_size, seq_len)。
                lengths (torch.Tensor): 每个序列的实际长度，形状为 (batch_size,)。

            返回:
                outputs_selected (torch.Tensor): 选择后的输出，形状为 (N,)。
                labels_selected (torch.Tensor): 选择后的标签，形状为 (N,)。
            """
            device = outputs.device
            batch_size, seq_len, _ = outputs.size()
            max_length = lengths.max().item()
            max_length = min(max_length, seq_len)  # 避免越界

            # 截取有效的时间步
            outputs = outputs[:, :max_length, :].squeeze(2)  # (batch_size, max_length)
            labels = labels[:, :max_length]  # (batch_size, max_length)

            # 创建掩码
            mask = torch.arange(max_length, device=device).expand(
                batch_size, max_length
            ) < lengths.unsqueeze(
                1
            )  # (batch_size, max_length)

            # 选择有效的输出和标签
            outputs_selected = outputs.masked_select(mask).view(-1)  # (N,)
            labels_selected = labels.masked_select(mask).view(-1)  # (N,)
            return outputs_selected, labels_selected, mask

        def calculate_accuracy(
            outputs,
            labels,
            lengths,
            resolution_kinds,
            kind_counter,
            kind_correct_counter,
            device,
            epoch_confusion_matrix=None,
            num_classes=2,
        ):
            """
            计算批次的准确率，并按类别统计，同时更新 epoch 的混淆矩阵。

            参数:
                outputs (torch.Tensor): 模型输出，形状为 (batch_size, max_length, 1)。
                labels (torch.Tensor): 标签，形状为 (batch_size, max_length)。
                lengths (torch.Tensor): 每个序列的实际长度，形状为 (batch_size,)。
                resolution_kinds (list): 每个样本的类别标签，长度为 batch_size。
                kind_counter (defaultdict): 按类别统计样本总数。
                kind_correct_counter (defaultdict): 按类别统计正确预测数。
                device (torch.device): 当前设备。
                epoch_confusion_matrix (torch.Tensor): 当前 epoch 的混淆矩阵，形状为 (num_classes, num_classes)。
                num_classes (int): 类别数量，默认为 2（二分类）。

            返回:
                float: 当前 batch 对于编辑脚本的准确率。
            """
            batch_size, max_length, _ = outputs.size()
            max_length = lengths.max().item()
            max_length = min(max_length, outputs.size(1))

            # 截取有效的时间步
            outputs = outputs[:, :max_length, :].squeeze(2)  # (batch_size, max_length)
            labels = labels[:, :max_length]  # (batch_size, max_length)

            # 创建掩码
            mask = torch.arange(max_length, device=device).expand(
                batch_size, max_length
            ) < lengths.unsqueeze(
                1
            )  # (batch_size, max_length)

            # 计算预测结果
            predictions = (
                outputs >= 0
            ).int()  # (batch_size, max_length)  注意输出没有 sigmoid，所以这里是 >= 0
            correct_matrix = (
                predictions == labels.int()
            ).float() * mask.float()  # (batch_size, max_length)

            # 计算序列级正确率
            sequence_correct = (
                correct_matrix.sum(dim=1) == lengths.float()
            )  # (batch_size,)

            # 统计每个类别
            for i in range(batch_size):
                kind = resolution_kinds[i]
                kind_counter[kind] += 1
                if sequence_correct[i]:
                    kind_correct_counter[kind] += 1

            # 生成 batch 的混淆矩阵
            true_labels = labels.masked_select(mask).long()  # (total_valid_steps,)
            predicted_labels = predictions.masked_select(
                mask
            ).long()  # (total_valid_steps,)
            batch_confusion_matrix = torch.zeros(
                (num_classes, num_classes), dtype=torch.int64, device=device
            )
            for t, p in zip(true_labels, predicted_labels):
                batch_confusion_matrix[t, p] += 1

            batch_confusion_matrix = batch_confusion_matrix.to(
                epoch_confusion_matrix.device
            )
            # 累加到 epoch 的混淆矩阵
            epoch_confusion_matrix += batch_confusion_matrix

        if stage != "train":
            kind_counter = defaultdict(int)
            kind_correct_counter = defaultdict(int)
            label_prediction_cnt = torch.zeros(
                (2, 2), dtype=torch.int64, device=self.accelerator.device
            )  # 预测结果统计，对于每个编辑脚本的预测结果（接受/不接受）进行统计
            total_loss_in_epoch = 0

        if stage == "train":
            self.model.train()
        else:
            self.model.eval()

        for loaded_feats, labels, lengths, resolution_kinds in tqdm(
            data_loader,
            dynamic_ncols=True,
            desc=f"{stage} Epoch {epoch + 1}/{epochs}",
            disable=not self.accelerator.is_main_process,
        ):
            assert len(lengths) == len(resolution_kinds)
            curr_batch_size = len(lengths)

            if stage == "train":
                outputs = self.model(
                    loaded_feats, lengths
                )  # model 调用中先 pack_padded_sequence，然后再 pad_packed_sequence
            else:
                with torch.no_grad():
                    outputs = self.model(loaded_feats, lengths)

                # 统计正确率
                calculate_accuracy(
                    outputs,
                    labels,
                    lengths,
                    resolution_kinds,
                    kind_counter,
                    kind_correct_counter,
                    outputs.device,
                    label_prediction_cnt,
                    2,
                )

            outputs_selected, labels_selected, _ = mask_output(outputs, labels, lengths)
            loss = self.criterion(outputs_selected, labels_selected)

            if stage != "train":
                total_loss_in_epoch += loss.item()

            if (
                "extra_block_loss_weight" in training_param
                and training_param.extra_block_loss_weight > 0
            ):
                # 加入块级别的 loss：如果一个块中有任一编辑脚本预测错误，则该块所有编辑脚本的 loss 权重加大
                start = 0
                extra_loss_weight = training_param.extra_block_loss_weight
                for length in lengths:
                    end = start + length
                    # 如果 outputs_selected[start:end] 中有任一元素和 labels_selected[start:end] 不同，则 loss 加一个额外的权重
                    if not (
                        outputs_selected[start:end] == labels_selected[start:end]
                    ).all():
                        loss += extra_loss_weight * self.criterion(
                            outputs_selected[start:end], labels_selected[start:end]
                        )
                    start = end

            if stage == "train":
                # 重置梯度
                self.optimizer.zero_grad()
                # 反向传播和优化
                self.accelerator.backward(loss)
                self.optimizer.step()

        if stage == "train":
            return None, None
        else:
            # ============= 合并字典 =============
            merged_kind_counter = gather_dict_from_all_procs(kind_counter)
            merged_kind_correct_counter = gather_dict_from_all_procs(
                kind_correct_counter
            )

            # ============= 合并张量 =============
            self.accelerator.reduce(label_prediction_cnt, reduction="sum")

            # ============= 合并 total_loss_in_epoch/total_batches =============
            # 先把它们变成 tensor 方便 reduce
            local_loss_t = torch.tensor(
                [total_loss_in_epoch], dtype=torch.float, device=self.accelerator.device
            )
            local_batch_t = torch.tensor(
                [len(data_loader)], device=self.accelerator.device
            )

            self.accelerator.reduce(local_loss_t, reduction="sum")
            self.accelerator.reduce(local_batch_t, reduction="sum")

            # 只有主进程做最终计算并返回
            if self.accelerator.is_main_process:
                global_loss = local_loss_t.item()
                global_batches = local_batch_t.item()
                average_loss = global_loss / global_batches  # 全局平均loss

                total_es_correct = label_prediction_cnt.diagonal().sum().item()
                total_es_valid = label_prediction_cnt.sum().item()
                es_accuracy = (
                    total_es_correct / total_es_valid if total_es_valid > 0 else 0.0
                )

                correct_num = sum(merged_kind_correct_counter.values())
                total_num = sum(merged_kind_counter.values())
                block_acc = round(correct_num / total_num * 100, 2)

                print("average loss:", average_loss, file=log_file)
                print(f"{stage} in epoch {epoch}, Accuracy: {block_acc}%", file=log_file)
                for key in merged_kind_counter.keys():
                    print(
                        f"\t{key} accuracy: {merged_kind_correct_counter[key] / merged_kind_counter[key] * 100:.2f}% ({merged_kind_correct_counter[key]}/{merged_kind_counter[key]})",
                        file=log_file,
                    )
                print(f"es_accuracy: {es_accuracy * 100:.2f}%", file=log_file)
                print("label_prediction_cnt:", label_prediction_cnt, file=log_file)
                print("\n\n", file=log_file)
                if log_file:
                    log_file.flush()

                return average_loss, block_acc
            else:
                # 对于非主进程，直接 return None, None
                return None, None

    from pathlib import PosixPath

    def train(
        self,
        model_output_path: PosixPath,
        date_str: str,
    ) -> pd.DataFrame:
        train_state_file_name = f"train_state_{date_str}"

        if self.accelerator.is_main_process:
            with open(train_state_file_name, "w") as f:
                f.write("1")
        self.accelerator.wait_for_everyone()
        
        epochs = training_param.epochs
        batch_size = training_param.batch_size
        log_file = self.log_file

        # 构造返回的 DataFrame，用于记录训练过程中的指标
        metric = []

        best_val_acc = -1
        patience = 3
        bad_count = 0
        for epoch in range(epochs):
            with open(train_state_file_name, "r") as f:
                if f.read() == "0":
                    break
            # 训练循环
            train_loss, train_acc = self.forward_in_stage(
                self.train_loader, "train", epoch, epochs, log_file
            )
            if self.scheduler:
                self.scheduler.step()

            # 验证循环
            val_loss, val_acc = self.forward_in_stage(
                self.val_loader, "val", epoch, epochs, log_file
            )

            if not self.accelerator.is_main_process:
                assert val_loss is None and val_acc is None
            else:
                metric.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                    }
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    bad_count = 0
                    # 保存模型
                    os.makedirs(model_output_path, exist_ok=True)
                    torch.save(
                        self.accelerator.unwrap_model(self.model).state_dict(),
                        model_output_path / f"best_model.pth",
                    )
                else:
                    bad_count += 1
                    if bad_count >= patience:
                        print(
                            "\n\n\n********************************************",
                            file=log_file,
                        )
                        print(f"Early stopping at epoch {epoch + 1}", file=log_file)
                        with open(train_state_file_name, "w") as f:
                            f.write("0")

            self.accelerator.wait_for_everyone()

        
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(
                torch.load(model_output_path / f"best_model.pth", weights_only=True)
            )
        else:
            self.model.load_state_dict(
                torch.load(model_output_path / f"best_model.pth", weights_only=True)
            )
        test_loss, test_acc = self.forward_in_stage(
            self.test_loader, "test", epoch, epochs, log_file
        )

        if log_file:
            log_file.close()
        if self.accelerator.is_main_process:
            # 删除 train_state 文件
            os.remove(train_state_file_name)
        return pd.DataFrame(metric)
