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
    """Merge a list of defaultdict(int)."""
    merged = defaultdict(int)
    for d in dicts_list:
        for k, v in d.items():
            merged[k] += v
    return merged


def gather_dict_from_all_procs(local_dict, only_main_process=True):
    """
    Use dist.all_gather_object to collect dictionaries (defaultdict(int)) from all processes,
    and return the merged dictionary. Only returns the merged result on the main process,
    other processes return None.

    Args:
      local_dict: The defaultdict(int) on the current process.
      only_main_process: Whether to return the merged dictionary only on the main process,
                         with other processes returning None.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return local_dict

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    dict_list = [None for _ in range(world_size)]

    dist.all_gather_object(dict_list, local_dict)

    if only_main_process and rank != 0:
        return None
    else:
        merged_dict = merge_defaultdict_int(dict_list)
        return merged_dict


class GRUTrainer:
    def __init__(
        self, dataset_path, model, accelerator, debug: bool = False, log_file=None
    ):
        if debug:
            self.log_file = None
        else:
            if log_file and not os.path.exists(
                log_file if isinstance(log_file, str) else log_file.name
            ):
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                self.log_file = open(log_file, "w")
            else:
                self.log_file = log_file

        batch_size = training_param.batch_size

        self.accelerator = accelerator
        device = self.accelerator.device
        print(f"Using device: {device}")

        self.dataset_path = dataset_path
        self.model = model
        self.train_split = training_param.TRAIN_SPLIT
        self.val_split = training_param.VAL_SPLIT

        if "pos_weight" in training_param:
            pos_weight = torch.tensor([training_param.pos_weight], device=device)
        else:
            pos_weight = torch.tensor([1.0], device=device)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        pretrained_params = model.CCEmbedding_class.cc_embedding.parameters()
        if hasattr(model.CCEmbedding_class, "attention"):
            attention_params = model.CCEmbedding_class.attention.parameters()
        else:
            attention_params = []
        if hasattr(model.CCEmbedding_class, "origin_embedding"):
            origin_embedding_params = model.CCEmbedding_class.origin_embedding.parameters()
        else:
            origin_embedding_params = []
        gru_params = model.gru.parameters()
        fc_params = model.fc.parameters()
        self.optimizer = optim.Adam(
            [
                {"params": pretrained_params, "lr": training_param.cc_embedding_lr},
                {"params": origin_embedding_params, "lr": training_param.cc_embedding_lr},
                {"params": attention_params, "lr": training_param.attention_lr},
                {"params": gru_params, "lr": training_param.lstm_lr},
                {"params": fc_params, "lr": training_param.fc_lr},
            ]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=training_param.scheduler_step_size,
            gamma=training_param.scheduler_gamma,
        )

        GRU_dataset = self.model.CCEmbedding_class.get_dataset(self.dataset_path)
        self.accelerator.print(f"Dataset size: {len(GRU_dataset)}")

        if debug:
            GRU_dataset = Subset(GRU_dataset, range(50))

        total_size = len(GRU_dataset)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            GRU_dataset, [train_size, val_size, test_size]
        )

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
            Masks the output and labels based on the actual sequence lengths.

            Args:
                outputs (torch.Tensor): Model output, shape (batch_size, seq_len, 1).
                labels (torch.Tensor): Labels, shape (batch_size, seq_len).
                lengths (torch.Tensor): Actual length of each sequence, shape (batch_size,).

            Returns:
                outputs_selected (torch.Tensor): Selected output, shape (N,).
                labels_selected (torch.Tensor): Selected labels, shape (N,).
            """
            device = outputs.device
            batch_size, seq_len, _ = outputs.size()
            max_length = lengths.max().item()
            max_length = min(max_length, seq_len)

            outputs = outputs[:, :max_length, :].squeeze(2)
            labels = labels[:, :max_length]

            mask = torch.arange(max_length, device=device).expand(
                batch_size, max_length
            ) < lengths.unsqueeze(
                1
            )

            outputs_selected = outputs.masked_select(mask).view(-1)
            labels_selected = labels.masked_select(mask).view(-1)
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
            Calculate the accuracy of the batch, and count by class, while updating the epoch's confusion matrix.

            Args:
                outputs (torch.Tensor): Model output, shape (batch_size, max_length, 1).
                labels (torch.Tensor): Labels, shape (batch_size, max_length).
                lengths (torch.Tensor): Actual length of each sequence, shape (batch_size,).
                resolution_kinds (list): Class label for each sample, length batch_size.
                kind_counter (defaultdict): Count of total samples by class.
                kind_correct_counter (defaultdict): Count of correct predictions by class.
                device (torch.device): Current device.
                epoch_confusion_matrix (torch.Tensor): Confusion matrix for the current epoch, shape (num_classes, num_classes).
                num_classes (int): Number of classes, default is 2 (binary classification).

            Returns:
                float: Accuracy of the current batch for the edit script.
            """
            batch_size, max_length, _ = outputs.size()
            max_length = lengths.max().item()
            max_length = min(max_length, outputs.size(1))

            outputs = outputs[:, :max_length, :].squeeze(2)
            labels = labels[:, :max_length]

            mask = torch.arange(max_length, device=device).expand(
                batch_size, max_length
            ) < lengths.unsqueeze(
                1
            )

            predictions = (
                outputs >= 0
            ).int()
            correct_matrix = (
                predictions == labels.int()
            ).float() * mask.float()

            sequence_correct = (
                correct_matrix.sum(dim=1) == lengths.float()
            )

            for i in range(batch_size):
                kind = resolution_kinds[i]
                kind_counter[kind] += 1
                if sequence_correct[i]:
                    kind_correct_counter[kind] += 1

            true_labels = labels.masked_select(mask).long()
            predicted_labels = predictions.masked_select(
                mask
            ).long()
            batch_confusion_matrix = torch.zeros(
                (num_classes, num_classes), dtype=torch.int64, device=device
            )
            for t, p in zip(true_labels, predicted_labels):
                batch_confusion_matrix[t, p] += 1

            batch_confusion_matrix = batch_confusion_matrix.to(
                epoch_confusion_matrix.device
            )
            epoch_confusion_matrix += batch_confusion_matrix

        if stage != "train":
            kind_counter = defaultdict(int)
            kind_correct_counter = defaultdict(int)
            label_prediction_cnt = torch.zeros(
                (2, 2), dtype=torch.int64, device=self.accelerator.device
            )
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
                )
            else:
                with torch.no_grad():
                    outputs = self.model(loaded_feats, lengths)

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
                start = 0
                extra_loss_weight = training_param.extra_block_loss_weight
                for length in lengths:
                    end = start + length
                    if not (
                        outputs_selected[start:end] == labels_selected[start:end]
                    ).all():
                        loss += extra_loss_weight * self.criterion(
                            outputs_selected[start:end], labels_selected[start:end]
                        )
                    start = end

            if stage == "train":
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

        if stage == "train":
            return None, None
        else:
            merged_kind_counter = gather_dict_from_all_procs(kind_counter)
            merged_kind_correct_counter = gather_dict_from_all_procs(
                kind_correct_counter
            )

            self.accelerator.reduce(label_prediction_cnt, reduction="sum")

            local_loss_t = torch.tensor(
                [total_loss_in_epoch], dtype=torch.float, device=self.accelerator.device
            )
            local_batch_t = torch.tensor(
                [len(data_loader)], device=self.accelerator.device
            )

            self.accelerator.reduce(local_loss_t, reduction="sum")
            self.accelerator.reduce(local_batch_t, reduction="sum")

            if self.accelerator.is_main_process:
                global_loss = local_loss_t.item()
                global_batches = local_batch_t.item()
                average_loss = global_loss / global_batches

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

        metric = []

        best_val_acc = -1
        patience = 3
        bad_count = 0
        for epoch in range(epochs):
            with open(train_state_file_name, "r") as f:
                if f.read() == "0":
                    break
            train_loss, train_acc = self.forward_in_stage(
                self.train_loader, "train", epoch, epochs, log_file
            )
            if self.scheduler:
                self.scheduler.step()

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
            os.remove(train_state_file_name)
        return pd.DataFrame(metric)
