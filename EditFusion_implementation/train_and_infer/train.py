from .model.CCEmbedding.MergeBertCCEmbedding import MergeBertCCEmbedding
from .model.CCEmbedding.PretrainedCCEmbedding import PretrainedCCEmbedding
from .model.CCEmbedding.PretrainedCCEmbedding_linear import PretrainedCCEmbedding_linear
from .model.CCEmbedding.PretrainedCCEmbedding_seq import PretrainedCCEmbeddingSeq
from .trainers.LSTMTrainer import LSTMTrainer
from .trainers.GRUTrainer import GRUTrainer
from .model.LSTM_model import LSTMClassifier
from .model.GRU_model import GRUClassifier
from .params import model_params, training_param
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch.nn as nn
import torch
from pathlib import Path
import os
import time
import json
import random
import numpy as np
import argparse
import torch.distributed as dist


def seed_everything(seed=42):
    """
    Set a seed for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for edit script resolution')
    parser.add_argument('--model', type=str, choices=['lstm', 'gru'], default='lstm',
                        help='Model type to use (lstm or gru)')
    args = parser.parse_args()
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    seed_everything()
    script_path = Path(os.path.dirname(os.path.abspath(__file__)))
    
    if accelerator.is_main_process:
        date = time.strftime("%m-%d-%H:%M:%S", time.localtime())
    else:
        date = ""
    
    if accelerator.is_main_process:
        with open("temp_date.txt", "w") as f:
            f.write(date)
    
    accelerator.wait_for_everyone()
    
    if not accelerator.is_main_process:
        with open("temp_date.txt", "r") as f:
            date = f.read().strip()
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if os.path.exists("temp_date.txt"):
            os.remove("temp_date.txt")
    
    accelerator.wait_for_everyone()
    
    accelerator.print("date:", date)

    dataset_name = "codebert_mergebert_all_lang"
    
    model_type = args.model
    model_output_path = (
        script_path / "data" / "model_output" / (f"{model_type}_{dataset_name}_{date}")
    )
    model_params_output_name = f"model_params_{date}_{dataset_name}.json"
    training_params_output_name = f"training_params_{date}_{dataset_name}.json"

    dataset_path = (
        script_path / "data" / "processed_data" / dataset_name
    )

    accelerator.print(f"Initializing {model_type.upper()} model...")
    
    if model_type == 'gru':
        model = GRUClassifier(
            **model_params, CCEmbedding_class=PretrainedCCEmbedding
        )
        trainer_class = GRUTrainer
    else:
        model = LSTMClassifier(
            **model_params, CCEmbedding_class= MergeBertCCEmbedding
        )
        trainer_class = LSTMTrainer

    if accelerator.is_main_process:
        training_params_output_path = model_output_path / training_params_output_name
        model_params_output_path = model_output_path / model_params_output_name
        os.makedirs(os.path.dirname(training_params_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(model_params_output_path), exist_ok=True)

        with open(training_params_output_path, "w") as f:
            f.write(json.dumps(training_param))
        with open(model_params_output_path, "w") as f:
            f.write(json.dumps(model_params))
        log_file = model_output_path / "log.txt"
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}\n\n")
    else:
        log_file = None

    trainer = trainer_class(
        dataset_path, model, accelerator, debug=False, log_file=log_file
    )

    train_res_df = trainer.train(model_output_path=model_output_path, date_str=date)

    if train_res_df is not None:

        train_res_df.insert(0, "dataset", dataset_name)
        train_res_df.insert(1, "cc_embedding_lr", training_param.cc_embedding_lr)
        train_res_df.insert(2, "lstm_lr", training_param.lstm_lr)
        train_res_df.insert(3, "fc_lr", training_param.fc_lr)
        train_res_df.insert(4, "batch_size", training_param.batch_size)
        train_res_df["hidden_size"] = model_params["hidden_size"]
        train_res_df["num_layers"] = model_params["num_layers"]
        train_res_df["bidir"] = model_params["bidirectional"]
        train_res_df["dropout"] = model_params["dropout"]

        train_res_df.to_csv(model_output_path / "train_res.csv", index=False)
