# Provides model inference interface for flask API service
import os
import torch
from pathlib import Path
from typing import List
from train_and_infer.collect_dataset import process_to_legal_ids
from train_and_infer.utils.conflict_utils import Conflict

from train_and_infer.utils.es_generator import compute, SequenceDiff, get_edit_sequence
from train_and_infer.utils.tokenizer_util import encode_text_to_tokens, encode_tokens_to_ids
from .model.LSTM_model import LSTMClassifier, model_params

MAX_TOKEN_LEN = 32
# Singleton pattern to ensure model is loaded only once
model_singleton = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} for inference')

def load_model():
    """
    Load the model and ensure it is loaded only once.
    """
    global model_singleton
    # Check if model instance is already created
    if model_singleton is None:
    # Create model instance
        model = LSTMClassifier(**model_params)
        script_path = Path(os.path.dirname(os.path.abspath(__file__)))
        model_path = script_path / 'data' / 'Bert_embedding_bs2_lr5e-06.pth'
        model.load_state_dict(torch.load(model_path))

    # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
    # Store the created model instance in the global variable
        model_singleton = model
    return model_singleton

class EditScriptWithFrom:
    def __init__(self, sd: SequenceDiff, _from: str) -> None:
        self.es = sd
        self._from = _from

def get_predicted_result(base: List[str], ours: List[str], theirs: List[str]):
    """
    Predict the edit script between two strings
    Args:
        base: original string
        ours: first modified string
        theirs: second modified string
    Returns:
        generated result
    """
    # Ensure the model is loaded before use
    model = load_model()
    
    conflict = Conflict(ours, theirs, base)
    # 1. Generate edit scripts from three versions
    ess_ours = compute(base, ours)
    ess_theirs = compute(base, theirs)
    ess_with_from = [EditScriptWithFrom(sd, 'ours') for sd in ess_ours] + [EditScriptWithFrom(sd, 'theirs') for sd in ess_theirs]
    ess_with_from.sort(key=lambda es: es.es.seq1Range)

    # 2. For a conflict chunk, extract features from each edit script
    def extract_features(ess_with_from: List[EditScriptWithFrom], conflict: Conflict) -> torch.Tensor:
        """
        Extract features from edit scripts
        Returns:
            feature vector
        """
        def process_to_str(content: List[str]) -> str:
            if content in [[''], []]:
                return ''
            return '\n'.join(content) + '\n'
        
        position_features = []
        all_edit_seq_processed_ids = []
        all_origin_processed_ids = []
        all_modified_processed_ids = []
        for es_with_from in ess_with_from:
            es = es_with_from.es
            # Extract features in order
            origin_start = es.seq1Range.start
            origin_end = es.seq1Range.end
            modified_start = es.seq2Range.start
            modified_end = es.seq2Range.end
            origin_length = origin_end - origin_start
            modified_length = modified_end - modified_start
            length_diff = modified_length - origin_length
            position_features.append([origin_start, origin_end, modified_start, modified_end, origin_length, modified_length, length_diff])

            # Build semantic features
            origin_content_str = process_to_str(conflict.base[origin_start:origin_end])
            modified_tmp = conflict.ours if es_with_from._from == 'ours' else conflict.theirs
            modified_content_str = process_to_str(modified_tmp[modified_start:modified_end])
            # Truncate tokens if too many, otherwise edit sequence computation will be slow
            origin_tokens = encode_text_to_tokens(origin_content_str)[:3 * MAX_TOKEN_LEN]
            modified_tokens = encode_text_to_tokens(modified_content_str)[:3 * MAX_TOKEN_LEN]
            # Do not pad before computing edit sequence
            edit_seq_tokens, origin_padded_tokens, modified_padded_tokens = get_edit_sequence(origin_tokens, modified_tokens)

            # Truncate and pad ids
            edit_seq_processed_ids = process_to_legal_ids(encode_tokens_to_ids(edit_seq_tokens))
            origin_processed_ids = process_to_legal_ids(encode_tokens_to_ids(origin_padded_tokens))
            modified_processed_ids = process_to_legal_ids(encode_tokens_to_ids(modified_padded_tokens))

            all_edit_seq_processed_ids.append(edit_seq_processed_ids)
            all_origin_processed_ids.append(origin_processed_ids)
            all_modified_processed_ids.append(modified_processed_ids)


    # Convert to tensor
        all_origin_processed_ids = torch.tensor(all_origin_processed_ids, dtype=torch.long)
        all_modified_processed_ids = torch.tensor(all_modified_processed_ids, dtype=torch.long)
        all_edit_seq_processed_ids = torch.tensor(all_edit_seq_processed_ids, dtype=torch.long)

        return torch.tensor(position_features, dtype=torch.float), \
            (all_origin_processed_ids, all_modified_processed_ids, all_edit_seq_processed_ids)



    position_features, (origin_processed_ids, modified_processed_ids, edit_seq_processed_ids) = extract_features(ess_with_from, conflict)
    # NOTE: Currently does not consider batch_size, only one sample per inference. Can be optimized.

    # Move to corresponding device
    position_features = position_features.to(device)
    origin_processed_ids = origin_processed_ids.to(device)
    modified_processed_ids = modified_processed_ids.to(device)
    edit_seq_processed_ids = edit_seq_processed_ids.to(device)

    # 3. Input features to model to get prediction
    position_features = position_features.unsqueeze(0)  # Add batch_size dimension
    origin_processed_ids = origin_processed_ids.unsqueeze(0)
    modified_processed_ids = modified_processed_ids.unsqueeze(0)
    edit_seq_processed_ids = edit_seq_processed_ids.unsqueeze(0)

    lengths = torch.tensor([len(ess_with_from)]) # Only one sample, so length is 1

    with torch.inference_mode():
        outputs = model(position_features, origin_processed_ids, modified_processed_ids, edit_seq_processed_ids, lengths)
    outputs = outputs.squeeze(2)    # [batch_size, seq_len]
    outputs = outputs.round().int()

    # 4. Post-process: generate actual merged code from prediction
    def generate_resolution(ess_with_from: List[EditScriptWithFrom], labels: List[int], conflict: Conflict) -> List[str]:
        """
        Generate merged code from edit scripts and prediction results
        Args:
            ess: edit scripts (sorted)
            labels: prediction results (corresponds to ess order)
            conflict: conflict chunk
        Returns:
            merged code
        """
        resolution = []
        base_to_add = 0

        for es_with_from, label in zip(ess_with_from, labels):
            if label == 0:
                continue
            else:
                # Add this edit script's modification
                # Add previous base to resolution
                if es_with_from.es.seq1Range.start < base_to_add:
                    # NOTE: Not implemented: handling identical edits and concat, merging tokens at same position, or using larger prediction value
                    raise Exception('No solution provided, please resolve manually')
                resolution += conflict.base[base_to_add:es_with_from.es.seq1Range.start]
                # Add modified content to resolution
                modified_content = conflict.ours if es_with_from._from == 'ours' else conflict.theirs
                resolution += modified_content[es_with_from.es.seq2Range.start:es_with_from.es.seq2Range.end]
                base_to_add = es_with_from.es.seq1Range.end
        # Add remaining base to resolution
        resolution += conflict.base[base_to_add:]
        return resolution

    return generate_resolution(ess_with_from, outputs.tolist()[0], conflict)

if __name__ == '__main__':
    import json
    # Load data
    script_path = Path(os.path.dirname(os.path.abspath(__file__)))
    json_data_path = script_path / 'data' / 'self_collected_most_50.json'  # NOTE: If this file contains sensitive data, review before publishing.
    with open(json_data_path, 'r') as f:
        data = json.load(f)
        conflicts = [Conflict(conflict['ours'], conflict['theirs'],
                    conflict['base'], conflict['resolve'], conflict['resolution_kind']) for conflict in data]
    from tqdm import tqdm
    samples = conflicts[:1000]
    correct_cnt = 0
    for conflict in tqdm(samples):
        try:
            if (conflict.resolution == get_predicted_result(conflict.base, conflict.ours, conflict.theirs)):
                correct_cnt += 1
        except Exception as e:
            pass
    print(f'Accuracy: {correct_cnt / len(samples) * 100}%')