import os
import torch
import json
import re
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

from train_and_infer.utils.conflict_utils import Conflict
from train_and_infer.utils.es_generator import compute, SequenceDiff, get_edit_sequence
from train_and_infer.model.LSTM_model import LSTMClassifier
from train_and_infer.params import model_params
from train_and_infer.utils.model_util import load_model_param
from train_and_infer.model.CCEmbedding.MergeBertCCEmbedding import MergeBertCCEmbedding, EDIT_ID_MAP, EDIT_PADDING_ID
from train_and_infer.utils.tokenizer_util import (
    tokenizer,
    bos_token_id,
    eos_token_id,
    pad_token_id,
    encode_text_to_tokens,
    encode_tokens_to_ids,
)

model_singleton = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_TOKEN_LEN = 64


def normalize_text(text: str) -> str:
    """Removes empty lines, indents, and trailing white spaces."""
    lines = text.split('\n')
    processed_lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(processed_lines)

def process_to_legal_ids(tokens: List[int]) -> List[int]:
    """
    <s> + truncate + </s> + padding
    """
    if len(tokens) > MAX_TOKEN_LEN - 2:
        return [bos_token_id] + tokens[: MAX_TOKEN_LEN - 2] + [eos_token_id]
    else:
        return (
            [bos_token_id]
            + tokens
            + [eos_token_id]
            + [pad_token_id] * (MAX_TOKEN_LEN - 2 - len(tokens))
        )


def es_gen_str2list(content: str) -> List[str]:
    """
    Process string for edit script generation.
    """
    return [line.strip() for line in content.split("\n") if line.strip() != ""]



def load_model():
    """
    Load the model, ensuring it is loaded only once.
    """
    global model_singleton
    if model_singleton is None:
        print("Loading model...")
        model = LSTMClassifier(
            **model_params, CCEmbedding_class=MergeBertCCEmbedding
        )
        script_path = Path(os.path.dirname(os.path.abspath(__file__)))
        model_path = (
            script_path
            / "data"
            / "model_output"
            / "lstm_codebert_mergebert_all_lang_08-31-21:45:11"
            / "best_model.pth"
        )
        model, _ = load_model_param(model, model_path)
        model_singleton = model
        print("Model loaded.")
    return model_singleton


class EditScriptWithFrom:
    def __init__(self, sd: SequenceDiff, _from: str) -> None:
        self.es = sd
        self.from_id = _from


def get_predicted_result(chunk: dict):
    """
    Predict the resolution for a single conflict chunk.
    """
    model = load_model()
    model.eval()

    a_content_lines = es_gen_str2list(chunk["a_content"])
    b_content_lines = es_gen_str2list(chunk["b_content"])
    o_content_lines = es_gen_str2list(chunk["o_content"])

    from_ours = compute(o_content_lines, a_content_lines)
    from_theirs = compute(o_content_lines, b_content_lines)

    ess_with_from = [EditScriptWithFrom(sd, "ours") for sd in from_ours] + [
        EditScriptWithFrom(sd, "theirs") for sd in from_theirs
    ]
    ess_with_from.sort(key=lambda es: es.es.seq1Range)
    
    if len(ess_with_from) > model_params['max_es_len']:
        return "SKIPPED: Too many edit scripts", None, None

    if not ess_with_from:
        return "\n".join(o_content_lines), ess_with_from, []

    position_features_list = []
    origin_processed_ids_list = []
    modified_processed_ids_list = []
    edit_seq_processed_ids_list = []

    for es_with_from in ess_with_from:
        es = es_with_from.es
        origin_start, origin_end = es.seq1Range.start, es.seq1Range.end
        modified_start, modified_end = es.seq2Range.start, es.seq2Range.end

        origin_length = origin_end - origin_start
        modified_length = modified_end - modified_start
        length_diff = modified_length - origin_length
        position_features_list.append(
            [origin_start, origin_end, modified_start, modified_end, origin_length, modified_length, length_diff]
        )

        def process_to_str(content: List[str]) -> str:
            return "\n".join(content) + "\n" if content else ""

        origin_content_str = process_to_str(o_content_lines[origin_start:origin_end])
        modified_tmp = a_content_lines if es_with_from.from_id == "ours" else b_content_lines
        modified_content_str = process_to_str(modified_tmp[modified_start:modified_end])

        origin_tokens = encode_text_to_tokens(origin_content_str)
        modified_tokens = encode_text_to_tokens(modified_content_str)

        edit_seq_tokens, origin_padded_tokens, modified_padded_tokens = get_edit_sequence(
            origin_tokens, modified_tokens
        )

        origin_processed_ids = process_to_legal_ids(encode_tokens_to_ids(origin_padded_tokens))
        modified_processed_ids = process_to_legal_ids(encode_tokens_to_ids(modified_padded_tokens))
        edit_seq_processed_ids = process_to_legal_ids(encode_tokens_to_ids(edit_seq_tokens))
        
        edit_seq_processed_ids_new = [EDIT_ID_MAP.get(token_id, EDIT_ID_MAP[pad_token_id]) for token_id in edit_seq_processed_ids]

        origin_processed_ids_list.append(origin_processed_ids)
        modified_processed_ids_list.append(modified_processed_ids)
        edit_seq_processed_ids_list.append(edit_seq_processed_ids_new)

    position_features = torch.tensor(position_features_list, dtype=torch.float).unsqueeze(0).to(device)
    origin_ids = torch.tensor(origin_processed_ids_list, dtype=torch.long).unsqueeze(0).to(device)
    modified_ids = torch.tensor(modified_processed_ids_list, dtype=torch.long).unsqueeze(0).to(device)
    edit_seq_ids = torch.tensor(edit_seq_processed_ids_list, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(ess_with_from)]).to(device)

    feats = (position_features, origin_ids, modified_ids, edit_seq_ids)

    with torch.no_grad():
        outputs = model(feats, lengths)
    
    outputs = outputs.squeeze(0).squeeze(-1)
    predictions = (outputs >= 0).int()

    resolution_lines = []
    base_to_add = 0
    conflict = Conflict(a_content_lines, b_content_lines, o_content_lines)

    for es_with_from, label in zip(ess_with_from, predictions):
        if label == 1:
            if es_with_from.es.seq1Range.start < base_to_add:
                continue

            resolution_lines.extend(conflict.base[base_to_add : es_with_from.es.seq1Range.start])
            
            modified_content = conflict.ours if es_with_from.from_id == "ours" else conflict.theirs
            resolution_lines.extend(modified_content[es_with_from.es.seq2Range.start : es_with_from.es.seq2Range.end])
            
            base_to_add = es_with_from.es.seq1Range.end

    resolution_lines.extend(conflict.base[base_to_add:])
    return "\n".join(resolution_lines), ess_with_from, predictions

def compareInToken(a_ls: List[str], b_ls: List[str]) -> bool:
    def toUnifiedStr(ls: List[str]) -> str:
        tmp = re.sub(r"\s+", " ", "\n".join(ls).strip() + "\n")
        return "" if ls == [] or ls == [""] else tmp
    a_processed = toUnifiedStr(a_ls)
    b_processed = toUnifiedStr(b_ls)
    return a_processed == b_processed

def check_prediction_correctness(all_edit_scripts, predictions, chunk) -> Tuple[bool, Optional[List[str]]]:
    a_content_lines = es_gen_str2list(chunk["a_content"])
    b_content_lines = es_gen_str2list(chunk["b_content"])
    o_content_lines = es_gen_str2list(chunk["o_content"])
    r_content_lines = es_gen_str2list(chunk["r_content"])

    def bt_eval(generated, i, last_end) -> Optional[List[str]]:
        if i == len(all_edit_scripts):
            whole_generated = generated + o_content_lines[last_end:]
            if compareInToken(whole_generated, r_content_lines):
                return whole_generated
            return None

        if predictions[i] == 0:
            return bt_eval(generated, i + 1, last_end)

        es = all_edit_scripts[i]
        if es.es.seq1Range.start < last_end:
            return None

        start = es.es.seq2Range.start
        end = es.es.seq2Range.end
        curr_content = a_content_lines[start:end] if es.from_id == "ours" else b_content_lines[start:end]
        
        new_generated = generated + o_content_lines[last_end:es.es.seq1Range.start] + curr_content
        new_last_end = es.es.seq1Range.end

        solution = bt_eval(new_generated, i + 1, new_last_end)
        if solution is not None:
            return solution

        if (
            i + 1 < len(all_edit_scripts) and 
            predictions[i+1] == 1 and
            es.es.seq1Range == all_edit_scripts[i+1].es.seq1Range
        ):
            
            next_es = all_edit_scripts[i+1]
            next_start = next_es.es.seq2Range.start
            next_end = next_es.es.seq2Range.end
            next_content = a_content_lines[next_start:next_end] if next_es.from_id == "ours" else b_content_lines[next_start:next_end]

            solution = bt_eval(generated + o_content_lines[last_end:es.es.seq1Range.start] + next_content + curr_content, i + 2, new_last_end)
            if solution is not None:
                return solution

        return None

    result_lines = bt_eval([], 0, 0)
    return (result_lines is not None, result_lines)

if __name__ == "__main__":
    print(f"Using {device} for inference")
    load_model()
    
    script_path = Path(os.path.dirname(os.path.abspath(__file__)))
    json_file_path = script_path / "data" / "raw1400.json"
    output_file_path = script_path / "evaluation_results_all_chunks_final.txt"
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    label_counts = defaultdict(int)
    label_correct_counts = defaultdict(int)
    label_bleu_scores = defaultdict(float)
    label_ned_scores = defaultdict(float)
    skipped_chunks = 0
    total_chunks = 0

    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        for conflict_file in tqdm(data, desc="Evaluating Conflicts"):
            if not conflict_file["conflict_chunks"]:
                continue
            
            for chunk in conflict_file["conflict_chunks"]:
                total_chunks += 1
                label = chunk['label']

                default_predicted_res, all_scripts, predictions = get_predicted_result(chunk)

                if default_predicted_res == "SKIPPED: Too many edit scripts":
                    skipped_chunks += 1
                    out_file.write(f'--- Conflict Chunk (SKIPPED) ---\n')
                    out_file.write(f'Label: {label}\n')
                    out_file.write('Reason: Too many edit scripts\n')
                    out_file.write('\n' + '='*40 + '\n\n')
                    continue

                label_counts[label] += 1
                ground_truth = chunk["r_content"]

                is_correct, correct_lines = check_prediction_correctness(all_scripts, predictions, chunk)

                if is_correct:
                    predicted_resolution = "\n".join(correct_lines)
                else:
                    predicted_resolution = default_predicted_res

                norm_predicted = normalize_text(predicted_resolution)
                norm_ground_truth = normalize_text(ground_truth)

                bleu = sentence_bleu([norm_ground_truth.split()], norm_predicted.split())
                edit_distance = Levenshtein.distance(norm_predicted, norm_ground_truth)
                max_len = max(len(norm_predicted), len(norm_ground_truth))
                ned = edit_distance / max_len if max_len > 0 else 0

                label_bleu_scores[label] += bleu
                label_ned_scores[label] += ned
                
                out_file.write(f'--- Conflict Chunk ---\n')
                out_file.write(f'Label: {label}\n')
                out_file.write(f'BLEU: {bleu:.4f}, Normalized Edit Distance: {ned:.4f}\n')
                out_file.write(f'\n--- Predicted Resolution (Normalized) ---\n')
                out_file.write(norm_predicted + '\n')
                out_file.write(f'\n--- Ground Truth Resolution (Normalized) ---\n')
                out_file.write(norm_ground_truth + '\n')

                if is_correct:
                    label_correct_counts[label] += 1
                    out_file.write('\n[SUCCESS] Backtracking check passed.\n')
                else:
                    out_file.write('\n[FAILURE] Backtracking check failed.\n')
                
                out_file.write('\n' + '='*40 + '\n\n')

    # --- Final Summary --- #
    print("\n--- Evaluation Summary ---")
    analyzed_chunks = total_chunks - skipped_chunks
    total_correct = sum(label_correct_counts.values())
    overall_accuracy = (total_correct / analyzed_chunks) * 100 if analyzed_chunks > 0 else 0
    overall_bleu = sum(label_bleu_scores.values()) / analyzed_chunks if analyzed_chunks > 0 else 0
    overall_ned = sum(label_ned_scores.values()) / analyzed_chunks if analyzed_chunks > 0 else 0

    print(f"Total Chunks: {total_chunks}")
    print(f"Skipped Chunks (too many edit scripts): {skipped_chunks}")
    print(f"Analyzed Chunks: {analyzed_chunks}")
    print(f"Overall Accuracy (on analyzed chunks): {overall_accuracy:.2f}% ({total_correct}/{analyzed_chunks})")
    print(f"Overall BLEU Score: {overall_bleu:.4f}")
    print(f"Overall Normalized Edit Distance: {overall_ned:.4f}")

    print("\n--- Metrics by Label (on analyzed chunks) ---")
    for label, count in sorted(label_counts.items()):
        correct_count = label_correct_counts[label]
        accuracy = (correct_count / count) * 100 if count > 0 else 0
        avg_bleu = label_bleu_scores[label] / count if count > 0 else 0
        avg_ned = label_ned_scores[label] / count if count > 0 else 0
        print(f"- {label}:")
        print(f"  Accuracy: {accuracy:.2f}% ({correct_count}/{count})")
        print(f"  Avg BLEU: {avg_bleu:.4f}")
        print(f"  Avg NED: {avg_ned:.4f}")


    print(f"\nDetailed results saved to {output_file_path}")
