import torch
import argparse
from model import MergeBERTModel
from utils import token_level_merge, align_and_get_edit_sequence, apply_pattern
from transformers import AutoTokenizer

# Constants
MODEL_PATH = "/home/foril/projects/EditFusion/mergebert_replication/CodeBERTa-small-v1/"
SAVED_MODEL_PATH = "/home/foril/projects/EditFusion/mergebert_replication/saved_models/mergebert.pt"
MAX_LENGTH = 512
NUM_LABELS = 9

# Reverse of the LABEL_MAP in preprocess.py
REVERSE_LABEL_MAP = {
    0: "select_a",
    1: "select_b",
    2: "select_o",
    3: "concat_ab",
    4: "concat_ba",
    5: "select_a_del_o",
    6: "select_b_del_o",
    7: "concat_ab_del_o",
    8: "concat_ba_del_o",
}

def main():
    """Main function to run inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_a", type=str, required=True, help="Path to file A (yours)")
    parser.add_argument("--file_b", type=str, required=True, help="Path to file B (theirs)")
    parser.add_argument("--file_o", type=str, required=True, help="Path to file O (base)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load model
    model = MergeBERTModel(encoder_path=MODEL_PATH, num_labels=NUM_LABELS)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Read file contents
    with open(args.file_a, 'r', encoding='utf-8') as f:
        a_text = f.read()
    with open(args.file_b, 'r', encoding='utf-8') as f:
        b_text = f.read()
    with open(args.file_o, 'r', encoding='utf-8') as f:
        o_text = f.read()

    # Preprocessing
    a_tokens = tokenizer.tokenize(a_text)
    b_tokens = tokenizer.tokenize(b_text)
    o_tokens = tokenizer.tokenize(o_text)

    hunks = token_level_merge(a_tokens, b_tokens, o_tokens)
    conflict_hunks = [h for h in hunks if isinstance(h, tuple)]

    if len(conflict_hunks) != 1:
        print("Could not resolve conflict: Found 0 or more than 1 token-level conflicts.")
        return

    conflict_idx = -1
    for i, hunk in enumerate(hunks):
        if isinstance(hunk, tuple):
            conflict_idx = i
            break
    
    prefix_tokens = [token for hunk in hunks[:conflict_idx] for token in hunk]
    suffix_tokens = [token for hunk in hunks[conflict_idx+1:] for token in hunk]
    a_conflict, b_conflict, o_conflict = conflict_hunks[0]

    # Prepare data for model
    a_o_aligned, o_a_aligned, _ = align_and_get_edit_sequence(a_tokens, o_tokens)
    b_o_aligned, o_b_aligned, _ = align_and_get_edit_sequence(b_tokens, o_tokens)

    def process_sequence(seq):
        ids = tokenizer.convert_tokens_to_ids(seq)
        if len(ids) > MAX_LENGTH:
            ids = ids[:MAX_LENGTH]
        else:
            ids += [tokenizer.pad_token_id] * (MAX_LENGTH - len(ids))
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    a_o_ids = process_sequence(a_o_aligned)
    o_a_ids = process_sequence(o_a_aligned)
    b_o_ids = process_sequence(b_o_aligned)
    o_b_ids = process_sequence(o_b_aligned)

    # Run inference
    with torch.no_grad():
        outputs = model(a_o_ids, o_a_ids, b_o_ids, o_b_ids)
        logits = outputs['logits']
        prediction = torch.argmax(logits, dim=1).item()

    # Decode and reconstruct
    predicted_pattern = REVERSE_LABEL_MAP.get(prediction)
    if not predicted_pattern:
        print(f"Error: Unknown pattern predicted with ID: {prediction}")
        return

    resolved_conflict = apply_pattern(predicted_pattern, a_conflict, b_conflict, o_conflict)
    reconstructed_tokens = prefix_tokens + resolved_conflict + suffix_tokens
    resolved_text = tokenizer.convert_tokens_to_string(reconstructed_tokens)

    print("--- Resolved Code ---")
    print(resolved_text)

if __name__ == "__main__":
    main()
