# This script will preprocess the raw JSON data from the dataset
# into a format suitable for training the MergeBERT model.

import os
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import tokenize_and_token_level_diff3, align_and_get_edit_sequence

# Constants
DATA_DIR = "/home/foril/projects/EditFusion/EditFusion_implementation/train_and_infer/data/gathered_data/mergebert_all_lang_with_no_newline/"
OUTPUT_DIR = "/home/foril/projects/EditFusion/mergebert_replication/data/"
MODEL_PATH = "/home/foril/projects/EditFusion/mergebert_replication/CodeBERTa-small-v1/"
MAX_LENGTH = 512

# We need to map the string labels to integer IDs.
# For the MVP, we'll only use the basic patterns.
LABEL_MAP = {
    "A": 0,
    "B": 1,
    "O": 2,
    "AB": 3,
    "BA": 4,
}
# The paper mentions 9 patterns, but we'll start with these 5.

def preprocess_data(file_path, tokenizer):
    """Processes a single raw data file."""
    processed_samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Processing {os.path.basename(file_path)}"):
        if 'conflict_chunks' not in item:
            continue

        for chunk in item['conflict_chunks']:
            label = chunk.get('label')
            if label not in LABEL_MAP:
                continue

            a_chunk = chunk.get('a_content', '')
            b_chunk = chunk.get('b_content', '')
            o_chunk = chunk.get('o_content', '')

            # Perform token-level diff3
            token_conflicts = tokenize_and_token_level_diff3(a_chunk, b_chunk, o_chunk, tokenizer)

            # MVP Simplification: Only handle one-to-one mappings
            if len(token_conflicts) != 1:
                continue
            
            conflict = token_conflicts[0]
            a_tokens = conflict['a']
            b_tokens = conflict['b']
            o_tokens = conflict['o']

            # Align sequences and get edit scripts
            a_o_aligned, o_a_aligned, delta_ao = align_and_get_edit_sequence(a_tokens, o_tokens)
            b_o_aligned, o_b_aligned, delta_bo = align_and_get_edit_sequence(b_tokens, o_tokens)

            # Convert tokens to IDs and pad/truncate
            def process_sequence(seq):
                ids = tokenizer.convert_tokens_to_ids(seq)
                if len(ids) > MAX_LENGTH:
                    ids = ids[:MAX_LENGTH]
                else:
                    ids += [tokenizer.pad_token_id] * (MAX_LENGTH - len(ids))
                return ids

            a_o_ids = process_sequence(a_o_aligned)
            o_a_ids = process_sequence(o_a_aligned)
            b_o_ids = process_sequence(b_o_aligned)
            o_b_ids = process_sequence(o_b_aligned)
            
            # For the MVP, we are not yet creating the edit type embeddings.
            # We will just store the processed token sequences.
            
            processed_samples.append({
                "a_o_aligned": a_o_ids,
                "o_a_aligned": o_a_ids,
                "b_o_aligned": b_o_ids,
                "o_b_aligned": o_b_ids,
                # "delta_ao": delta_ao, # Store for later use
                # "delta_bo": delta_bo, # Store for later use
                "label": LABEL_MAP[label]
            })

    return processed_samples

def main():
    """Main function to run the preprocessing."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # For the MVP, we'll process one file.
    # In a full implementation, we would process all files.
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON files found in {DATA_DIR}")
        return

    all_samples = []
    for file_name in json_files:
        file_path = os.path.join(DATA_DIR, file_name)
        all_samples.extend(preprocess_data(file_path, tokenizer))

    # Shuffle and split the data
    random.shuffle(all_samples)
    
    # 80/20 split for train/validation
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"Total samples: {len(all_samples)}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    # Save the processed data
    with open(os.path.join(OUTPUT_DIR, "train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_samples, f)
        
    with open(os.path.join(OUTPUT_DIR, "validation.json"), 'w', encoding='utf-8') as f:
        json.dump(val_samples, f)

    print(f"Processed data saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
