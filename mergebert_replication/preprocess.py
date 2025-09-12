import os
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import token_level_merge, align_and_get_edit_sequence, apply_pattern, compare_token_lists, normalize_code
import logging
from collections import Counter
from multiprocessing import Pool, cpu_count

# Constants
DATA_DIR = "/home/foril/projects/EditFusion/EditFusion_implementation/train_and_infer/data/gathered_data/mergebert_all_lang_with_no_newline/"
OUTPUT_DIR = "/home/foril/projects/EditFusion/mergebert_replication/data/"
MODEL_PATH = "/home/foril/projects/EditFusion/mergebert_replication/CodeBERTa-small-v1/"
MAX_LENGTH = 512

LABEL_MAP = {
    "select_a": 0,
    "select_b": 1,
    "select_o": 2,
    "concat_ab": 3,
    "concat_ba": 4,
    "select_a_del_o": 5,
    "select_b_del_o": 6,
    "concat_ab_del_o": 7,
    "concat_ba_del_o": 8,
}

# Global tokenizer for multiprocessing
tokenizer = None

def init_worker(model_path):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

def process_item(item):
    global tokenizer
    stats = Counter()
    samples = []
    detailed_stats = {
        "A": Counter(), "B": Counter(), "AB": Counter(), "BA": Counter(),
        "mixline": Counter(), "O": Counter(), "unknown": Counter()
    }

    if 'conflict_chunks' not in item:
        return samples, stats, detailed_stats

    for chunk in item['conflict_chunks']:
        stats["total_chunks"] += 1
        descriptive_label = chunk.get('label', 'unknown')
        if descriptive_label not in detailed_stats:
            descriptive_label = "unknown"
        detailed_stats[descriptive_label]["total"] += 1

        a_text = chunk.get('a_content', '')
        b_text = chunk.get('b_content', '')
        o_text = chunk.get('o_content', '')
        r_text = chunk.get('r_content', '')

        if a_text.count('\n') > 50 or b_text.count('\n') > 50 or o_text.count('\n') > 50:
            stats["large_chunks_skipped"] += 1
            detailed_stats[descriptive_label]["large_chunks_skipped"] += 1
            continue

        a_text_norm = normalize_code(a_text)
        b_text_norm = normalize_code(b_text)
        o_text_norm = normalize_code(o_text)
        r_text_norm = normalize_code(r_text)

        a_tokens = tokenizer.tokenize(a_text_norm)
        b_tokens = tokenizer.tokenize(b_text_norm)
        o_tokens = tokenizer.tokenize(o_text_norm)
        r_tokens = tokenizer.tokenize(r_text_norm)

        if len(a_tokens) > 1000 or len(b_tokens) > 1000 or len(o_tokens) > 1000 or len(r_tokens) > 1000:
            stats["token_limit_skipped"] += 1
            detailed_stats[descriptive_label]["token_limit_skipped"] += 1
            continue

        hunks = token_level_merge(a_tokens, b_tokens, o_tokens)
        conflict_hunks = [h for h in hunks if isinstance(h, tuple)]

        if len(conflict_hunks) == 0:
            stats["no_conflict_chunks"] += 1
            detailed_stats[descriptive_label]["no_conflict_chunks"] += 1
            merged_tokens = hunks[0] if hunks else []
            if compare_token_lists(merged_tokens, r_tokens):
                stats["pseudo_conflict_resolved_cleanly"] += 1
                detailed_stats[descriptive_label]["pseudo_conflict_resolved_cleanly"] += 1
            continue

        if len(conflict_hunks) > 1:
            stats["multi_conflict_chunks"] += 1
            detailed_stats[descriptive_label]["multi_conflict_chunks"] += 1
            continue
        
        conflict_idx = -1
        for i, hunk in enumerate(hunks):
            if isinstance(hunk, tuple):
                conflict_idx = i
                break
        
        prefix_tokens = [token for hunk in hunks[:conflict_idx] for token in hunk]
        suffix_tokens = [token for hunk in hunks[conflict_idx+1:] for token in hunk]
        a_conflict, b_conflict, o_conflict = conflict_hunks[0]

        found_resolution = False
        for pattern_name, label_id in LABEL_MAP.items():
            resolved_conflict = apply_pattern(pattern_name, a_conflict, b_conflict, o_conflict)
            if resolved_conflict is None: continue
            reconstructed_tokens = prefix_tokens + resolved_conflict + suffix_tokens

            if compare_token_lists(reconstructed_tokens[:MAX_LENGTH], r_tokens[:MAX_LENGTH]):
                stats["resolvable_chunks"] += 1
                detailed_stats[descriptive_label]["resolvable_chunks"] += 1
                stats[f"label_{pattern_name}"] += 1
                stats[f"descriptive_label_{chunk.get('label', 'unknown')}"] += 1
                
                a_o_aligned, o_a_aligned, _ = align_and_get_edit_sequence(a_tokens, o_tokens)
                b_o_aligned, o_b_aligned, _ = align_and_get_edit_sequence(b_tokens, o_tokens)

                def process_sequence(seq):
                    ids = tokenizer.convert_tokens_to_ids(seq)
                    if len(ids) > MAX_LENGTH: ids = ids[:MAX_LENGTH]
                    else: ids += [tokenizer.pad_token_id] * (MAX_LENGTH - len(ids))
                    return ids

                samples.append({
                    "a_o_aligned": process_sequence(a_o_aligned),
                    "o_a_aligned": process_sequence(o_a_aligned),
                    "b_o_aligned": process_sequence(b_o_aligned),
                    "o_b_aligned": process_sequence(o_b_aligned),
                    "numeric_label": label_id,
                    "descriptive_label": chunk.get('label', 'unknown')
                })
                found_resolution = True
                break
        
        if not found_resolution:
            stats["unresolvable_chunks"] += 1
            detailed_stats[descriptive_label]["unresolvable_chunks"] += 1
    return samples, stats, detailed_stats

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler("preprocessing.log"), logging.StreamHandler()])

    dataset_name = os.path.basename(os.path.normpath(DATA_DIR))
    processed_file_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_all_processed.json")

    logging.info(f"Starting preprocessing. Full log will be in preprocessing.log")
    if os.path.exists(processed_file_path):
        logging.warning(f"Found existing processed file: {processed_file_path}.")
        logging.warning("To generate new statistics, please delete this file and run the script again.")
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
    else:
        all_items = []
        json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
        for file_name in json_files:
            file_path = os.path.join(DATA_DIR, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_items.extend(data)
        
        # --- START of new code for sampling ---
        sample_size = int(len(all_items))
        all_items = all_items[:sample_size]
        logging.info(f"Running a preview on {len(all_items)} items (10% of total).")
        # --- END of new code for sampling ---

        all_samples = []
        overall_stats = Counter()
        overall_detailed_stats = {
            "A": Counter(), "B": Counter(), "AB": Counter(), "BA": Counter(),
            "mixline": Counter(), "O": Counter(), "unknown": Counter()
        }

        num_processes = 14
        logging.info(f"Using {num_processes} processes for preprocessing.")

        with Pool(processes=num_processes, initializer=init_worker, initargs=(MODEL_PATH,)) as pool:
            results = list(tqdm(pool.imap(process_item, all_items), total=len(all_items)))

        for samples, stats, detailed_stats in results:
            all_samples.extend(samples)
            overall_stats.update(stats)
            for label, counters in detailed_stats.items():
                if label in overall_detailed_stats:
                    overall_detailed_stats[label].update(counters)

        logging.info("\n--- Overall Data Processing Statistics ---")
        for key, value in sorted(overall_stats.items()):
            logging.info(f"{key}: {value}")
        logging.info("------------------------------------------\n")

        logging.info("\n--- Conflict Resolution Statistics by Category ---")
        for label, counters in sorted(overall_detailed_stats.items()):
            total = counters.get('total', 0)
            if total == 0:
                continue
            
            resolvable = counters.get('resolvable_chunks', 0)
            multi_conflict = counters.get('multi_conflict_chunks', 0)
            unresolvable = counters.get('unresolvable_chunks', 0)
            pseudo_conflict = counters.get('pseudo_conflict_resolved_cleanly', 0)
            
            resolvable_percent = (resolvable / total) * 100 if total > 0 else 0
            multi_conflict_percent = (multi_conflict / total) * 100 if total > 0 else 0
            unresolvable_percent = (unresolvable / total) * 100 if total > 0 else 0
            pseudo_conflict_percent = (pseudo_conflict / total) * 100 if total > 0 else 0

            logging.info(f"Category: {label} (Total: {total})")
            logging.info(f"  - Resolvable: {resolvable} ({resolvable_percent:.2f}%)")
            logging.info(f"  - Multi-conflict-chunks: {multi_conflict} ({multi_conflict_percent:.2f}%)")
            logging.info(f"  - Unresolvable: {unresolvable} ({unresolvable_percent:.2f}%)")
            logging.info(f"  - Pseudo-conflict resolved cleanly: {pseudo_conflict} ({pseudo_conflict_percent:.2f}%)")
        logging.info("--------------------------------------------------\n")

        with open(processed_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f)
        logging.info(f"Saved all {len(all_samples)} processed samples to {processed_file_path}")

    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    logging.info(f"Total samples for training/validation: {len(all_samples)}")
    logging.info(f"Training samples: {len(train_samples)}")
    logging.info(f"Validation samples: {len(val_samples)}")

    train_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_train.json")
    val_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_validation.json")

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f)
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f)

    logging.info(f"Processed data saved to {train_file} and {val_file}")

if __name__ == "__main__":
    main()