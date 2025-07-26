from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple
import json
from .utils.tokenizer_util import encode_text_to_tokens, encode_tokens_to_ids, pad_token_id, bos_token_id, eos_token_id
from .utils.conflict_utils import Conflict
from .utils.es_generator import compute, SequenceDiff, get_edit_sequence
from tqdm import tqdm
from collections import Counter
import pandas as pd
import os
import signal

 # Shuffle for debug
import random
random.seed(42)
script_path = Path(os.path.dirname(os.path.abspath(__file__)))
json_data_path = script_path / 'data' / 'gathered_data' / '2000repos.json'
output_file = script_path / 'data' / 'processed_data' / '2000_repos_token32.csv'


# Maximum token length for code change embedding (for padding and truncation)
MAX_TOKEN_LEN = 32

def process_to_legal_ids(tokens: List[int]) -> List[int]:
    '''
    <s> + truncate + </s> + padding
    '''
    if len(tokens) > MAX_TOKEN_LEN - 2:     # 留给 <s> 和 </s>
        return [bos_token_id] +tokens[:MAX_TOKEN_LEN - 2] + [eos_token_id]
    else:
        return [bos_token_id] + tokens + [eos_token_id] + [pad_token_id] * (MAX_TOKEN_LEN - 2 - len(tokens))


class EditScriptLabel:
    def __init__(self, block_id: int, sd: SequenceDiff, _from: str, accept: bool):
        self.edit_script = sd
        self._from = _from
        self.accept = accept
        self.block_id = block_id


def check_conflict_resolvable(id_conflick_tuple: Tuple[int, Conflict]) -> [pd.DataFrame, List[Conflict], str, bool, bool]:
    '''
    Collect conflicts that can be resolved with edit scripts and build the training dataset.
    Returns:
        resolvable_dataset: DataFrame of all edit scripts for resolvable conflicts
        conflicts_unable_to_resolve_with_edit_scripts: List of conflicts not resolvable with edit scripts (empty if resolvable)
        kind: Conflict resolution type
        line_too_many: Count of samples with too many lines
        script_num_too_many: Count of samples with too many edit scripts
    '''
    block_id, conflict = id_conflick_tuple


    def filter_empty_lines(lines: List[str]) -> List[str]:
        return [line for line in lines if line != '']

    def bt(generated, i, last_end, accept_mark: List[bool]) -> bool:
        '''
    Backtracking to generate all possible solutions; if matches resolution, add to result
        '''
        if i == len(all_edit_scripts):
            whole_generated = generated + conflict.base[last_end:]
            # 过滤 whole_generated 和 resolution 中的空行
            if filter_empty_lines(whole_generated) == filter_empty_lines(conflict.resolution):
                # 可以使用组合 ES 的方式解决的冲突
                for i, edit_script_label in enumerate(accept_mark):
                    all_edit_scripts[i].accept = edit_script_label
                return True
            return False

    # Do not accept this script
        accept_mark[i] = False
        if bt(generated, i + 1, last_end, accept_mark):
            return True

    # If current script's start position is less than last_end, it conflicts with previous script
    # Cannot accept this script, skip
        if all_edit_scripts[i].edit_script.seq1Range.start < last_end:
            return False     # 因为是小于号，所以可以解决伪冲突

    # Accept this script
        start = all_edit_scripts[i].edit_script.seq2Range.start
        end = all_edit_scripts[i].edit_script.seq2Range.end
        if all_edit_scripts[i]._from == 'ours':
            curr_content = conflict.ours[start:end]
        else:
            curr_content = conflict.theirs[start:end]
        accept_mark[i] = True
        if bt(generated
                + conflict.base[last_end:all_edit_scripts[i].edit_script.seq1Range.start]
                + curr_content,
                i + 1,
                all_edit_scripts[i].edit_script.seq1Range.end,
                accept_mark
            ):
            return True


        if (
            i + 1 < len(all_edit_scripts) and
            all_edit_scripts[i].edit_script.seq1Range == all_edit_scripts[i + 1].edit_script.seq1Range
        ):
            start = all_edit_scripts[i + 1].edit_script.seq2Range.start
            end = all_edit_scripts[i + 1].edit_script.seq2Range.end
            if all_edit_scripts[i + 1]._from == 'ours':
                next_content = conflict.ours[start:end]
            else:
                next_content = conflict.theirs[start:end]

            # Another way to concat
            if len(all_edit_scripts[i].edit_script.seq1Range) == 0:
                accept_mark[i + 1] = True
                if bt(generated
                        + conflict.base[last_end:all_edit_scripts[i].edit_script.seq1Range.start]
                        + next_content
                        + curr_content,
                    i + 2,
                    all_edit_scripts[i].edit_script.seq1Range.end,
                    accept_mark
                    ):
                    return True

    df_columns = [
        'block_id',
        'conflict_base',
        'conflict_ours',
        'conflict_theirs',
        'conflict_resolution',
        'origin_start',
        'origin_end',
        'modified_start',
        'modified_end',
        'origin_content',
        'modified_content',
        'from',
        'accept',
        'resolution_kind',
        'origin_tokens',
        'modified_tokens',
        'edit_seq_tokens',
        'origin_tokens_truncated',
        'modified_tokens_truncated',
        'edit_seq_tokens_truncated',
        'origin_ids_truncated',
        'modified_ids_truncated',
        'edit_seq_ids_truncated',
        'origin_processed_ids',
        'modified_processed_ids',
        'edit_seq_processed_ids',
    ]

    # Start collecting dataset
    kind = conflict.resolution_kind

    # Skip if conflict is of type 'newline'
    if kind == 'newline':
        return pd.DataFrame(columns=df_columns), [conflict], kind, False, False

    # Skip if any content has too many lines
    if any([len(content) > 1000 for content in [conflict.base, conflict.ours, conflict.theirs]]):
        return pd.DataFrame(columns=df_columns), [conflict], kind, True, False

    from_ours = compute(conflict.base, conflict.ours)
    from_theirs = compute(conflict.base, conflict.theirs)
    # Add identifier to distinguish between 'ours' and 'theirs'
    from_ours = [EditScriptLabel(block_id, sd, 'ours', False) for sd in from_ours]
    from_theirs = [EditScriptLabel(block_id, sd, 'theirs', False) for sd in from_theirs]

    all_edit_scripts = from_ours + from_theirs
    # Limit number of scripts to avoid excessive computation
    if len(all_edit_scripts) > 20:
        return pd.DataFrame(columns=df_columns), [conflict], kind, False, True

    all_edit_scripts.sort(key=lambda x: x.edit_script.seq1Range)

    if bt([], 0, 0, [False] * len(all_edit_scripts)):  # This conflict can be resolved
        # Add to dataset
        # Build a complete dataset including all conflict block content, edit script start/end, and source
        all_es_for_this_conflict = []
        for es in all_edit_scripts:
            origin_start = es.edit_script.seq1Range.start
            origin_end = es.edit_script.seq1Range.end
            modified_start = es.edit_script.seq2Range.start
            modified_end = es.edit_script.seq2Range.end

            def process_to_str(content: List[str]) -> str:
                if content in [[''], []]:
                    return ''
                return '\n'.join(content) + '\n'
            
            origin_content_str = process_to_str(conflict.base[origin_start:origin_end])
            modified_tmp = conflict.ours if es._from == 'ours' else conflict.theirs
            modified_content_str = process_to_str(modified_tmp[modified_start:modified_end])
            # Note: Empty lines are not processed here; they are part of the change but filtered during final comparison
            
            # Truncate tokens first
            origin_tokens = encode_text_to_tokens(origin_content_str)[:3 * MAX_TOKEN_LEN]           # 这里如果 tokens 太多，下面计算编辑序列会很慢
            modified_tokens = encode_text_to_tokens(modified_content_str)[:3 * MAX_TOKEN_LEN]       # 取 3 倍是为了防止对齐时找不到对应的 token，保留尽可能多信息
            # Do not pad before computing edit sequence
            edit_seq_tokens, origin_padded_tokens, modified_padded_tokens = get_edit_sequence(origin_tokens, modified_tokens)
            # TODO: Store ids before padding in the dataset
            edit_seq_tokens_truncated = edit_seq_tokens[:MAX_TOKEN_LEN - 2]
            origin_tokens_truncated = origin_padded_tokens[:MAX_TOKEN_LEN - 2]
            modified_tokens_truncated = modified_padded_tokens[:MAX_TOKEN_LEN - 2]
            
            # Store ids before padding in the dataset
            edit_seq_ids_truncated = encode_tokens_to_ids(edit_seq_tokens_truncated)
            origin_ids_truncated = encode_tokens_to_ids(origin_tokens_truncated)
            modified_ids_truncated = encode_tokens_to_ids(modified_tokens_truncated)

            # Truncate and pad ids
            edit_seq_processed_ids = process_to_legal_ids(encode_tokens_to_ids(edit_seq_tokens))
            origin_processed_ids = process_to_legal_ids(encode_tokens_to_ids(origin_padded_tokens))
            modified_processed_ids = process_to_legal_ids(encode_tokens_to_ids(modified_padded_tokens))

            
            all_es_for_this_conflict.append([
                es.block_id,
                conflict.base,                                  # 整个冲突块的 base: List[str]
                conflict.ours,
                conflict.theirs,
                conflict.resolution,
                origin_start,
                origin_end,
                modified_start,
                modified_end,
                origin_content_str,
                modified_content_str,
                es._from,
                es.accept,
                conflict.resolution_kind,
                origin_tokens,
                modified_tokens,
                edit_seq_tokens,
                origin_tokens_truncated,
                modified_tokens_truncated,
                edit_seq_tokens_truncated,
                origin_ids_truncated,
                modified_ids_truncated,
                edit_seq_ids_truncated,
                origin_processed_ids,
                modified_processed_ids,
                edit_seq_processed_ids,
            ])
        # Convert to DataFrame
        resolvable_dataset = pd.DataFrame(all_es_for_this_conflict, columns=df_columns)
        return resolvable_dataset, [], kind, False, False
    else:
        return pd.DataFrame(columns=df_columns), [conflict], kind, False, False



def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)    # Child process will not immediately respond to interrupt signal. This leaves control to the main process.
if __name__ == "__main__":
    cpus = 5
    print('Start collecting, using processes:', cpus)
    # Record runtime
    import time
    start_time = time.time()

    with Pool(cpus, initializer=initializer) as pool:
        try:
            # Read collected conflicts
            with open(json_data_path, 'r') as f:
                data = json.load(f)
            conflicts = [Conflict(conflict['ours'], conflict['theirs'],
                                conflict['base'], conflict['resolve'], conflict['resolution_kind']) for conflict in data]
            random.shuffle(conflicts)

            return_tuples = list(tqdm(pool.imap(check_conflict_resolvable, enumerate(conflicts)), total=len(conflicts)))
            dataset = pd.concat([_tuple[0] for _tuple in return_tuples])
            # Write to file
            dataset.to_csv(output_file, index=False)

            # Show statistics
            kind_counter = Counter([ret[2] for ret in return_tuples])
            upper_bound_kind_counter = Counter([ret[2] for ret in return_tuples if ret[1] == []])
            line_too_many = sum([ret[3] for ret in return_tuples])
            script_num_too_many = sum([ret[4] for ret in return_tuples])
            able_to_generate_with_edit_scripts = sum([1 for ret in return_tuples if ret[1] == []])

            print(f'Too many scripts: {script_num_too_many}')
            print(f'Too many lines: {line_too_many}')
            print(f'Resolvable by edit script acceptance: {able_to_generate_with_edit_scripts}, ratio {able_to_generate_with_edit_scripts / len(conflicts) * 100}%')

            # Show data distribution
            print('All conflict resolution type distribution:')
            print(kind_counter)
            print('Upper bound data distribution:')
            print(upper_bound_kind_counter)

            # Show label distribution
            print('Label distribution:')
            print(dataset['accept'].value_counts())

            # Record runtime
            end_time = time.time()
            print(f'Runtime: {end_time - start_time} seconds')
        except KeyboardInterrupt:
            print('Manually stopped, exiting all processes')
            pool.terminate()
