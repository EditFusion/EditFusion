"""
This file converts the code from script.ipynb into a Python script for multi-process execution.
"""

import os
from pathlib import Path
from collections import defaultdict
import json
from util.conflict_util import Conflict, conflict2file
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import re

# NOTE: Absolute path may expose personal directory structure. Consider using a relative path or
# a configuration file for this value, especially for public repositories.
work_dir = Path('/root/projects/dataset_collect_analysis')
print(work_dir)

class ConflictChunk:
    def __init__(self, m_start, m_end, a_content, b_content,
                 o_content, r_content, label: str | None, chunk_idx):
        self.m_start = m_start
        self.m_end = m_end
        self.a_content: 'str' = a_content
        self.b_content: 'str' = b_content
        self.o_content: 'str' = o_content
        self.r_content: 'str' = r_content
        self.label = label
        self.chunk_idx = chunk_idx

    def to_dict(self):
        return {
            "m_start": self.m_start,
            "m_end": self.m_end,
            "a_content": self.a_content,
            "b_content": self.b_content,
            "o_content": self.o_content,
            "r_content": self.r_content,
            "label": self.label,
        }

    def getJSONstr(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)


class ConflictFile:
    def __init__(self, path, repo_url, file_a_content, file_b_content, file_o_content, file_r_content, file_m_content, commit_hash):
        self.path = path
        self.repo_url = repo_url
        self.file_a_content = file_a_content
        self.file_b_content = file_b_content
        self.file_o_content = file_o_content
        self.file_r_content = file_r_content
        self.file_m_content = file_m_content
        self.commit_hash = commit_hash
        self.conflict_chunks = []

    def add_conflict_chunk(self, conflict_chunk_obj):
        self.conflict_chunks.append(conflict_chunk_obj)

    def to_dict(self):
        return {
            "path": self.path,
            "repo_url": self.repo_url,
            "file_a_content": self.file_a_content,
            "file_b_content": self.file_b_content,
            "file_o_content": self.file_o_content,
            "file_r_content": self.file_r_content,
            "file_m_content": self.file_m_content,
            "conflict_chunks": [chunk.to_dict() for chunk in self.conflict_chunks],
        }

    def getJSONstr(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

class ConflictFileCollector:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    @staticmethod
    def sample(output_dir, n, random_seed=0, label=None):
        cnt = 0
        # Randomly sample n Conflict chunks of the given label type from all conflict files
        # Read all JSON files in output_dir
        jsons = list(ConflictFileCollector.getAllJsonsUnder(output_dir))
        print(f"Found {len(jsons)} JSON files in {output_dir}")
        # Read all Conflict chunks from JSON files
        for json_file in jsons:
            with open(json_file) as f:
                data = json.load(f)
            for conflict_file in data:
                for chunk in conflict_file['conflict_chunks']:
                    if label is None or chunk['label'] == label:
                        if cnt >= n:
                            return
                        cnt += 1
                        yield chunk

    def collect(self):
        '''
        Returns an iterator, each iteration returns a ConflictFile object
        '''
        raise NotImplementedError

    def collect_in_batches(self, batch_size=10000):
        batch = []
        for conflict_file in self.collect():
            batch.append(conflict_file)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def collect_and_save(self, output_dir, batch_size=10000):
        output_dir = Path(output_dir)  # Ensure output_dir is a Path object
        output_dir.mkdir(parents=True, exist_ok=True)  # Automatically create directory and parents
        for i, batch in enumerate(self.collect_in_batches(batch_size)):
            with open(output_dir / f"{i}.json", 'w') as f:
                print(f"Saving batch {i} to {output_dir / f'{i}.json'}")
                json.dump([json.loads(x.getJSONstr()) for x in batch], f)

    @staticmethod
    def preprocessContent(content: str):
        return '' if content.strip() == '' else re.sub(r'\s+', ' ', content.strip() + '\n')

    @staticmethod
    def getLabel(a, b, o, r):
        r_processed = ConflictFileCollector.preprocessContent(r)
        a_processed = ConflictFileCollector.preprocessContent(a)
        b_processed = ConflictFileCollector.preprocessContent(b)
        o_processed = ConflictFileCollector.preprocessContent(o)
        if a_processed == b_processed:
            return "same modification, formatting maybe different"
        if r_processed == a_processed:
            return "A"
        if r_processed == b_processed:
            return "B"
        if r_processed == o_processed:
            return "O"
        if r_processed == a_processed + b_processed:
            return "AB"
        if r_processed == b_processed + a_processed:
            return "BA"

        r_lines = set(r.split('\n'))
        a_lines = set(a.split('\n'))
        b_lines = set(b.split('\n'))
        o_lines = set(o.split('\n'))
        for rl in r_lines:
            if (rl not in a_lines) and (rl not in b_lines) and (rl not in o_lines) and not rl.isspace():
                return 'newline'
        return 'mixline'

    @staticmethod
    def getAllJsonsUnder(dirPath: str):
        for root, _, files in os.walk(dirPath):
            for file in files:
                if(file.endswith(".json")):
                    yield os.path.join(root, file)

    @staticmethod
    def list2str(l):
        if l == [] or l == ['']:
            return ''
        return '\n'.join(l) + '\n'


from util.edit_script import compute, SequenceDiff

class EditScriptLabel:
    def __init__(self, sd: SequenceDiff, _from: str, accept: bool):
        self.edit_script = sd
        self._from = _from
        self.accept = accept

def analyze_edit_script(dir2analyze):
    dataset_name = os.path.basename(dir2analyze)
    print(f'Statistics for {dataset_name}')
    accept_mark_cnt = defaultdict(int)
    es_cnt = defaultdict(int)
    cc_with_es_intersects = 0
    resolvable_cc_cnt = 0
    all_cc_cnt = 0
    too_many_lines_cnt = 0
    label_cnt = defaultdict(int)
    label_resolvable_cnt = defaultdict(int)

    def cc_check(chunk: ConflictChunk) -> None:
        '''
        Count conflicts that can be resolved by edit script, count accepted and rejected numbers
        Count the number of edit scripts, skip if too many
        For final comparison, I want to convert to token
        When generating edit scripts, remove blank line effects, indentation... remove?
        '''
        nonlocal accept_mark_cnt
        nonlocal es_cnt
        nonlocal resolvable_cc_cnt
        nonlocal too_many_lines_cnt
        nonlocal label_resolvable_cnt

        def es_gen_str2list(content: str) -> List[str]:
            '''
            Processing when generating edit scripts
            '''
            return [line.strip() for line in content.split('\n') if line.strip() != '']

        if len(chunk.a_content) > 5000 or len(chunk.b_content) > 5000 or len(chunk.o_content) > 5000 or len(chunk.r_content) > 5000:
            too_many_lines_cnt += 1
            return

        a_contents = es_gen_str2list(chunk.a_content)
        b_contents = es_gen_str2list(chunk.b_content)
        o_contents = es_gen_str2list(chunk.o_content)
        r_contents = es_gen_str2list(chunk.r_content)

        def compareInToken(a_ls: List[str], b_ls: List[str]) -> bool:
            '''
            Preprocessing for final comparison, ignore whitespace effects
            '''
            def toUnifiedStr(ls: List[str]) -> str:
                return '' if ls == [] or ls == [''] else re.sub(r'\s+', ' ', '\n'.join(ls).strip() + '\n')
            a_processed = toUnifiedStr(a_ls)
            b_processed = toUnifiedStr(b_ls)
            # print(a_processed)
            # print(b_processed)
            # print(a_processed == b_processed)
            # print('-' * 20)
            return a_processed == b_processed

        def bt(generated, i, last_end, all_edit_scripts: List[EditScriptLabel]) -> bool:
            '''
            Backtracking to generate all possible solutions, add to result set if matches resolution
            '''
            nonlocal cc_with_es_intersects
            # exit
            if i == len(all_edit_scripts):
                whole_generated = generated + o_contents[last_end:]
                # Filter out blank lines from whole_generated and resolution
                if compareInToken(whole_generated, r_contents):
                    # Conflicts that can be resolved using combined edit scripts
                    return True
                return False

            # Do not accept this script
            all_edit_scripts[i].accept = False
            if bt(generated, i + 1, last_end, all_edit_scripts):
                return True

            # If the current script's starting position is smaller than last_end,
            # it means this script conflicts with the previous one.
            # Cannot accept this script, skip directly.
            if all_edit_scripts[i].edit_script.seq1Range.start < last_end:
                cc_with_es_intersects += 1
                return False # Can resolve pseudo-conflicts because it's a less than sign

            # Accept this script
            start = all_edit_scripts[i].edit_script.seq2Range.start
            end = all_edit_scripts[i].edit_script.seq2Range.end
            if all_edit_scripts[i]._from == 'ours':
                curr_content = a_contents[start:end]
            else:
                curr_content = b_contents[start:end]
            all_edit_scripts[i].accept = True
            if bt(generated
                    + o_contents[last_end:all_edit_scripts[i].edit_script.seq1Range.start]
                    + curr_content,
                    i + 1,
                    all_edit_scripts[i].edit_script.seq1Range.end,
                    all_edit_scripts
                ):
                return True

            # If there is a next script, and both correspond to the same base position
            if (
                i + 1 < len(all_edit_scripts) and
                all_edit_scripts[i].edit_script.seq1Range == all_edit_scripts[i + 1].edit_script.seq1Range
            ):
                start = all_edit_scripts[i + 1].edit_script.seq2Range.start
                end = all_edit_scripts[i + 1].edit_script.seq2Range.end
                if all_edit_scripts[i + 1]._from == 'ours':
                    next_content = a_contents[start:end]
                else:
                    next_content = b_contents[start:end]

                # Case where base length is 0, only need to add another concat (seq1Range length 0 means insertion at the same position by both sides)
                all_edit_scripts[i + 1].accept = True
                if bt(generated
                        + o_contents[last_end:all_edit_scripts[i].edit_script.seq1Range.start]
                        + next_content
                        + curr_content,
                        i + 2,
                        all_edit_scripts[i].edit_script.seq1Range.end,
                        all_edit_scripts
                    ):
                    return True
                # Case where base length is not 0, need to consider two concats
                if len(all_edit_scripts[i].edit_script.seq1Range) > 0:
                    all_edit_scripts[i + 1].accept = True
                    if bt(generated
                            + o_contents[last_end:all_edit_scripts[i].edit_script.seq1Range.start]
                            + curr_content
                            + next_content,
                            i + 2,
                            all_edit_scripts[i].edit_script.seq1Range.end,
                            all_edit_scripts
                        ):
                        return True
            return False


        # Start collecting dataset
        kind = chunk.label

        # Skip if it's a 'newline' conflict
        if kind == 'newline':
            return

        # Skip if the number of lines is too large
        if any([len(content) > 1000 for content in [a_contents, b_contents, o_contents, r_contents]]):
            too_many_lines_cnt += 1
            return
        from_ours = compute(o_contents, a_contents)
        from_theirs = compute(o_contents, b_contents)
        # Add '_from' tag
        from_ours = [EditScriptLabel(sd, 'ours', False) for sd in from_ours]
        from_theirs = [EditScriptLabel(sd, 'theirs', False) for sd in from_theirs]
        all_edit_scripts = from_ours + from_theirs
        es_cnt[len(all_edit_scripts)] += 1


        # Limit the number of scripts to avoid excessive computation
        if len(all_edit_scripts) > 20:
            return

        all_edit_scripts.sort(key=lambda editScriptLabel: editScriptLabel.edit_script.seq1Range)

        if bt([], 0, 0, all_edit_scripts):  # This conflict can be resolved
            resolvable_cc_cnt += 1
            label_resolvable_cnt[kind] += 1
            # Count accept_mark
            for i, es in enumerate(all_edit_scripts):
                accept_mark_cnt[es.accept] += 1


    # Start counting dataset results
    jsonPaths = [path for path in ConflictFileCollector.getAllJsonsUnder(dir2analyze)]
    if len(jsonPaths) == 0:
        raise FileNotFoundError("No metadata json files found in the dataset path")
    for jsonPath in tqdm(jsonPaths, desc="Processing files", position=0, leave=True, dynamic_ncols=True):
        # jsonData
        try:
            with open(jsonPath, 'r') as f:
                cfs = json.load(f)
        except Exception as e:
            print(f"Error reading {jsonPath}: {e} (type: {type(e).__name__})")
            import traceback
            traceback.print_exc()
            continue # Continue to the next file if an error occurs

        for cf in tqdm(cfs, desc=f"Process items", position=1, leave=False, dynamic_ncols=True):
            for cc in cf['conflict_chunks']:
                all_cc_cnt += 1
                label_cnt[cc['label']] += 1
                cc_obj = ConflictChunk(cc['m_start'], cc['m_end'], cc['a_content'], cc['b_content'], cc['o_content'], cc['r_content'], cc['label'], cc['chunk_idx'])
                cc_check(cc_obj)

    def print_res_to_file(file=os.sys.stdout):
        print(f'Statistics for {dataset_name}:', file=file)
        print(f'Total {all_cc_cnt} conflict chunks, {resolvable_cc_cnt} can be resolved by edit script, ratio {resolvable_cc_cnt / all_cc_cnt * 100:.2f}%', file=file)
        print(f'{cc_with_es_intersects} conflict chunks have intersecting edit scripts', file=file)
        print(f'{too_many_lines_cnt} conflict chunks have too many lines and cannot be processed', file=file)
        print(f'Edit script count distribution: {es_cnt}', file=file)
        print(f'Accept mark distribution: {accept_mark_cnt}', file=file)
        print(f'Type distribution: {label_cnt}', file=file)
        print(f'Resolvable type distribution: {label_resolvable_cnt}', file=file)
        for k, v in label_cnt.items():
            print(f'{k}: {v}, resolvable: {label_resolvable_cnt[k]}, ratio: {label_resolvable_cnt[k] / v * 100:.2f}%', file=file)

    # Create a new directory
    os.makedirs(work_dir / 'data_collect_analysis' / 'bt_log', exist_ok=True)
    print_res_to_file(file=open(work_dir / 'data_collect_analysis' / 'bt_log' / f'{dataset_name}.log', 'w'))


# dir2analyze = work_dir / "data_collect_analysis" / "output" / "100+stars_4GB-_multidev_org_lang"
# dir2analyze = work_dir / "data_collect_analysis" / "output" / "2000repos"
dir2analyze = work_dir / "data_collect_analysis" / "output" / "top50"
# dir2analyze = work_dir / "data_collect_analysis" / "output" / "mergebert_ts"
# dir2analyze = work_dir / "data_collect_analysis" / "output" / "mergebert_all_lang"
analyze_edit_script(dir2analyze)