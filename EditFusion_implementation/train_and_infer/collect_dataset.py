from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple
import json
from .utils.tokenizer_util import (
    encode_text_to_tokens,
    encode_tokens_to_ids,
    pad_token_id,
    bos_token_id,
    eos_token_id,
)
from .utils.conflict_utils import Conflict
from .utils.es_generator import compute, SequenceDiff, get_edit_sequence
from tqdm import tqdm
from collections import Counter
import pandas as pd
import os
import signal
import re

script_path = Path(os.path.dirname(os.path.abspath(__file__)))
json_data_dir = Path(
    "/home/foril/projects/EditFusion/EditFusion_implementation/train_and_infer/data/gathered_data/mergebert_all_lang"
)
output_dir = script_path / "data" / "processed_data" / "codebert_mergebert_all_lang"
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

MAX_TOKEN_LEN = 64


class ConflictChunk:
    def __init__(
        self,
        m_start,
        m_end,
        a_content,
        b_content,
        o_content,
        r_content,
        label: str | None,
        chunk_idx,
    ):
        self.m_start = m_start
        self.m_end = m_end
        self.a_content: "str" = a_content
        self.b_content: "str" = b_content
        self.o_content: "str" = o_content
        self.r_content: "str" = r_content
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
    def __init__(
        self,
        path,
        repo_url,
        file_a_content,
        file_b_content,
        file_o_content,
        file_r_content,
        file_m_content,
        commit_hash,
    ):
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
        """Randomly sample n conflict chunks of a specific label type from all conflict files."""
        cnt = 0
        jsons = list(ConflictFileCollector.getAllJsonsUnder(output_dir))
        print(f"Found {len(jsons)} JSON files in {output_dir}")
        for json_file in jsons:
            with open(json_file) as f:
                data = json.load(f)
            for conflict_file in data:
                for chunk in conflict_file["conflict_chunks"]:
                    if label == None or chunk["label"] == label:
                        if cnt >= n:
                            return
                        cnt += 1
                        yield chunk

    def collect(self):
        """
        Returns an iterator that yields a ConflictFile object at each iteration.
        """
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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(self.collect_in_batches(batch_size)):
            with open(output_dir / f"{i}.json", "w") as f:
                print(f"Saving batch {i} to {output_dir / f'{i}.json'}")
                json.dump([json.loads(x.getJSONstr()) for x in batch], f)

    @staticmethod
    def preprocessContent(content: str):
        return (
            "" if content.strip() == "" else re.sub(r"\s+", " ", content.strip() + "\n")
        )

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

        r_lines = set(r.split("\n"))
        a_lines = set(a.split("\n"))
        b_lines = set(b.split("\n"))
        o_lines = set(o.split("\n"))
        for rl in r_lines:
            if (
                (rl not in a_lines)
                and (rl not in b_lines)
                and (rl not in o_lines)
                and not rl.isspace()
            ):
                return "newline"
        return "mixline"

    @staticmethod
    def getAllJsonsUnder(dirPath: str):
        for root, _, files in os.walk(dirPath):
            for file in files:
                if file.endswith(".json"):
                    yield os.path.join(root, file)

    @staticmethod
    def list2str(l):
        if l == [] or l == [""]:
            return ""
        return "\n".join(l) + "\n"


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


class EditScriptLabel:
    def __init__(self, block_id: str, sd: SequenceDiff, _from: str, accept: bool):
        self.edit_script = sd
        self._from = _from
        self.accept = accept
        self.block_id = block_id


def es_gen_str2list(content: str) -> List[str]:
    """
    Process string for edit script generation.
    """
    return [line.strip() for line in content.split("\n") if line.strip() != ""]



def compareInToken(a_ls: List[str], b_ls: List[str]) -> bool:
    """
    Pre-process for final comparison, ignoring whitespace.
    """

    def toUnifiedStr(ls: List[str]) -> str:
        tmp = re.sub(r"\s+", " ", "\n".join(ls).strip() + "\n")
        return "" if ls == [] or ls == [""] else tmp

    a_processed = toUnifiedStr(a_ls)
    b_processed = toUnifiedStr(b_ls)
    return a_processed == b_processed


def check_conflict_resolvable(cfJSON: dict) -> Tuple[pd.DataFrame, List[dict], int, int]:
    """
    Input a JSON dict in the shape of ConflictFile,
    collect conflicts that can be resolved with edit scripts, and build a training dataset.
    Returns:
        resolvable_dataset: A DataFrame of all edit scripts for conflicts that can be resolved with edit scripts.
        conflicts_unable_to_resolve_with_edit_scripts: Conflicts that cannot be resolved with edit scripts, returns an empty list if resolvable (elements are ConflictChunk JSON dicts).
        line_too_many: Count of samples with too many lines.
        script_num_too_many: Count of samples with too many edit scripts.
    """

    df_columns = [
        "block_id",
        "conflict_base",
        "conflict_ours",
        "conflict_theirs",
        "conflict_resolution",
        "origin_start",
        "origin_end",
        "modified_start",
        "modified_end",
        "origin_content",
        "modified_content",
        "from",
        "accept",
        "resolution_kind",
        "origin_tokens",
        "modified_tokens",
        "edit_seq_tokens",
        "origin_tokens_truncated",
        "modified_tokens_truncated",
        "edit_seq_tokens_truncated",
        "origin_ids_truncated",
        "modified_ids_truncated",
        "edit_seq_ids_truncated",
        "origin_processed_ids",
        "modified_processed_ids",
        "edit_seq_processed_ids",
    ]
    resolvable_dataset = pd.DataFrame(columns=df_columns)
    conflicts_unable_to_resolve_with_edit_scripts = []
    line_too_many = 0
    script_num_too_many = 0

    def bt(generated, i, last_end, accept_mark: List[bool]) -> bool:
        """
        Backtracking to generate all possible solutions, if it is the same as the resolution, add it to the result set.
        """
        if i == len(all_edit_scripts):
            whole_generated = generated + o_content_lines[last_end:]
            if compareInToken(whole_generated, r_content_lines):
                for i, edit_script_label in enumerate(accept_mark):
                    all_edit_scripts[i].accept = edit_script_label
                return True
            return False

        accept_mark[i] = False
        if bt(generated, i + 1, last_end, accept_mark):
            return True

        if all_edit_scripts[i].edit_script.seq1Range.start < last_end:
            return False

        start = all_edit_scripts[i].edit_script.seq2Range.start
        end = all_edit_scripts[i].edit_script.seq2Range.end
        if all_edit_scripts[i]._from == "ours":
            curr_content = a_content_lines[start:end]
        else:
            curr_content = b_content_lines[start:end]
        accept_mark[i] = True
        if bt(
            generated
            + o_content_lines[
                last_end : all_edit_scripts[i].edit_script.seq1Range.start
            ]
            + curr_content,
            i + 1,
            all_edit_scripts[i].edit_script.seq1Range.end,
            accept_mark,
        ):
            return True

        if (
            i + 1 < len(all_edit_scripts)
            and all_edit_scripts[i].edit_script.seq1Range
            == all_edit_scripts[i + 1].edit_script.seq1Range
        ):
            start = all_edit_scripts[i + 1].edit_script.seq2Range.start
            end = all_edit_scripts[i + 1].edit_script.seq2Range.end
            if all_edit_scripts[i + 1]._from == "ours":
                next_content = a_content_lines[start:end]
            else:
                next_content = b_content_lines[start:end]

            accept_mark[i + 1] = True
            if bt(
                generated
                + o_content_lines[
                    last_end : all_edit_scripts[i].edit_script.seq1Range.start
                ]
                + next_content
                + curr_content,
                i + 2,
                all_edit_scripts[i].edit_script.seq1Range.end,
                accept_mark,
            ):
                return True
            if len(all_edit_scripts[i].edit_script.seq1Range) > 0:
                accept_mark[i + 1] = True
                if bt(
                    generated
                    + o_content_lines[
                        last_end : all_edit_scripts[i].edit_script.seq1Range.start
                    ]
                    + curr_content
                    + next_content,
                    i + 2,
                    all_edit_scripts[i].edit_script.seq1Range.end,
                    accept_mark,
                ):
                    return True

    for i, chunk in enumerate(cfJSON["conflict_chunks"]):
        if 'commit_hash' in cfJSON:
            block_id = f"{cfJSON['repo_url']}_{cfJSON['commit_hash']}_{cfJSON['path']}_{id(cfJSON)}_{id(chunk)}"
        else:
            block_id = f"{cfJSON['repo_url']}_{cfJSON['commitHash']}_{cfJSON['path']}_{id(cfJSON)}_{id(chunk)}"
        kind = chunk["label"]
        if kind == "newline":
            conflicts_unable_to_resolve_with_edit_scripts.append(chunk)
            continue
        if kind == "same modification, formatting maybe different":
            continue

        a_content_lines = es_gen_str2list(chunk["a_content"])
        b_content_lines = es_gen_str2list(chunk["b_content"])
        o_content_lines = es_gen_str2list(chunk["o_content"])
        r_content_lines = es_gen_str2list(chunk["r_content"])

        # 如果行数过大，直接跳过
        if any(
            [
                len(content) > 200
                for content in [a_content_lines, b_content_lines, o_content_lines]
            ]
        ):
            line_too_many += 1
            continue

        from_ours = compute(
            o_content_lines, a_content_lines
        )
        from_theirs = compute(o_content_lines, b_content_lines)
        from_ours = [EditScriptLabel(block_id, sd, "ours", False) for sd in from_ours]
        from_theirs = [
            EditScriptLabel(block_id, sd, "theirs", False) for sd in from_theirs
        ]

        all_edit_scripts = from_ours + from_theirs
        # 限制脚本数量，避免计算量过大
        if len(all_edit_scripts) > 10:
            script_num_too_many += 1
            continue
        all_edit_scripts.sort(key=lambda x: x.edit_script.seq1Range)

        if bt([], 0, 0, [False] * len(all_edit_scripts)):  # 这个冲突能解决
            # 加入数据集
            # 构造一个完全一点的数据集，包含所有冲突块内容，以及本编辑脚本起止位置，以及来源方
            all_es_for_this_conflict = []
            for es in all_edit_scripts:
                origin_start = es.edit_script.seq1Range.start
                origin_end = es.edit_script.seq1Range.end
                modified_start = es.edit_script.seq2Range.start
                modified_end = es.edit_script.seq2Range.end

                def process_to_str(content: List[str]) -> str:
                    if content in [[""], []]:
                        return ""
                    return "\n".join(content) + "\n"

                origin_content_str = process_to_str(
                    o_content_lines[origin_start:origin_end]
                )
                modified_tmp = (
                    a_content_lines if es._from == "ours" else b_content_lines
                )
                modified_content_str = process_to_str(
                    modified_tmp[modified_start:modified_end]
                )

                origin_tokens = encode_text_to_tokens(origin_content_str)[
                    : 3 * MAX_TOKEN_LEN
                ]
                modified_tokens = encode_text_to_tokens(modified_content_str)[
                    : 3 * MAX_TOKEN_LEN
                ]
                edit_seq_tokens, origin_padded_tokens, modified_padded_tokens = (
                    get_edit_sequence(origin_tokens, modified_tokens)
                )
                edit_seq_tokens_truncated = edit_seq_tokens[: MAX_TOKEN_LEN - 2]
                origin_tokens_truncated = origin_padded_tokens[: MAX_TOKEN_LEN - 2]
                modified_tokens_truncated = modified_padded_tokens[: MAX_TOKEN_LEN - 2]

                # 这里把 padding 之前的 id 也存在数据集里
                edit_seq_ids_truncated = encode_tokens_to_ids(edit_seq_tokens_truncated)
                origin_ids_truncated = encode_tokens_to_ids(origin_tokens_truncated)
                modified_ids_truncated = encode_tokens_to_ids(modified_tokens_truncated)

                # truncate and pad ids
                edit_seq_processed_ids = process_to_legal_ids(
                    encode_tokens_to_ids(edit_seq_tokens)
                )
                origin_processed_ids = process_to_legal_ids(
                    encode_tokens_to_ids(origin_padded_tokens)
                )
                modified_processed_ids = process_to_legal_ids(
                    encode_tokens_to_ids(modified_padded_tokens)
                )

                all_es_for_this_conflict.append(
                    [
                        es.block_id,
                        o_content_lines,
                        a_content_lines,
                        b_content_lines,
                        r_content_lines,
                        origin_start,
                        origin_end,
                        modified_start,
                        modified_end,
                        origin_content_str,
                        modified_content_str,
                        es._from,
                        es.accept,
                        kind,
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
                    ]
                )
            df2add = pd.DataFrame(all_es_for_this_conflict, columns=df_columns)
            resolvable_dataset = pd.concat(
                [resolvable_dataset, df2add], ignore_index=True
            )
        else:
            conflicts_unable_to_resolve_with_edit_scripts.append(chunk)

    return (
        resolvable_dataset,
        conflicts_unable_to_resolve_with_edit_scripts,
        line_too_many,
        script_num_too_many,
    )


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == "__main__":
    cpus = 12
    print(f"Starting collection with {cpus} processes...")
    import time

    start_time = time.time()

    jsons = list(json_data_dir.glob("*.json"))
    for json_file in tqdm(jsons, position=0, dynamic_ncols=True, leave=False):
        file_basic_name = json_file.stem
        conflictFileJSONs: List[dict] = json.load(open(json_file))

        with Pool(cpus, initializer=initializer) as pool:
            try:
                return_tuples = list(
                    tqdm(
                        pool.imap(check_conflict_resolvable, conflictFileJSONs),
                        dynamic_ncols=True,
                        position=1,
                        total=len(conflictFileJSONs),
                    )
                )
                dataset = pd.concat([_tuple[0] for _tuple in return_tuples])
                print(f"{json_file.stem} processed.")
                print(f"Processed {len(conflictFileJSONs)} conflict files.")
                all_ccs = sum(
                    [len(cfJSON["conflict_chunks"]) for cfJSON in conflictFileJSONs]
                )
                unresolved_ccs = sum([len(_tuple[1]) for _tuple in return_tuples])
                print(
                    f"Total {all_ccs} conflict chunks, {unresolved_ccs} unresolved. "
                    f"({unresolved_ccs / all_ccs * 100:.2f}%)"
                )
                dataset.to_csv(output_dir / (file_basic_name + ".csv"), index=False)
                end_time = time.time()
                print(f"Elapsed time: {end_time - start_time}s")
            except KeyboardInterrupt:
                print("Manually stopped, exiting all processes.")
                pool.terminate()
