# 用于收集数据集
# 读物 json_data_dir 下的所有 json 文件，每个文件中包含多个 ConflictFile 对象，每个 ConflictFile 对象包含多个 ConflictChunk 对象

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

# 读取收集到的冲突
script_path = Path(os.path.dirname(os.path.abspath(__file__)))
# json_data_dir = script_path / 'data' / 'gathered_data' / '100+sample'
json_data_dir = Path(
    # "/root/projects/conflictManager/edit_script_resolver/train_and_infer/data/gathered_data/zero_shot_conflict_files"
    # "/root/projects/conflictManager/edit_script_resolver/train_and_infer/data/gathered_data/codebert_conflict_files"
    
    "/home/foril/projects/EditFusion/EditFusion_implementation/train_and_infer/data/gathered_data/mergebert_all_lang"
)
output_dir = script_path / "data" / "processed_data" / "codebert_mergebert_all_lang"
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

# 获取代码变更嵌入时的最大 token 长度（用于填充和截断）
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
        cnt = 0
        # 从所有冲突文件中随机抽取 n 个 label 类型的 Conflict chunk
        # 读取 output_dir 中的所有 JSON 文件
        jsons = list(ConflictFileCollector.getAllJsonsUnder(output_dir))
        print(f"Found {len(jsons)} JSON files in {output_dir}")
        # 读取所有 JSON 文件中的 Conflict chunk
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
        返回一个迭代器，每次迭代返回一个ConflictFile对象
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
        output_dir = Path(output_dir)  # 确保 output_dir 是 Path 对象
        output_dir.mkdir(parents=True, exist_ok=True)  # 自动创建目录及其父目录
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
    if len(tokens) > MAX_TOKEN_LEN - 2:  # 留给 <s> 和 </s>
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
    生成编辑脚本时的处理
    """
    return [line.strip() for line in content.split("\n") if line.strip() != ""]


def compareInToken(a_ls: List[str], b_ls: List[str]) -> bool:
    """
    最后比较的预处理，忽略空白符的影响
    """

    def toUnifiedStr(ls: List[str]) -> str:
        tmp = re.sub(r"\s+", " ", "\n".join(ls).strip() + "\n")
        return "" if ls == [] or ls == [""] else tmp

    a_processed = toUnifiedStr(a_ls)
    b_processed = toUnifiedStr(b_ls)
    return a_processed == b_processed


def check_conflict_resolvable(cfJSON: dict) -> Tuple[pd.DataFrame, List[dict], int, int]:
    """
    输入 ConflictFile 形状的 JSON dict，
    收集可以用编辑脚本解决的冲突，构建训练用数据集
    Returns:
        resolvable_dataset: 可以用编辑脚本解决的冲突的所有编辑脚本构成的数据集（DataFrame）
        conflicts_unable_to_resolve_with_edit_scripts: 冲突不可以用编辑脚本解决，可解决返回空列表（在这里元素是 ConflictChunk JSON dict）
        line_too_many: 行数过多的样本计数
        script_num_too_many: 编辑脚本过多的样本计数
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
        回溯法生成所有可能的解决方案，如果和 resolution 相同则加入结果集
        """
        if i == len(all_edit_scripts):
            whole_generated = generated + o_content_lines[last_end:]
            # 比较时过滤 whole_generated 和 resolution 中的空白符
            if compareInToken(whole_generated, r_content_lines):
                # 可以使用组合 ES 的方式解决的冲突
                for i, edit_script_label in enumerate(accept_mark):
                    all_edit_scripts[i].accept = edit_script_label
                return True
            return False

        # 不接受这个脚本
        accept_mark[i] = False
        if bt(generated, i + 1, last_end, accept_mark):
            return True

        # 如果当前脚本的起始位置比 last_end 还小，说明这个脚本和上一个脚本有冲突（想接受都接受不了，这条路径不能解决）
        # 不能接受这个脚本，直接跳过
        if all_edit_scripts[i].edit_script.seq1Range.start < last_end:
            return False  # 因为是小于号，所以可以解决伪冲突

        # 接受这个脚本
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

        # 有下一个脚本，且两者对应 base 的位置相同
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

            # base 长度为 0 的情况，只需要加入另一种 concat（seq1Range 的长度为 0，代表双方在同一位置的插入），ABconcat 已经可以解决
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
            # base 长度不为 0 的情况，需要考虑两种 concat
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
        # mergebert 数据集是 commit_hash
        if 'commit_hash' in cfJSON:
            block_id = f"{cfJSON['repo_url']}_{cfJSON['commit_hash']}_{cfJSON['path']}_{id(cfJSON)}_{id(chunk)}"  # 就是 chunk_id，名字没起好，保证全局唯一即可，无意义
        else:
            block_id = f"{cfJSON['repo_url']}_{cfJSON['commitHash']}_{cfJSON['path']}_{id(cfJSON)}_{id(chunk)}"  # 就是 chunk_id，名字没起好，保证全局唯一即可，无意义
        kind = chunk["label"]
        if kind == "newline":
            conflicts_unable_to_resolve_with_edit_scripts.append(chunk)
            continue
        if kind == "same modification, formatting maybe different":
            # 两边修改一样，但是格式不一样，这种情况不处理，感觉对训练没有帮助
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
            # conflicts_unable_to_resolve_with_edit_scripts.append(chunk)
            continue

        from_ours = compute(
            o_content_lines, a_content_lines
        )  # 生成时使用的 List 要和 bt 中切片时的 List 一致
        from_theirs = compute(o_content_lines, b_content_lines)
        # 加入标识，用于区分是来自ours还是theirs
        from_ours = [EditScriptLabel(block_id, sd, "ours", False) for sd in from_ours]
        from_theirs = [
            EditScriptLabel(block_id, sd, "theirs", False) for sd in from_theirs
        ]

        all_edit_scripts = from_ours + from_theirs
        # 限制脚本数量，避免计算量过大
        if len(all_edit_scripts) > 10:
            script_num_too_many += 1
            # conflicts_unable_to_resolve_with_edit_scripts.append(chunk)
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
                # ! 这里使用去掉缩进和空行的文本，因为编辑脚本生成时去掉了缩进和空行

                # 先 truncate tokens
                origin_tokens = encode_text_to_tokens(origin_content_str)[
                    : 3 * MAX_TOKEN_LEN
                ]  # 这里如果 tokens 太多，下面计算编辑序列会很慢
                modified_tokens = encode_text_to_tokens(modified_content_str)[
                    : 3 * MAX_TOKEN_LEN
                ]  # 取 3 倍是为了防止对齐时找不到对应的 token，保留尽可能多信息
                # 计算编辑序列前，先不要填充
                edit_seq_tokens, origin_padded_tokens, modified_padded_tokens = (
                    get_edit_sequence(origin_tokens, modified_tokens)
                )
                # ToDo 这里把 padding 之前的 id 也存在数据集里
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
                        o_content_lines,  # 整个冲突块的 base: List[str]
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
            # 转换为 DataFrame
            df2add = pd.DataFrame(all_es_for_this_conflict, columns=df_columns)
            resolvable_dataset = pd.concat(
                [resolvable_dataset, df2add], ignore_index=True
            )
        else:
            # 不能解决的冲突
            conflicts_unable_to_resolve_with_edit_scripts.append(chunk)

    return (
        resolvable_dataset,
        conflicts_unable_to_resolve_with_edit_scripts,
        line_too_many,
        script_num_too_many,
    )


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(
        signal.SIGINT, signal.SIG_IGN
    )  # 子进程不会立即响应中断信号。这样做的目的是将中断信号的控制权留给主进程。


if __name__ == "__main__":
    cpus = 12
    print("开始收集，使用进程数：", cpus)
    # 记录运行时间
    import time

    start_time = time.time()

    # 获取 json_data_dir 下的所有 json 文件
    jsons = list(json_data_dir.glob("*.json"))
    for json_file in tqdm(jsons, position=0, dynamic_ncols=True, leave=False):
        # 文件名
        file_basic_name = json_file.stem
        conflictFileJSONs: List[dict] = json.load(open(json_file))

        # 多进程处理
        with Pool(cpus, initializer=initializer) as pool:
            try:
                # 这一个 JSON 文件中所有 ConflictFile 的处理结果
                return_tuples = list(
                    tqdm(
                        pool.imap(check_conflict_resolvable, conflictFileJSONs),
                        dynamic_ncols=True,
                        position=1,
                        total=len(conflictFileJSONs),
                    )
                )
                # return_tuples = list(tqdm(pool.imap(check_conflict_resolvable, conflictFileJSONs[:100]), total=100))
                dataset = pd.concat([_tuple[0] for _tuple in return_tuples])
                # 写入文件
                print(json_file.stem + "处理完成")
                print("共处理了", len(conflictFileJSONs), "个冲突文件")
                all_ccs = sum(
                    [len(cfJSON["conflict_chunks"]) for cfJSON in conflictFileJSONs]
                )
                unresolved_ccs = sum([len(_tuple[1]) for _tuple in return_tuples])
                print(
                    "共",
                    all_ccs,
                    "个冲突块，其中不能解决的有",
                    sum([len(_tuple[1]) for _tuple in return_tuples]),
                    "个（这个数字不精确，具体看 script.ipynb）",
                    "占比 %.2f%%"
                    % (
                        sum([len(_tuple[1]) for _tuple in return_tuples])
                        / all_ccs
                        * 100
                    ),
                )
                dataset.to_csv(output_dir / (file_basic_name + ".csv"), index=False)
                # 记录运行时间
                end_time = time.time()
                print(f"运行时间：{end_time - start_time}秒")
            except KeyboardInterrupt:
                print("manually stop, exiting all processes")
                pool.terminate()

        # # 单进程处理
        # dataset = pd.DataFrame()
        # cc_cnt = 0
        # unresolvable_cnt = 0
        # for i, cfJSON in tqdm(enumerate(conflictFileJSONs), total=len(conflictFileJSONs), position=1, dynamic_ncols=True, leave=False):
        #     cc_cnt += len(cfJSON['conflict_chunks'])
        #     df2add, unresolvables, line_too_many, script_num_too_many = check_conflict_resolvable(cfJSON)
        #     unresolvable_cnt += len(unresolvables)
        #     dataset = pd.concat([dataset, df2add], ignore_index=True)
        # print(f'{json_file.stem}.json 处理完成')
        # print('共处理了', i+1, '个冲突文件')
        # print('共', cc_cnt, '个冲突块，其中不能解决的有', unresolvable_cnt, '个（这个数字不精确，具体看 script.ipynb）', '占比 %.2f%%' % (unresolvable_cnt / cc_cnt * 100))
        # print(f'获得 {len(dataset)} 个编辑脚本')
        # dataset.to_csv(output_dir / 'tmp.csv' , index=False)
