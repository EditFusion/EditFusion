import os
import re
import shutil
from pathlib import Path
from git import Repo, Git
from transformers import PreTrainedTokenizer
from typing import Callable, List
import numpy as np

def normalize_code(text: str) -> str:
    """Removes indents, empty lines, and duplicate whitespace."""
    lines = text.split('\n')
    processed_lines = [line.strip() for line in lines if line.strip()]
    return re.sub(r'\s+', ' ', " ".join(processed_lines))

def compare_token_lists(list1, list2):
    """Compare two lists of tokens, ignoring whitespace differences."""
    str1 = "".join(list1).replace(" ", "")
    str2 = "".join(list2).replace(" ", "")
    return str1 == str2

def apply_pattern(pattern, a, b, o):
    if pattern == "select_a": return a
    if pattern == "select_b": return b
    if pattern == "select_o": return o
    if pattern == "concat_ab": return a + b
    if pattern == "concat_ba": return b + a
    o_set = set(o)
    if pattern == "select_a_del_o": return [token for token in a if token not in o_set]
    if pattern == "select_b_del_o": return [token for token in b if token not in o_set]
    if pattern == "concat_ab_del_o": return [token for token in a + b if token not in o_set]
    if pattern == "concat_ba_del_o": return [token for token in b + a if token not in o_set]
    return None

def align_and_get_edit_sequence(before: List[str], after: List[str]) -> List[List[str]]:
    return get_edit_sequence(before, after)

TEMP_REPO_BASE_PATH = Path("/tmp/mergebert_repo_temp")

edit_seq_tokens = {
    "eql": "=", "add": "+", "del": "-", "rep": "↔", "padding": "[PAD]"
}

class OffsetRange:
    def __init__(self, start: int, end: int):
        self.start, self.end = start, end
    def __lt__(self, other): return (self.start, self.end) < (other.start, other.end)
    def __eq__(self, other: object) -> bool:
        return isinstance(other, OffsetRange) and self.start == other.start and self.end == other.end
    def __len__(self): return self.end - self.start

class SequenceDiff:
    def __init__(self, seq1Range: OffsetRange, seq2Range: OffsetRange):
        self.seq1Range, self.seq2Range = seq1Range, seq2Range
    def __repr__(self): return f"SequenceDiff({self.seq1Range.start}, {self.seq1Range.end}, {self.seq2Range.start}, {self.seq2Range.end})"

def compute(sequence1: List[str], sequence2: List[str], equalityScore: Callable[[int, int], float] = lambda x, y: 1 if x == y else 0) -> List[SequenceDiff]:
    if not sequence1 or not sequence2: return [SequenceDiff(OffsetRange(0, len(sequence1)), OffsetRange(0, len(sequence2)))]
    lcsLengths, directions, lengths = np.zeros((len(sequence1), len(sequence2))), np.zeros((len(sequence1), len(sequence2))), np.zeros((len(sequence1), len(sequence2)))
    for s1 in range(len(sequence1)):
        for s2 in range(len(sequence2)):
            h_len, v_len = (lcsLengths[s1 - 1, s2] if s1 > 0 else 0), (lcsLengths[s1, s2 - 1] if s2 > 0 else 0)
            ext_score = -1
            if sequence1[s1] == sequence2[s2]:
                ext_score = (lcsLengths[s1 - 1, s2 - 1] if s1 > 0 and s2 > 0 else 0) + equalityScore(sequence1[s1], sequence2[s2])
                if s1 > 0 and s2 > 0 and directions[s1 - 1, s2 - 1] == 3: ext_score += lengths[s1 - 1, s2 - 1]
            newValue = max(h_len, v_len, ext_score)
            if newValue == ext_score: lengths[s1, s2], directions[s1, s2] = (lengths[s1 - 1, s2 - 1] if s1 > 0 and s2 > 0 else 0) + 1, 3
            elif newValue == h_len: lengths[s1, s2], directions[s1, s2] = 0, 1
            else: lengths[s1, s2], directions[s1, s2] = 0, 2
            lcsLengths[s1, s2] = newValue
    result, s1, s2, last_s1, last_s2 = [], len(sequence1) - 1, len(sequence2) - 1, len(sequence1), len(sequence2)
    while s1 >= 0 and s2 >= 0:
        if directions[s1, s2] == 3:
            if s1 + 1 != last_s1 or s2 + 1 != last_s2: result.append(SequenceDiff(OffsetRange(s1 + 1, last_s1), OffsetRange(s2 + 1, last_s2)))
            last_s1, last_s2 = s1, s2
            s1, s2 = s1 - 1, s2 - 1
        elif directions[s1, s2] == 1: s1 -= 1
        else: s2 -= 1
    if -1 + 1 != last_s1 or -1 + 1 != last_s2: result.append(SequenceDiff(OffsetRange(0, last_s1), OffsetRange(0, last_s2)))
    result.reverse()
    return result

def get_edit_sequence(before: List[str], after: List[str]) -> List[List[str]]:
    ess, seq, before_padded, after_padded, b_trav, a_trav = compute(before, after), [], [], [], 0, 0
    for es in ess:
        b_start, b_end, a_start, a_end, b_len, a_len = es.seq1Range.start, es.seq1Range.end, es.seq2Range.start, es.seq2Range.end, len(es.seq1Range), len(es.seq2Range)
        seq.extend([edit_seq_tokens["eql"]] * (b_start - b_trav)); before_padded.extend(before[b_trav:b_start]); after_padded.extend(after[a_trav:a_start])
        if b_len == 0: seq.extend([edit_seq_tokens["add"]] * a_len); before_padded.extend([edit_seq_tokens["padding"]] * a_len); after_padded.extend(after[a_start:a_end])
        elif a_len == 0: seq.extend([edit_seq_tokens["del"]] * b_len); before_padded.extend(before[b_start:b_end]); after_padded.extend([edit_seq_tokens["padding"]] * b_len)
        else:
            if b_len > a_len: seq.extend([edit_seq_tokens["rep"]] * a_len + [edit_seq_tokens["del"]] * (b_len - a_len)); before_padded.extend(before[b_start:b_end]); after_padded.extend(after[a_start:a_end] + [edit_seq_tokens["padding"]] * (b_len - a_len))
            else: seq.extend([edit_seq_tokens["rep"]] * b_len + [edit_seq_tokens["add"]] * (a_len - b_len)); after_padded.extend(after[a_start:a_end]); before_padded.extend(before[b_start:b_end] + [edit_seq_tokens["padding"]] * (a_len - b_len))
        b_trav, a_trav = b_end, a_end
    seq.extend([edit_seq_tokens["eql"]] * (len(before) - b_trav)); before_padded.extend(before[b_trav:]); after_padded.extend(after[a_trav:])
    return seq, before_padded, after_padded

def write_tokens_to_file(tokens, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for token in tokens: f.write(token + '\n')

def read_tokens_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f: return [line.strip('\n') for line in f.readlines()]

def token_level_merge(a_tokens, b_tokens, o_tokens):
    temp_repo_path = TEMP_REPO_BASE_PATH / str(os.getpid())
    if temp_repo_path.exists(): shutil.rmtree(temp_repo_path)
    repo = Repo.init(temp_repo_path); _git = Git(temp_repo_path); _git.config("merge.conflictstyle", "diff3"); tmp_file = temp_repo_path / "tokens.txt"
    write_tokens_to_file(o_tokens, tmp_file); _git.add('.'); repo.index.commit("base")
    _git.branch('a_branch'); _git.checkout('a_branch'); write_tokens_to_file(a_tokens, tmp_file); _git.add('.'); repo.index.commit("commit a")
    _git.checkout('master'); _git.branch('b_branch'); _git.checkout('b_branch'); write_tokens_to_file(b_tokens, tmp_file); _git.add('.'); repo.index.commit("commit b")
    try: _git.merge('a_branch'); return [read_tokens_from_file(tmp_file)]
    except Exception: return parse_conflict_markers(tmp_file.read_text())

def parse_conflict_markers(content):
    hunks, lines, current_hunk, i = [], content.split('\n'), [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('<<<<<<<'):
            if current_hunk: hunks.append([h for h in current_hunk if h]); current_hunk = []
            a_conflict, o_conflict, b_conflict = [], [], []; i += 1
            while not lines[i].startswith('|||||||'): a_conflict.append(lines[i]); i += 1
            i += 1
            while not lines[i].startswith('======='): o_conflict.append(lines[i]); i += 1
            i += 1
            while not lines[i].startswith('>>>>>>>'): b_conflict.append(lines[i]); i += 1
            hunks.append((a_conflict, b_conflict, o_conflict))
        else: current_hunk.append(line)
        i += 1
    if current_hunk: current_hunk = [h for h in current_hunk if h];
    if current_hunk: hunks.append(current_hunk)
    return hunks