"""
Defines two data structures: OffsetRange and SequenceDiff.
OffsetRange represents an interval, and SequenceDiff represents an edit script.
"""

from pprint import pprint
from typing import Callable, List
import numpy as np
from .tokenizer_util import edit_seq_tokens


class OffsetRange:
    """
    A range of offsets (0-based).
    Left-closed, right-open.
    """

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def intersect(self, other: "OffsetRange") -> "OffsetRange":
        l1, r1 = self.start, self.end
        l2, r2 = other.start, other.end
        L = max(l1, l2)
        R = min(r1, r2)

        if L < R:
            return OffsetRange(L, R)

        if l1 == r1:
            if l2 < l1 < r2:
                return OffsetRange(l1, r1)
        if l2 == r2:
            if l1 < l2 < r1:
                return OffsetRange(l2, r2)

        return None

    def __lt__(self, other):
        return (self.start, self.end) < (other.start, other.end)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OffsetRange):
            return self.start == other.start and self.end == other.end
        return False

    def __len__(self):
        return self.end - self.start


class SequenceDiff:
    def __init__(self, seq1Range: OffsetRange, seq2Range: OffsetRange):
        self.seq1Range = seq1Range
        self.seq2Range = seq2Range

    def __repr__(self):
        return f"SequenceDiff({self.seq1Range.start}, {self.seq1Range.end}, {self.seq2Range.start}, {self.seq2Range.end})"

    def __json__(self):
        return {
            "seq1Range": {"start": self.seq1Range.start, "end": self.seq1Range.end},
            "seq2Range": {"start": self.seq2Range.start, "end": self.seq2Range.end},
        }


def compute(
    sequence1: List[str],
    sequence2: List[str],
    equalityScore: Callable[[int, int], float] = lambda x, y: 1 if x == y else 0,
) -> List[SequenceDiff]:
    if len(sequence1) == 0 or len(sequence2) == 0:
        return [
            SequenceDiff(OffsetRange(0, len(sequence1)), OffsetRange(0, len(sequence2)))
        ]

    lcsLengths = np.zeros((len(sequence1), len(sequence2)))
    directions = np.zeros((len(sequence1), len(sequence2)))
    lengths = np.zeros((len(sequence1), len(sequence2)))

    for s1 in range(len(sequence1)):
        for s2 in range(len(sequence2)):
            horizontalLen = 0 if s1 == 0 else lcsLengths[s1 - 1, s2]
            verticalLen = 0 if s2 == 0 else lcsLengths[s1, s2 - 1]

            extendedSeqScore = 0
            if sequence1[s1] == sequence2[s2]:
                if s1 == 0 or s2 == 0:
                    extendedSeqScore = 0
                else:
                    extendedSeqScore = lcsLengths[s1 - 1, s2 - 1]

                extendedSeqScore += equalityScore(sequence1[s1], sequence2[s2])

                if s1 > 0 and s2 > 0 and directions[s1 - 1, s2 - 1] == 3:
                    extendedSeqScore += lengths[s1 - 1, s2 - 1]

            else:
                extendedSeqScore = -1

            newValue = max(horizontalLen, verticalLen, extendedSeqScore)

            if newValue == extendedSeqScore:
                prevLen = lengths[s1 - 1, s2 - 1] if s1 > 0 and s2 > 0 else 0
                lengths[s1, s2] = prevLen + 1
                directions[s1, s2] = 3
            elif newValue == horizontalLen:
                lengths[s1, s2] = 0
                directions[s1, s2] = 1
            elif newValue == verticalLen:
                lengths[s1, s2] = 0
                directions[s1, s2] = 2

            lcsLengths[s1, s2] = newValue

    # ==== 回溯 ====
    result: List[SequenceDiff] = []
    lastAligningPosS1 = len(sequence1)
    lastAligningPosS2 = len(sequence2)

    def reportDecreasingAligningPositions(s1: int, s2: int) -> None:
        nonlocal lastAligningPosS1, lastAligningPosS2
        if s1 + 1 != lastAligningPosS1 or s2 + 1 != lastAligningPosS2:
            result.append(
                SequenceDiff(
                    OffsetRange(s1 + 1, lastAligningPosS1),
                    OffsetRange(s2 + 1, lastAligningPosS2),
                )
            )
        lastAligningPosS1 = s1
        lastAligningPosS2 = s2

    s1 = len(sequence1) - 1
    s2 = len(sequence2) - 1
    while s1 >= 0 and s2 >= 0:
        if directions[s1, s2] == 3:
            reportDecreasingAligningPositions(s1, s2)
            s1 -= 1
            s2 -= 1
        else:
            if (
                directions[s1, s2] == 1
            ):
                s1 -= 1
            else:
                s2 -= 1

    reportDecreasingAligningPositions(-1, -1)
    result.reverse()
    return result


def get_edit_sequence(before: List[str], after: List[str]) -> List[List[str]]:
    """
    生成一个由 + - = padding <-> 构成的编辑序列以及填充后的序列
    Args:
        before: 原始序列
        after: 修改后的序列
    Returns:
        seq: 编辑序列
        before_padded: 填充后的序列
        after_padded: 填充后的序列
    """

    ess = compute(before, after)  # ? 应该排序了？检查一下

    eql_token = edit_seq_tokens["eql"]
    add_token = edit_seq_tokens["add"]
    del_token = edit_seq_tokens["del"]
    rep_token = edit_seq_tokens["rep"]
    padding_token = edit_seq_tokens["padding"]

    edit_seq = []
    before_padded = []
    after_padded = []
    before_traversed = 0
    after_traversed = 0

    for es in ess:
        before_start = es.seq1Range.start
        before_end = es.seq1Range.end
        after_start = es.seq2Range.start
        after_end = es.seq2Range.end
        before_len = len(es.seq1Range)
        after_len = len(es.seq2Range)

        edit_seq += [eql_token] * (before_start - before_traversed)
        before_padded += before[before_traversed:before_start]
        after_padded += after[after_traversed:after_start]

        if before_len == 0:
            # 插入操作
            edit_seq += [add_token] * after_len
            before_padded += [padding_token] * after_len
            after_padded += after[after_start:after_end]
        elif after_len == 0:
            # 删除操作
            edit_seq += [del_token] * before_len
            before_padded += before[before_start:before_end]
            after_padded += [padding_token] * before_len
        else:
            # 替换操作
            if before_len > after_len:
                edit_seq += [rep_token] * after_len + [del_token] * (
                    before_len - after_len
                )
                before_padded += before[before_start:before_end]
                after_padded += after[after_start:after_end] + [padding_token] * (
                    before_len - after_len
                )
            else:
                edit_seq += [rep_token] * before_len + [add_token] * (
                    after_len - before_len
                )
                after_padded += after[after_start:after_end]
                before_padded += before[before_start:before_end] + [padding_token] * (
                    after_len - before_len
                )

        before_traversed = before_end
        after_traversed = after_end

    edit_seq += [eql_token] * (len(before) - before_traversed)
    before_padded += before[before_traversed:]
    after_padded += after[after_traversed:]

    return edit_seq, before_padded, after_padded


if __name__ == "__main__":
    before = ["a", "b", "c", "d", "e", "f", "g"]
    after = ["a", "1", "b", "c", "dddd", "ee", "e", "g", "2", "3"]
    pprint(get_edit_sequence(before, after))
