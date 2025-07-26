from typing import List, Union
import numpy as np


class OffsetRange:
    '''
    A range of offsets (0-based).
    左闭右开
    '''

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def intersect(self, other: 'OffsetRange') -> Union['OffsetRange', None]:
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
            'seq1Range': {
                'start': self.seq1Range.start,
                'end': self.seq1Range.end
            },
            'seq2Range': {
                'start': self.seq2Range.start,
                'end': self.seq2Range.end
            }
        }


def compute(sequence1: List[str], sequence2: List[str], 
            equalityScore = lambda x, y: 1 if x == y else 0) -> List[SequenceDiff]:
    # 如果其中一个字符串为空，直接返回 trivial LCS 操作序列
    if len(sequence1) == 0 or len(sequence2) == 0:
        # return a List containing a single SequenceDiff in which both ranges are from 0 to the lentgh of the non-empty sequence
        return [SequenceDiff(OffsetRange(0, len(sequence1)), OffsetRange(0, len(sequence2)))]

    lcsLengths = np.zeros((len(sequence1), len(sequence2)))     # 走这条路线的 LCS 总长度
    directions = np.zeros((len(sequence1), len(sequence2)))
    lengths = np.zeros((len(sequence1), len(sequence2)))        # 记录斜线走了多少步

    # ==== 初始化 lcsLengths ====
    for s1 in range(len(sequence1)):
        for s2 in range(len(sequence2)):
            # 计算水平方向和垂直方向的 LCS 长度
            horizontalLen = 0 if s1 == 0 else lcsLengths[s1-1, s2]
            verticalLen = 0 if s2 == 0 else lcsLengths[s1, s2 - 1]

            # 计算斜对角的 LCS 长度
            extendedSeqScore = 0
            if sequence1[s1] == sequence2[s2]:
                # 初始化 extendedSeqScore
                if s1 == 0 or s2 == 0:
                    extendedSeqScore = 0
                else:
                    extendedSeqScore = lcsLengths[s1 - 1, s2 - 1]

                # 如果两个字符相同，则当前位置的 LCS 长度可以由左上角的 LCS 长度加 1 得到
                # 如果参数提供了匹配得分函数，则使用该函数计算得分；否则默认得分为 1
                extendedSeqScore += equalityScore(sequence1[s1], sequence2[s2])

                # 如果左上角的 LCS 操作是斜线（表示字符是相同的），那么斜线方向的 LCS 长度需要加上相应的长度
                if s1 > 0 and s2 > 0 and directions[s1 - 1, s2 - 1] == 3:
                    extendedSeqScore += lengths[s1 - 1, s2 - 1]

            else:
                # 如果两个字符不同，则当前位置的 LCS 长度为 -1，表示没有 LCS 长度
                extendedSeqScore = -1

            # 计算当前位置的 LCS 长度
            newValue = max(horizontalLen, verticalLen, extendedSeqScore)

            if newValue == extendedSeqScore:
                # 如果左上角的 LCS 操作是斜线，那么斜线操作
                # Prefer diagonals
                prevLen = lengths[s1 - 1, s2 - 1] if s1 > 0 and s2 > 0 else 0
                lengths[s1, s2] = prevLen + 1
                directions[s1, s2] = 3
            elif newValue == horizontalLen:
                # 如果左边的 LCS 操作更优，那么向左操作
                lengths[s1, s2] = 0
                directions[s1, s2] = 1
            elif newValue == verticalLen:
                # 如果上面的 LCS 操作更优，那么向上操作
                lengths[s1, s2] = 0
                directions[s1, s2] = 2

            lcsLengths[s1, s2] = newValue

    # ==== 回溯 ====
    result: List[SequenceDiff] = []
    lastAligningPosS1 = len(sequence1)  # 先置于末尾
    lastAligningPosS2 = len(sequence2)  # 先置于末尾

    # todo 这里先按 int
    def reportDecreasingAligningPositions(s1: int, s2: int) -> None:
        nonlocal lastAligningPosS1, lastAligningPosS2
        if s1 + 1 != lastAligningPosS1 or s2 + 1 != lastAligningPosS2:
            # 下一步不是走斜线（是对角线的终点），新建一个 SequenceDiff 直到上一个对应的行号，放入 result	（我的理解是，如果不是连续对应，说明不是同一个对应块，断开）
            result.append(SequenceDiff(
                OffsetRange(s1 + 1, lastAligningPosS1),
                OffsetRange(s2 + 1, lastAligningPosS2),
            ))
        # 标记对应行号
        lastAligningPosS1 = s1
        lastAligningPosS2 = s2

    s1 = len(sequence1) - 1
    s2 = len(sequence2) - 1
    while s1 >= 0 and s2 >= 0:
        # 有斜线优先走斜线
        if directions[s1, s2] == 3:
            # 如果是斜线操作，记录位置并 **继续** 斜线操作
            reportDecreasingAligningPositions(s1, s2)
            s1 -= 1
            s2 -= 1
        else:
            if directions[s1, s2] == 1: # 这里的顺序会影响到编辑脚本生成的结果，可能会有lcs相同的不同的编辑方式
                # 如果是向左操作，向左移动
                s1 -= 1
            else:
                # 如果是向上操作，向上移动
                s2 -= 1

    reportDecreasingAligningPositions(-1, -1)
    result.reverse()
    # 返回结果
    return result

 