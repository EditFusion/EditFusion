# This file will contain helper functions for data processing,
# including diff3, sequence alignment, and tokenization.

import diff_match_patch as dmp_module
from transformers import PreTrainedTokenizer

def get_diffs(text1, text2):
    dmp = dmp_module.diff_match_patch()
    diff = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemantic(diff)
    return diff

def line_level_diff3(a_content, b_content, o_content):
    """
    Performs a line-level 3-way diff and extracts conflict regions.
    Returns (prefix, suffix, [(a_chunk, b_chunk, o_chunk), ...])
    """
    # This is a complex function to implement correctly.
    # For this MVP, we will assume a simplified scenario where there is
    # only one conflict block per file. A full implementation would need
    # to handle multiple conflict blocks.

    # This is a placeholder for a proper diff3 implementation.
    # A real implementation would involve finding common regions and
    # identifying conflicting hunks.
    
    # Let's find the conflict markers if they are present
    # This is a trick if the input is already in diff3 format.
    # The dataset does not seem to be in this format, so we need to do it from scratch.
    
    # A proper implementation is out of scope for a quick MVP.
    # I will mock this function to return the whole file as a single conflict
    # if they are different, and assume no prefix/suffix.
    # This is a major simplification from the paper.

    if a_content == o_content and b_content == o_content:
        # No changes
        return a_content, "", []
    
    if a_content == b_content:
        # A and B are the same, but different from O
        return "", "", [(a_content, b_content, o_content)]

    # Simplified assumption: the whole file is one conflict
    return "", "", [(a_content, b_content, o_content)]


def tokenize_and_token_level_diff3(a_chunk, b_chunk, o_chunk, tokenizer: PreTrainedTokenizer):
    """
    Tokenizes the conflicting chunks and performs a token-level diff3.
    """
    a_tokens = tokenizer.tokenize(a_chunk)
    b_tokens = tokenizer.tokenize(b_chunk)
    o_tokens = tokenizer.tokenize(o_chunk)

    # Again, a proper token-level diff3 is complex.
    # We will use a similar simplification as above.
    # The function will return the token lists as a single conflict.
    
    token_conflict = {
        'a': a_tokens,
        'b': b_tokens,
        'o': o_tokens,
        'prefix': [],
        'suffix': []
    }
    
    return [token_conflict]


def align_and_get_edit_sequence(seq1, seq2):
    """
    Aligns two token sequences and returns the aligned sequences and an edit script.
    Returns (aligned1, aligned2, edit_sequence)
    """
    dmp = dmp_module.diff_match_patch()
    
    # diff_main works on strings, so we need to convert token lists to a string representation
    # This is a common technique using unicode characters as proxies for tokens.
    
    token_map1, text1 = _tokens_to_text(seq1)
    token_map2, text2 = _tokens_to_text(seq2)
    
    diffs = dmp.diff_main(text1, text2)
    
    aligned1 = []
    aligned2 = []
    edit_sequence = []
    
    for op, data in diffs:
        if op == dmp.DIFF_EQUAL:
            for char in data:
                aligned1.append(token_map1[char])
                aligned2.append(token_map2[char])
                edit_sequence.append('=')
        elif op == dmp.DIFF_INSERT:
            for char in data:
                aligned1.append('[PAD]')
                aligned2.append(token_map2[char])
                edit_sequence.append('+')
        elif op == dmp.DIFF_DELETE:
            for char in data:
                aligned1.append(token_map1[char])
                aligned2.append('[PAD]')
                edit_sequence.append('-')

    # The paper also mentions a replacement operation '↔'.
    # diff-match-patch doesn't have a replace op, it's a delete then an insert.
    # We can post-process to create replacements, but for MVP we'll skip this.

    return aligned1, aligned2, edit_sequence

def _tokens_to_text(tokens):
    """Converts a list of tokens to a string for diffing."""
    token_map = {}
    text = []
    for i, token in enumerate(tokens):
        char = chr(i + 1000) # Use a high range of unicode characters
        token_map[char] = token
        text.append(char)
    return token_map, "".join(text)
