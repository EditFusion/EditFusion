from typing import List
from transformers import AutoTokenizer
import os

script_path = os.path.dirname(os.path.abspath(__file__))
bert_path = os.path.join(script_path, "../bert/CodeBERTa-small-v1")

tokenizer = AutoTokenizer.from_pretrained(bert_path)
model_max_length = tokenizer.model_max_length

bos_token_id = tokenizer.bos_token_id
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id

edit_seq_tokens = {
    "eql": "<cc_eql_token>",
    "add": "<cc_add_token>",
    "del": "<cc_del_token>",
    "rep": "<cc_rep_token>",
    "padding": "<cc_padding_token>",
}
tokenizer.add_tokens(list(edit_seq_tokens.values()))
vocab_size = len(tokenizer)


# 获取指定 token 的 id
def get_token_id(token: str) -> int:
    return tokenizer.convert_tokens_to_ids(token)


def encode_text_to_tokens(text: str) -> List[str]:
    """
    将输入文本分割为 tokens[]
    """
    return tokenizer.tokenize(
        text, truncation=True, max_length=model_max_length
    )  # truncation to remove warning


def encode_text_to_id(text: str) -> list[int]:
    """
    直接将输入文本嵌入为 id，包含开头的 <s> 与 </s>
    """
    return tokenizer.encode(text)


def encode_tokens_to_ids(tokens: List[str]) -> list[int]:
    return tokenizer.convert_tokens_to_ids(
        tokens[:model_max_length]
    )

def decode_ids_to_tokens(token_ids: list[int]) -> List[str]:
    return tokenizer.convert_ids_to_tokens(token_ids)


if __name__ == "__main__":
    text = '\t\t\t @Parameters(commandDescription = "Frequency count a structured input instance file.")\n   \t '
    tokens = encode_text_to_tokens(text)
    print(tokens)
    print(encode_tokens_to_ids(tokens))

    print(encode_text_to_id(text))
    print(decode_ids_to_tokens([0, 1, 2]))
    print(tokenizer(text))

    print(tokenizer.bos_token_id) # 0
    print(tokenizer.eos_token_id) # 2
    print(tokenizer.pad_token_id) # 1
    print(tokenizer.sep_token_id) # 2
    print(tokenizer.cls_token_id) # 0

    print(tokenizer.special_tokens_map)
    print(encode_tokens_to_ids(["<cc_padding_token>"]))