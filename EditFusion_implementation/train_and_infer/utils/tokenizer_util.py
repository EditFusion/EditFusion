# Load model directly
from typing import List
from transformers import AutoTokenizer
import os

# tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
# 从本地加载 tokenizer，模型在 train_and_infer/bert/CodeBERTa-small-v1
script_path = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(script_path, '../bert/CodeBERTa-small-v1')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model_max_length = tokenizer.model_max_length

# 获取特殊 token 的 id
bos_token_id = tokenizer.bos_token_id
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id

# 加入编辑序列的特殊 token
edit_seq_tokens = {
    'eql': '<cc_eql_token>',
    'add': '<cc_add_token>',
    'del': '<cc_del_token>',
    'rep': '<cc_rep_token>',
    'padding': '<cc_padding_token>',       # Notice: `padding token` is different from `padding token` of a tokenizer
}
tokenizer.add_tokens(list(edit_seq_tokens.values()))
vocab_size = len(tokenizer)

def get_token_id(token: str) -> int:
    return tokenizer.convert_tokens_to_ids(token)

def encode_text_to_tokens(text: str) -> List[str]:
    return tokenizer.tokenize(text, truncation=True, max_length=model_max_length)   # truncation to remove warning

def encode_text_to_id(text: str) -> list[int]:
    return tokenizer.encode(text)

def encode_tokens_to_ids(tokens: List[str]) -> list[int]:
    return tokenizer.convert_tokens_to_ids(tokens[:model_max_length])   # truncation to remove warning

def decode_ids_to_tokens(token_ids: list[int]) -> List[str]:
    return tokenizer.convert_ids_to_tokens(token_ids)

if __name__ == '__main__':
    text = '\t\t\t @Parameters(commandDescription = "Frequency count a structured input instance file.")\n   \t '
    tokens = encode_text_to_tokens(text)
    print(tokens)
    print(encode_tokens_to_ids(tokens))

    print(encode_text_to_id(text))
    print(decode_ids_to_tokens([0, 1, 2]))
    print(tokenizer(text))

    print(tokenizer.bos_token_id)
    print(tokenizer.eos_token_id)
    