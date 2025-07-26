# Load model directly
from typing import List
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")

def encode(text: str) -> list[int]:
    return tokenizer.encode(text)

def decode(token_ids: list[int]) -> List[str]:
    return tokenizer.convert_ids_to_tokens(token_ids) # type: ignore

if __name__ == '__main__':
    output = encode('\t\t\t @Parameters(commandDescription = "Frequency count a structured input instance file.")\n   \t ')
    output = encode('\t这是一段临时测试，测试是否能够编码。😂')
    output = encode('')
    print(output)
    print(tokenizer.convert_ids_to_tokens(output))