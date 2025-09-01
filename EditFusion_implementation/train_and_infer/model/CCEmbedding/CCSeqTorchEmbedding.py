import torch.nn as nn


class CCSeqEmbedding(nn.Module):
    """
    DEPRECATED: This class is not used in the final version of the project.

    embedding for code change sequence
    """

    def __init__(self, vocab_size, embedding_dim, pad_token_id):
        super(CCSeqEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        # 初始化嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )  # 告诉 Embedding 层哪个 token 是 padding token

    def forward(self, token_ids):
        # token_ids 的期望形状: [batch_size, seq_length, 512]
        # 512 是 token 的最大长度

        # 获取批次大小
        batch_size, seq_length, _ = token_ids.size()

        # 调整为二维: (batch_size * seq_length) * 512
        token_ids = token_ids.view(-1, token_ids.size(-1))

        # 应用嵌入
        output = self.embedding(token_ids)  # [batch_size * seq_length, 512, 128]

        # 恢复 [batch_size, seq_length, 512, 128]   # todo 这维度太多了吧？
        output = output.view(batch_size, seq_length, -1, self.embedding_dim)
        return output
