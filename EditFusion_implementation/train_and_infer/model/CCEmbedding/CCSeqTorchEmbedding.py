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
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )

    def forward(self, token_ids):
        batch_size, seq_length, _ = token_ids.size()

        token_ids = token_ids.view(-1, token_ids.size(-1))

        output = self.embedding(token_ids)

        output = output.view(batch_size, seq_length, -1, self.embedding_dim)
        return output
