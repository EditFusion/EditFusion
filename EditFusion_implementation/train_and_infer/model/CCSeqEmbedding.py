import torch.nn as nn

class CCSeqEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_token_id):
        super(CCSeqEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        # Initialize embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id)  # Specify which token is the padding token
        
    def forward(self, token_ids):
        # Expected shape of token_ids: [batch_size, seq_length, 512]
        # 512 is the maximum token length

        batch_size, seq_length, _ = token_ids.size()
        # Reshape to 2D: (batch_size * seq_length) * 512
        token_ids = token_ids.view(-1, token_ids.size(-1))

        # Apply embedding
        output = self.embedding(token_ids)  # [batch_size * seq_length, 512, embedding_dim]

        # Restore shape to [batch_size, seq_length, 512, embedding_dim]
        output = output.view(batch_size, seq_length, -1, self.embedding_dim)
        return output
