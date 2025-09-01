import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_layer = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, lstm_output, mask):
        # lstm_output: [batch_size, seq_len, hidden_size]
        # mask: [batch_size, seq_len]

        # Generate query, key, value
        query = self.query_layer(lstm_output)  # [batch_size, seq_len, hidden_size]
        key = self.key_layer(lstm_output)  # [batch_size, seq_len, hidden_size]
        value = self.value_layer(lstm_output)  # [batch_size, seq_len, hidden_size]

        # Calculate attention scores
        attention_scores = torch.bmm(
            query, key.transpose(1, 2)
        )  # [batch_size, seq_len, seq_len]

        # Apply mask
        attention_scores = attention_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        # Normalize scores to probabilities
        attention_weights = torch.softmax(
            attention_scores, dim=2
        )  # softmax over seq_len

        # Apply attention weights to value
        weighted_output = torch.bmm(
            attention_weights, value
        )  # [batch_size, seq_len, hidden_size]

        return weighted_output, attention_weights
