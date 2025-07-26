import torch.nn as nn
import torch
from ..utils.tokenizer_util import vocab_size, pad_token_id
from transformers import AutoModel
from .CCSeqEmbedding import CCSeqEmbedding
import os

script_path = os.path.dirname(os.path.abspath(__file__))
bert_path = os.path.join(script_path, '../bert/CodeBERTa-small-v1')
# Model parameters
model_params = {
    'input_size': 7 + 768,      # Number of constructed features
    'output_size': 1,
    'hidden_size': 256,
    'num_layers': 4,
    'dropout': 0.2,
}

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.2, **kwargs):
        super(LSTMClassifier, self).__init__(*kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_first = True

        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=self.batch_first)
        
    # Use pretrained model for embedding
    # Concatenate three sequences, then use pretrained model for embedding
    self.cc_embedding = AutoModel.from_pretrained(bert_path)     # Use Roberta model with no head
    # for param in self.cc_embedding.parameters():
    #     param.requires_grad = False   # Freeze pretrained model parameters if needed
    self.cc_embedding.resize_token_embeddings(vocab_size)   # Update vocabulary size after adding new tokens
        
    self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

    def forward(self, position_features, origin_ids, modified_ids, edit_seq_ids, lengths):        
        # Use pretrained model to embed three sequences
        # Concatenate three sequences (origin, modified, edit_seq)
        # Note: Roberta does not have cls_token or sep_token
        catted_ids = torch.cat([origin_ids, modified_ids, edit_seq_ids], dim=2)   # N, L, 3*T
        # Embed as tensor of shape N, L, 3*T, embedding_dim
        # Flatten
        flattened_ids = catted_ids.view(-1, catted_ids.shape[-1])           # N * L, 3*T
        attention_mask = (flattened_ids != pad_token_id).type(torch.long) # 0 for pad, 1 for others
        bert_embedding = self.cc_embedding(flattened_ids, attention_mask=attention_mask).last_hidden_state     # N * L, 3*T, embedding_dim
        # Take the embedding of the first <s> token
        cls_embedding = bert_embedding[:, 0, :]    # N * L, embedding_dim
        
        # Reshape to batch_size * time_step * feature_size        
        cls_embedding = cls_embedding.view(catted_ids.shape[0], catted_ids.shape[1], -1)

        # Concatenate with position_features
        catted_embedding = torch.cat((position_features, cls_embedding), dim=2)
        # Pack sequence
        packed = torch.nn.utils.rnn.pack_padded_sequence(catted_embedding, lengths, batch_first=self.batch_first, enforce_sorted=True) # Already sorted in collate_fn
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(packed)
        # Unpack; since enforce_sorted=True, batch lengths can be used
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=self.batch_first, padding_value=12)
        # If bidirectional LSTM, merge both directions' hidden states
        if self.bidirectional:
            lstm_out = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], 2, self.hidden_size)     # N, L, 2, H
            lstm_out = torch.cat((lstm_out[:, :, 0, :], lstm_out[:, :, 1, :]), dim=2)               # N, L, 2H
        # Fully connected layer
        out = self.fc(lstm_out)
        # Apply sigmoid activation for final output
        out = torch.sigmoid(out)
        return out