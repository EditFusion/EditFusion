import torch.nn as nn
import torch

from train_and_infer.model.Attention import Attention


class GRUClassifier(nn.Module):
    """
    GRU model utilizing the edit script to judge whether to accept some of the edit scripts
    need a CCEmbedding class to embed the input explicitly aligned code change
    """

    def __init__(
        self,
        input_size,
        output_size,
        CCEmbedding_class,
        max_es_len=20,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.2,
        **kwargs
    ):
        super(GRUClassifier, self).__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.CCEmbedding_class = CCEmbedding_class()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_first = True
        self.max_es_len = max_es_len

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=self.batch_first,
        )

        # self.attention = Attention(hidden_size * 2 if bidirectional else hidden_size)
        self.fc = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, output_size
        )

    def forward(self, feats, lengths):

        cc_embedding = self.CCEmbedding_class(feats)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            cc_embedding,
            lengths.to("cpu"),
            batch_first=self.batch_first,
            enforce_sorted=True,
        )
        gru_out, h_n = self.gru(packed)
        gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            gru_out, batch_first=self.batch_first, padding_value=12
        )

        assert gru_out.shape[1] <= self.max_es_len
        if gru_out.shape[1] < self.max_es_len:
            gru_out = torch.nn.functional.pad(
                gru_out, (0, 0, 0, self.max_es_len - gru_out.shape[1], 0, 0), value=0
            )
        out = self.fc(gru_out)
        return out
