# This file will contain the PyTorch implementation of the MergeBERT model.

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class MergeBERTModel(nn.Module):
    """MergeBERT model for merge conflict resolution."""
    def __init__(self, encoder_path, num_labels=9):
        super(MergeBERTModel, self).__init__()
        self.num_labels = num_labels

        # The paper describes 4 parallel encoders. We'll instantiate one and use it four times.
        # This enforces weight sharing across the encoders, which is a common practice.
        config = RobertaConfig.from_pretrained(encoder_path)
        self.encoder = RobertaModel.from_pretrained(encoder_path, config=config)

        # Aggregation weights (learnable)
        self.aggregation_weights = nn.Parameter(torch.ones(4))

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, a_o_ids, o_a_ids, b_o_ids, o_b_ids, labels=None):
        """
        Forward pass of the model.
        For the MVP, we are not using the edit type embeddings yet.
        `*_ids` are tensors of shape (batch_size, seq_length).
        """
        # Pass each input through the shared encoder
        # The output of RobertaModel is a tuple. The first element is the last_hidden_state
        # and the second is the pooler_output.
        output_ao = self.encoder(input_ids=a_o_ids).pooler_output
        output_oa = self.encoder(input_ids=o_a_ids).pooler_output
        output_bo = self.encoder(input_ids=b_o_ids).pooler_output
        output_ob = self.encoder(input_ids=o_b_ids).pooler_output

        # Weighted aggregation of the [CLS] token representations (pooler_output)
        # We apply softmax to the weights to make them sum to 1.
        agg_weights = nn.functional.softmax(self.aggregation_weights, dim=0)

        aggregated_output = (
            agg_weights[0] * output_ao +
            agg_weights[1] * output_oa +
            agg_weights[2] * output_bo +
            agg_weights[3] * output_ob
        )

        # Classification
        logits = self.classifier(aggregated_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits
        }
