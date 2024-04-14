# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class BiLSTMLayer(nn.Module):
    def __init__(self, input_size=1024, debug=False, hidden_size=256, num_layers=2, dropout=0.3,
                 bidirectional=True, rnn_type='LSTM'):
        super(BiLSTMLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.rnn_type = rnn_type
        self.debug = debug
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )

    def forward(self, src_feats, src_lens, max_len, hidden=None):
        """
        Args:
            - src_feats: (batch_size, max_src_len, 512)
            - src_lens: (batch_size)
        Returns:
            - outputs: (max_src_len, batch_size, hidden_size * num_directions)
            - hidden : (num_layers, batch_size, hidden_size * num_directions)
        """
        packed_emb = nn.utils.rnn.pack_padded_sequence(src_feats, src_lens, batch_first=True, enforce_sorted=True)
        packed_outputs, hidden = self.rnn(packed_emb)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=max_len)

        return rnn_outputs
