#! /bin/python3

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """
    Implementation of a simple LSTM module, with a fully connected layer at the end.
    """
    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, embedding_weights, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, label_size)
        self.act = nn.Softmax(dim=1)

    def forward(self, sentence, src_len, train=True):
        embeds = self.word_embeddings(sentence)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embeds, src_len)

        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        hidden = self.dropout(hidden)

        fc_output = self.fc(hidden)
        outputs = self.act(fc_output)
        return outputs


class LSTM(nn.Module):
    """
    Implementation of a simple LSTM module, with a fully connected layer at the end.
    """
    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, embedding_weights, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weights, freeze=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            bidirectional=False,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, label_size)
        self.act = nn.Softmax(dim=1)

    def forward(self, sentence, src_len, train=True):
        embeds = self.word_embeddings(sentence)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embeds, src_len)

        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)

        hidden = hidden.squeeze(dim=0)
        hidden = self.dropout(hidden)
        dense_outputs = self.fc(hidden)
        outputs = self.act(dense_outputs)
        return outputs

