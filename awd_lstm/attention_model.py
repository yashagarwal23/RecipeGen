import torch
import torch.nn as nn

from awd_lstm.embed_regularize import embedded_dropout
from awd_lstm.locked_dropout import LockedDropout
#  from awd_lstm.weight_drop import WeightDrop
from entity_composite.LSTMcell import LSTM

class RNNAttentionModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, is_cuda = False):
        super(RNNAttentionModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.rnns = [LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
            attention = True if l == nlayers-1 else False, is_cuda = is_cuda) for l in range(nlayers)]
        self.last_layer = self.rnns[-1]

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights
        self.attetion = True

    def is_attention_model(self):
        return self.attetion

    def reset_last_layer(self):
        self.last_layer.previous_hidden = None
        self.last_layer.detach_attention_params()

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.rnns):
            layer_raw_output = []
            temp_h = hidden[l]
            for i in range(raw_output.size(0)):
                temp_output, temp_h = rnn(raw_output[i, :, :], temp_h)
                layer_raw_output.append(temp_output)
            layer_raw_output = torch.stack(layer_raw_output)
            raw_outputs.append(layer_raw_output)
            new_hidden.append(temp_h)
            if l != self.nlayers-1:
                raw_output = self.lockdrop(layer_raw_output, self.dropouth)
                outputs.append(layer_raw_output)
        outputs = torch.stack(outputs)

        output = self.lockdrop(layer_raw_output, self.dropout)
        hidden = new_hidden

        decoded = self.decoder(output.view(
            output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0)*output.size(1), -1)
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
