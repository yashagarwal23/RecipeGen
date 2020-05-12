import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from rnn_model.decoder_functions import attention_combined as decoder_function

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, attention=False, is_cuda = False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bias=bias)
        self.attention=attention
        if attention:
            self.attention_linear = nn.Linear(2*hidden_size, hidden_size, bias=bias)
            self.ws = Parameter(torch.randn(hidden_size, hidden_size))
            self.vs = Parameter(torch.randn(hidden_size, 1))
        self.previous_hidden = None
        self.device = "cuda:0" if is_cuda else "cpu"

    def detach_attention_params(self):
        self.ws.detach_().requires_grad = True
        self.vs.detach_().requires_grad = True

    def __str__(self):
        return 'LSTM ({}, {} attention:{})'.format(self.input_size, self.hidden_size, self.attention)

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, state):
        batch_size = input.size(0)
        output, new_state = self.lstm(input.unsqueeze(0), state)
        if not self.attention:
            return output.squeeze(0), state

        if self.previous_hidden is None:
            self.previous_hidden = torch.zeros((batch_size, 1, self.hidden_size), device=self.device)

        combined_hy = decoder_function(output.squeeze(0), self.previous_hidden, self.ws, self.vs)
        new_hy = self.attention_linear(combined_hy)
        self.previous_hidden = torch.cat([self.previous_hidden, output.permute([1, 0, 2])], dim=1)
        return new_hy, state
