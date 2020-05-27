import torch.nn as nn
import torch

class combined_model(nn.Module):
    def __init__(self, model_type_to_word, model_type):
        super(combined_model, self).__init__()
        self.model_type_to_word = model_type_to_word
        self.model_type = model_type
        self.model_list = [model_type_to_word, model_type]
        self.num_hidden_params = sum([m.num_hidden_params for m in self.model_list])

    def reset(self):
        self.model_type.reset()
        self.model_type_to_word.reset()

    def is_attention_model(self):
        return False

    def forward(self, input, hidden, return_h = False, emb_inp = False, emb_out = False):
        input = input.unsqueeze(0)
        output_type, new_type_hidden = self.model_type(input, hidden[1], return_h=False, emb_inp=emb_inp, emb_out=True)

        output, new_type_to_word_hidden, rnn_hs, dropped_rnn_hs = self.model_type_to_word(input, output_type, hidden[0], return_h=True, emb_inp1=emb_inp, emb_inp2=True, emb_out=emb_out)
        
        if return_h:
            return output, (new_type_to_word_hidden, new_type_hidden), rnn_hs, dropped_rnn_hs
        return output, (new_type_to_word_hidden, new_type_hidden)

    def init_hidden(self, bsz):
        return (
            self.model_type_to_word.init_hidden(bsz),
            self.model_type.init_hidden(bsz)
        )
    
    def init_hidden_from_encoder(self, hidden):
        hidden = hidden.view(hidden.size(0), len(self.model_list), -1)
        hiddens = []
        for i, model in enumerate(self.model_list):
            hiddens.append(model.init_hidden_from_encoder(hidden[:, i, :].squeeze(1)))
        return tuple(hiddens)