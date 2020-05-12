import torch.nn as nn
import torch
class hierarchial_model(nn.Module):
    def __init__(self, model_subtype_to_word, model_type_to_subtype, model_type):
        super(hierarchial_model, self).__init__()
        self.model_subtype_to_word = model_subtype_to_word
        self.model_type_to_subtype = model_type_to_subtype
        self.model_type = model_type

    def reset(self):
        self.model_type.reset()
        self.model_type_to_subtype.reset()
        self.model_subtype_to_word.reset()

    def is_attention_model(self):
        return False

    def forward(self, input, hidden, return_h = False):
        output_type, new_type_hidden = self.model_type(input, hidden[2], return_h=False, emb_out=True)
        # input_type_to_subtype = torch.argmax(output_type, dim=1).view(input.size())

        # output_type_to_subtype, new_type_to_subtype_hidden = self.model_type_to_subtype(input, input_type_to_subtype, hidden[1])
        output_type_to_subtype, new_type_to_subtype_hidden = self.model_type_to_subtype(input, output_type, hidden[1], return_h=False, emb_inp=True, emb_out=True)
        # input_subtype_to_word = torch.argmax(output_type_to_subtype, dim=1).view(input.size())

        # output, new_subtype_to_word_hidden, rnn_hs, dropped_rnn_hs = self.model_subtype_to_word(input, input_subtype_to_word, hidden[0], True)
        output, new_subtype_to_word_hidden, rnn_hs, dropped_rnn_hs = self.model_subtype_to_word(input, output_type_to_subtype, hidden[0], return_h=True, emb_inp=True, emb_out=False)
        if return_h:
            return output, (new_subtype_to_word_hidden, new_type_to_subtype_hidden, new_type_hidden), rnn_hs, dropped_rnn_hs
        return output, (new_subtype_to_word_hidden, new_type_to_subtype_hidden, new_type_hidden)

    def init_hidden(self, bsz):
        return (
            self.model_subtype_to_word.init_hidden(bsz),
            self.model_type_to_subtype.init_hidden(bsz),
            self.model_type.init_hidden(bsz)
        )
