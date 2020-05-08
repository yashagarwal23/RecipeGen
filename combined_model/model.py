import torch.nn as nn
import torch

class combined_model(nn.Module):
    def __init__(self,model_entity_composite, model_type):
        super(combined_model, self).__init__()
        self.model_entity_composite = model_entity_composite
        self.model_type = model_type

    def is_attention_model(self):
        return False

    def forward(self, input, input2, hidden, return_h = False):
        output_type, new_type_hidden = self.model_type(input2, hidden[1], False)
        type_input_to_entity_composite = torch.argmax(output_type, dim=1).view(input2.size())
        output, new_entity_composite_hidden, rnn_hs, dropped_rnn_hs = self.model_entity_composite(input, type_input_to_entity_composite, hidden[0], True)
        if return_h:
            return output, (new_entity_composite_hidden, new_type_hidden), rnn_hs, dropped_rnn_hs
        return output, (new_entity_composite_hidden, new_type_hidden)

    def init_hidden(self, bsz):
        return (self.model_entity_composite.init_hidden(bsz), self.model_type.init_hidden(bsz))
