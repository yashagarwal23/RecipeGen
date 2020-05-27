import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=2, bidirectional=True, dropout=0.3):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.num_layers = num_layers
        self.rnn = nn.GRU(emb_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)
        return output, final


class EncoderDecoder(nn.Module):
    """ Base Encoder Decoder Wrapper """

    def __init__(self, encoder, decoder, bridge, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bridge = bridge

    def forward(self, input, target):
        input, input_lengths, input_masks = input
        target, target_lengths, target_masks = target

        encoder_hidden, encoder_final = self.encoder(input, input_lengths)
        encoder_final = encoder_final.permute(1, 0, 2)
        encoder_final = encoder_final.contiguous().view(encoder_final.size(0), -1)
        hidden_encoder_to_decoder = self.bridge(encoder_final)
        return self.decoder(target, encoder_hidden, encoder_final, input_masks, target_masks, hidden_encoder_to_decoder)


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, decoder_model, attention):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.attention = attention
        self.decoder_model = decoder_model
        self.lm_input_layer = nn.Linear(emb_size + 2*attention.encoder_hidden_size, emb_size, bias=True)
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        query = hidden[0][-1].permute(1, 0, 2).contiguous()
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)
        rnn_input = torch.cat([prev_embed, context], dim=2)
        rnn_input = self.lm_input_layer(rnn_input).squeeze(1)
        output, hidden = self.decoder_model(rnn_input, hidden, emb_inp=True)
        return output, hidden
    
    def forward(self, trg, encoder_hidden, encoder_final, src_mask, trg_mask, hidden=None, max_len=None):
        if max_len is None:
            max_len = trg_mask.size(-1)
        hidden = self.decoder_model.init_hidden_from_encoder(hidden)
        proj_key = self.attention.key_layer(encoder_hidden)
        decoder_states = []
        trg_embed = self.embedding(trg)

        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden = self.forward_step(
              prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output.unsqueeze(1))
        decoder_states = torch.cat(decoder_states, dim=1)
        return decoder_states, hidden


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, encoder_hidden_size, decoder_out_size, query_size=None):
        super(BahdanauAttention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.key_layer = nn.Linear(2*encoder_hidden_size, decoder_out_size, bias=False)
        self.energy_layer = nn.Linear(decoder_out_size, 1, bias=False)
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"
        
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        scores.data.masked_fill_(mask == 0, -float('inf'))
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas
        context = torch.bmm(alphas, value)
        return context, alphas


def get_encoder_decoder_model(ingred_vocab_size, recipe_vocab_size, combined_language_model, args):
    encoder = Encoder(ingred_vocab_size, args['ingred_embeddiing_size'], args['encoder_nhid'], 
                    args['encoder_num_layers'], args['encoder_bidirectional'], args['encoder_dropout'])

    bridge = nn.Linear(args['encoder_num_layers'] * (1+args["encoder_bidirectional"]) * args['encoder_nhid'],
                        combined_language_model.num_hidden_params)

    attention = BahdanauAttention(args['encoder_nhid'], args['decoder_embedding_size'], args['decoder_embedding_size'])
    decoder = Decoder(recipe_vocab_size, args['decoder_embedding_size'], combined_language_model, attention)

    encoder_decoder = EncoderDecoder(encoder, decoder, bridge)
    return encoder_decoder