import operator
from main import args, data_without_types
from awd_lstm.build_model import get_model as get_rnn_model, model_load
from data_utils import load_text_dataset
import torch
import torch.nn as nn
from queue import PriorityQueue

EOS_TOKEN = '<eos>'
LogSoftmax = nn.LogSoftmax(dim=1)

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

def idx2word(idx, corpus):
    return corpus.dictionary.idx2word[idx]

def word2idx(word, corpus):
    return corpus.dictionary.word2idx[word]

def get_next_word(model, corpus, word, hidden, isIndex = True):
    if not isIndex:
        word = word2idx(word, corpus)
    model_awd_lstm.eval()
    output, new_hidden = model_awd_lstm(torch.LongTensor([word]).view(1, 1), hidden)
    return torch.argmax(LogSoftmax(output)).data, new_hidden

def beam_search(model, corpus, hidden, initial_sentence):
    model.eval()

    beam_width = 10
    topk = 1
    sentence = []
    #  hidden.requires_grad = False
    for word in initial_sentence.split(' '):
        next_word, hidden = get_next_word(model, corpus, word, hidden, False)
        sentence.append(idx2word(next_word, corpus))

    start_word = torch.LongTensor([next_word], device = "gpu" if args["cuda"] else "cpu")

    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    node = BeamSearchNode(hidden, None, start_word, 0, 1)
    nodes = PriorityQueue()

    nodes.put((-node.eval(), node))
    qsize = 1
    while True:
        if qsize > 2000: break

        score, n = nodes.get()
        new_word = n.wordid
        hidden = n.h

        if new_word == word2idx(EOS_TOKEN, corpus) and n.prevNode != None:
            endnodes.append((score, n))

            if len(endnodes) >= number_required:
                break
            else:
                continue

        output, hidden = model(new_word.view(1, 1), hidden)
        log_prob, indexes = torch.topk(output, beam_width)
        next_nodes = []

        for new_k in range(beam_width):
            decoded_t = indexes[0][new_k].view(1, -1)
            log_p = log_prob[0][new_k].item()

            node = BeamSearchNode(hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
            score = -node.eval()
            next_nodes.append((score, node))

        for i in range(len(next_nodes)):
            score, nn = next_nodes[i]
            nodes.put((score, nn))
        qsize += len(next_nodes) - 1

    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    utterances = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance = []
        utterance.append(n.wordid)
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid)

        utterance = utterance[::-1]
        utterances.append(utterance)

    return utterances

corpus_awd_lstm = load_text_dataset(data_without_types)
model_awd_lstm, _, _ = get_rnn_model(corpus_awd_lstm, args)
model_awd_lstm_state_dict, _, _ = model_load("model_awd_lstm.pt", "cpu")
model_awd_lstm.load_state_dict(model_awd_lstm_state_dict)

hidden = model_awd_lstm.init_hidden(1)
search = [idx2word(word.item(), corpus_awd_lstm) for word in beam_search(model_awd_lstm, corpus_awd_lstm, hidden, "preheat the oven")[0]]
print(search)
