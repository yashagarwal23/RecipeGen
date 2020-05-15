import operator
import os
import re
from main import args
import torch
import torch.nn as nn
from queue import PriorityQueue

args["cuda"] = False

def idx2word(idx, corpus):
    return corpus.dictionary.idx2word[idx]

def word2idx(word, corpus):
    return corpus.dictionary.word2idx[word]

EOS_TOKEN = '<eos>'
Softmax = nn.Softmax(dim=1)
device = "cuda:0" if args["cuda"] else "cpu"

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

    def __lt__(self, other):
        return self.eval() < other.eval()

    def __le__(self, other):
        return self.eval() <= other.eval()

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def get_next_word(model, corpus, word, hidden, isIndex = True):
    if not isIndex:
        word = word2idx(word, corpus)
    #  model.eval()
    output, new_hidden = model(torch.LongTensor([word]).view(1, 1), hidden)
    return Softmax(output), new_hidden


def beam_search(model, corpus, hidden, initial_sentence):
    if model.is_attention_model():
        model.reset_last_layer()
    beam_width = 15
    topk = 8
    for word in initial_sentence:
        output, hidden = get_next_word(model, corpus, word, hidden, False)

    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))
    nodes = PriorityQueue()
    qsize = 0

    output = torch.log(output)
    log_prob, indexes = torch.topk(output, beam_width)
    next_nodes = []

    for new_k in range(beam_width):
        decoded_t = indexes[0][new_k].view(1, -1)
        log_p = log_prob[0][new_k].item()

        node = BeamSearchNode(hidden, None, decoded_t, 0, 1)
        score = -node.eval()
        next_nodes.append((score, node))

    for i in range(len(next_nodes)):
        score, nn = next_nodes[i]
        nodes.put((score, nn))
    qsize += len(next_nodes) - 1

    while True:
        if qsize > 20000: break
        score, n = nodes.get()
        new_word = n.wordid
        hidden = n.h

        if new_word == word2idx(EOS_TOKEN, corpus) and n.prevNode != None:
            endnodes.append((score, n))

            if len(endnodes) >= number_required:
                break
            else:
                continue

        output, hidden = get_next_word(model, corpus, new_word.view(1, 1), hidden)
        output = torch.log(output)
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