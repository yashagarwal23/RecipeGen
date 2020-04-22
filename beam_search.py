import operator
import os
import re
from main import args, data_without_types, data_with_type, data_entity_composite
from data_utils import load_text_dataset, load_entity_composite_dataset
import torch
import torch.nn as nn
from queue import PriorityQueue

args["cuda"] = False

def idx2word(idx, corpus):
    return corpus.dictionary.idx2word[idx]

def word2idx(word, corpus):
    return corpus.dictionary.word2idx[word]

def get_type(word, index = False):
    if index:
        word_ori = idx2word(word, corpus_entity_composite)
        if word_ori in all_type_entities:
            word_type = entity_type[word_ori]
            return word2idx(word_type, corpus_entity_composite)
        else:
            return word
    else:
        if word in all_type_entities:
            return entity_type[word]
        else:
            return word

entity_type_folder = "recipe_data_clean/types/"

corpus_ori = load_text_dataset(data_without_types)
corpus_awd_lstm = load_text_dataset(data_without_types)
corpus_type = load_text_dataset(data_with_type)
corpus_entity_composite = load_entity_composite_dataset(data_entity_composite)

EOS_TOKEN = '<eos>'
LogSoftmax = nn.LogSoftmax(dim=1)
Softmax = nn.Softmax(dim=1)
types = [file_name[:-4] for file_name in os.listdir(entity_type_folder)]
type_indexes = list(map(lambda x : word2idx(x, corpus_type), types))


regex = re.compile(".*?\((.*?)\)")
all_type_entities = set()
type_to_entites_dict = {}
entity_type = {}
for file_name in os.listdir(entity_type_folder):
    f = open(entity_type_folder + file_name, "r")
    entities = list(map(lambda x : x.lower().strip(), f.readlines()))
    all_type_entities.update(entities)
    type_to_entites_dict[file_name[:-4]] = entities
    for entity in entities:
        entity_type[entity] = file_name[:-4]

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
    beam_width = 10
    topk = 1
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

def get_next_word_entity_composite(model, corpus, word_with_type, hidden, isIndex=True):
    model_entity_composite, model_type = model
    corpus_entity_composite, corpus_type = corpus
    word, word_type = word_with_type
    hidden_entity_composite, hidden_type = hidden

    #type model prediction
    output_type, new_hidden_type = get_next_word(model_type, corpus_type, word_type, hidden_type, isIndex)
    output_type = output_type.view(-1)

    # entity_composite prediction
    if not isIndex:
        word = word2idx(word, corpus_entity_composite)
        word_type = word2idx(word_type, corpus_entity_composite)
    #  model_entity_composite.eval()
    output_entity_composite, new_hidden_entity_composite = model_entity_composite(torch.LongTensor([word]).view(1, -1), torch.LongTensor([word_type]).view(1, -1), hidden_entity_composite)
    output_entity_composite = Softmax(output_entity_composite).view(-1)

    # combined prediction
    type_prob_sum = {}
    for t in types:
        p = 0.0
        for entity in type_to_entites_dict[t]:
            idx = word2idx(entity, corpus_ori)
            p += output_entity_composite[idx].data
        type_prob_sum[t] = p
    other_prob_sum = 1.0 - sum([prob_sum for _,prob_sum in type_prob_sum.items()])

    probs = []
    all_type_prob = sum([output_type[type_idx].data for type_idx in type_indexes])

    for idx, word in enumerate(corpus_ori.dictionary.idx2word):
        if word in all_type_entities:
            p_type = output_type[get_type(idx, True)].data
            type_of_word = entity_type[word]
            p_entity_composite = output_entity_composite[idx].data/type_prob_sum[type_of_word]
            prob = p_type * p_entity_composite
        else:
            p_entity_composite = output_entity_composite[idx].data/other_prob_sum
            prob = (1 - all_type_prob)*p_entity_composite
        probs.append(prob)
    return Softmax(torch.Tensor(probs).view(1, -1)), (new_hidden_entity_composite, new_hidden_type)


def beam_search_entity_composite(model, corpus, hidden, initial_sentence):
    if model[0].is_attention_model():
        model[0].reset_last_layer()
    if model[1].is_attention_model():
        model[1].reset_last_layer()

    beam_width = 15
    topk = 1
    for word in initial_sentence:
        output, hidden = get_next_word_entity_composite(model, corpus, word, hidden, False)

    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))
    nodes = PriorityQueue()
    qsize = 0

    output = torch.log(output)
    log_prob, indexes = torch.topk(output, beam_width)
    next_nodes = []

    for new_k in range(beam_width):
        decoded_t = indexes[0][new_k].view(1, -1)
        decoded_type_idx = get_type(decoded_t, True)
        log_p = log_prob[0][new_k].item()

        node = BeamSearchNode(hidden, None, torch.LongTensor([decoded_t, decoded_type_idx], device = device), 0, 1)
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
        if new_word[0].data == word2idx(EOS_TOKEN, corpus_ori) and n.prevNode != None:
            endnodes.append((score, n))
            if len(endnodes) >= number_required:
                break
            else:
                continue

        output, hidden = get_next_word_entity_composite(model, corpus, new_word.view(-1), hidden)
        output = torch.log(output)
        log_prob, indexes = torch.topk(output, beam_width)
        next_nodes = []

        for new_k in range(beam_width):
            decoded_t = indexes[0][new_k].view(1, -1)
            decoded_type_idx = get_type(decoded_t, True)
            log_p = log_prob[0][new_k].item()

            node = BeamSearchNode(hidden, n, torch.LongTensor([decoded_t, decoded_type_idx], device = device), n.logp + log_p, n.leng + 1)
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
        utterance.append(n.wordid[0])
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid[0])

        utterance = utterance[::-1]
        utterances.append(utterance)

    return utterances
