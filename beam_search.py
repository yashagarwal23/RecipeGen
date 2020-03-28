import operator
import os
import re
from main import args, data_without_types, data_with_type, data_entity_composite
from awd_lstm.build_model import get_model as get_rnn_model, model_load
from entity_composite.build_model import get_model as get_entity_composite_model
from data_utils import load_text_dataset, load_entity_composite_dataset
import torch
import torch.nn as nn
from queue import PriorityQueue

def idx2word(idx, corpus):
    return corpus.dictionary.idx2word[idx]

def word2idx(word, corpus):
    return corpus.dictionary.word2idx[word]

def get_type(word, index = False):
    if index:
        if word in all_type_entities:
            word_ori = idx2word(word, corpus_entity_composite)
            word_type = entity_type[word_ori]
            return word2idx(word_type, corpus_type)
        else:
            return word
    else:
        if word in all_type_entities:
            return entity_type[word]
        else:
            return word

corpus_ori = load_text_dataset(data_without_types)
corpus_awd_lstm = load_text_dataset(data_without_types)
corpus_type = load_text_dataset(data_with_type)
corpus_entity_composite = load_entity_composite_dataset(data_entity_composite)

EOS_TOKEN = '<eos>'
LogSoftmax = nn.LogSoftmax(dim=1)
Softmax = nn.Softmax(dim=1)
types = [file_name[:-4] for file_name in os.listdir("superingredients/")]
type_indexes = list(map(lambda x : word2idx(x, corpus_type), types))

regex = re.compile(".*?\((.*?)\)")
all_type_entities = set()
type_to_entites_dict = {}
entity_type = {}
for file_name in os.listdir("superingredients/"):
    f = open("superingredients/" + file_name, "r")
    entities = list(map(lambda x : re.sub("[\(\[].*?[\)\]]", "", x.lower()).strip(), f.readlines()))
    all_type_entities.update(entities)
    type_to_entites_dict[file_name[:-4]] = entities
    for entity in entities:
        entity_type[entity] = file_name[:-4]

device = "gpu" if args["cuda"] else "cpu"

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
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def get_next_word(model, corpus, word, hidden, isIndex = True):
    if not isIndex:
        word = word2idx(word, corpus)
    #  model.eval()
    output, new_hidden = model(torch.LongTensor([word]).view(1, 1), hidden)
    return Softmax(output), new_hidden


def beam_search(model, corpus, hidden, initial_sentence):
    beam_width = 10
    topk = 1
    sentence = []
    for word in initial_sentence:
        try:
            output, hidden = get_next_word(model, corpus, word, hidden, False)
            next_word = torch.argmax(output).data
        except:
            continue
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

        #  output, hidden = model(new_word.view(1, 1), hidden)
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
        word_type = word2idx(word_type, corpus_type)
    #  model_entity_composite.eval()
    output_entity_composite, new_hidden_entity_composite = model_entity_composite(torch.LongTensor([word]).view(1, -1), torch.LongTensor([word_type]).view(1, -1), hidden_entity_composite)
    output_entity_composite = Softmax(output_entity_composite).view(-1)

    # combined prediction
    type_prob_sum = {}
    for t in types:
        p = 0.0
        for entity in type_to_entites_dict[t]:
            try:
                idx = word2idx(entity, corpus_ori)
                p += output_entity_composite[idx].data
            except:
                pass
        type_prob_sum[t] = p
    other_prob_sum = 1.0 - sum([prob_sum for _,prob_sum in type_prob_sum.items()])

    probs = []
    all_type_prob = sum([output_type[type_idx].data for type_idx in type_indexes])

    for idx, word in enumerate(corpus_ori.dictionary.idx2word):
        if word in all_type_entities:
            p_type = output_type[idx].data
            type_of_word = entity_type[word]
            p_entity_composite = output_entity_composite[idx].data/type_prob_sum[type_of_word]
            prob = p_type * p_entity_composite
        else:
            p_entity_composite = output_entity_composite[idx].data/other_prob_sum
            prob = (1 - all_type_prob)*p_entity_composite
        probs.append(prob)
    return Softmax(torch.Tensor(probs).view(1, -1)), (new_hidden_entity_composite, new_hidden_type)


def beam_search_entity_composite(model, corpus, hidden, initial_sentence):
    beam_width = 10
    topk = 1
    #  sentence = []
    for word in initial_sentence:
        output, hidden = get_next_word_entity_composite(model, corpus, word, hidden, False)
        next_word_idx = torch.argmax(output).data
        next_word_type_idx = get_type(next_word_idx, True)

    start_word = torch.LongTensor([next_word_idx, next_word_type_idx], device = device)

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
        if new_word[0].data == word2idx(EOS_TOKEN, corpus_ori) and n.prevNode != None:
            endnodes.append((score, n))
            if len(endnodes) >= number_required:
                break
            else:
                continue

        #  output, hidden = model(new_word.view(1, 1), hidden)
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

with torch.no_grad():
    #  model_awd_lstm, _, _ = get_rnn_model(corpus_awd_lstm, args)
    #  model_awd_lstm_state_dict, _, _ = model_load("model_awd_lstm.pt", "cpu")
    #  model_awd_lstm.load_state_dict(model_awd_lstm_state_dict)

    #  hidden = model_awd_lstm.init_hidden(1)
    #  search = [idx2word(word.item(), corpus_awd_lstm) for word in beam_search(model_awd_lstm, corpus_awd_lstm, hidden, initial_sentence)[0]]
    #  print(search)


    model_type, _, _ = get_rnn_model(corpus_type, args)
    model_type_state_dict, _, _ = model_load("model_type.pt", "cpu")
    model_type.load_state_dict(model_type_state_dict)

    model_entity_composite, _, _ = get_entity_composite_model(corpus_entity_composite, args)
    model_entity_composite_state_dict, _, _ = model_load("model_entity_composite.pt", "cpu")
    model_entity_composite.load_state_dict(model_entity_composite_state_dict)




    model = (model_entity_composite, model_type)
    hidden = (model_entity_composite.init_hidden(1), model_type.init_hidden(1))
    corpus = (corpus_entity_composite, corpus_type)


    initial_sentence = "preheat the oven .".split(' ')
    initial_sentence = [(word, get_type(word)) for word in initial_sentence]
    search = [idx2word(word.item(), corpus_awd_lstm) for word in beam_search_entity_composite(model, corpus, hidden, initial_sentence)[0]]
    print(search)





    # Generate output file

    #  test_file = open("data_without_type/train.txt", "r")
    #  test_recipes = test_file.readlines()[50:100]
    #
    #  output_file = open("generated_recipes.txt", "w")
    #
    #  for recipe in test_recipes:
    #      first_sentence = recipe.split('.')[0] + '.'
    #      hidden = model_awd_lstm.init_hidden(1)
    #      search = [idx2word(word.item(), corpus_awd_lstm) for word in beam_search(model_awd_lstm, corpus_awd_lstm, hidden, first_sentence)[0]]
    #      if search[0] == '<eos>':
    #          search = search[1:]
    #      generated_recipe = " ".join(search)
    #      output_file.write("-"*40 + '\n\n')
    #      output_file.write("INPUT : " + first_sentence + "\n\n")
    #      output_file.write("OUTPUT : " + generated_recipe + '\n')
    #      output_file.write("\n\n")
