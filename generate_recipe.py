import torch
from data_utils import load_hierarchical_corpus
from main import args, data_hierarchical_path, model_type_save_path, model_type_to_subtype_save_path, model_subtype_to_word_save_path, model_hierarchical_save_path
from rnn_model.build_model import get_model as get_rnn_model
from double_input_rnn_model.build_model import get_model as get_double_input_rnn_model
from hierarchical_model.build_model import get_model as get_hierarchical_model
from rnn_model.utils import model_load
from beam_search import beam_search, idx2word

device = "cuda:0" if args["cuda"] else "cpu"

with torch.no_grad():
    corpus_hierarchical = load_hierarchical_corpus(data_hierarchical_path)
    model_type, _, _ = get_rnn_model(corpus_hierarchical, args)
    model_type_to_subtype, _, _ = get_double_input_rnn_model(corpus_hierarchical, args)
    model_subtype_to_word, _, _ = get_double_input_rnn_model(corpus_hierarchical, args)

    model_hierarchical, _, _ = get_hierarchical_model(corpus_hierarchical, model_subtype_to_word, model_type_to_subtype, model_type, args)
    model_hierarchical_state_dict, _, _ = model_load(model_hierarchical_save_path, device)
    model_hierarchical.load_state_dict(model_hierarchical_state_dict)

    initial_sentence = 'season chicken with salt and pepper .'
    print()
    print("initial_sentence : ", initial_sentence.replace('_', ' '))
    print()
    initial_sentence = initial_sentence.split()
    hidden_hierarchical = model_hierarchical.init_hidden(1)
    search = ' '.join([idx2word(word, corpus_hierarchical) for word in beam_search(model_hierarchical, corpus_hierarchical, hidden_hierarchical, initial_sentence)[-1]])
    search = search.replace('_', ' ')
    print(search)