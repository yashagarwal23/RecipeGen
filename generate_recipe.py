import torch
from data_utils import load_hierarchial_corpus
from main import args, data_hierarchial_path, model_type_save_path, model_type_to_subtype_save_path, model_subtype_to_word_save_path, model_hierarchial_save_path
from rnn_model.build_model import get_model as get_rnn_model
from double_input_rnn_model.build_model import get_model as get_double_input_rnn_model
from hierarchial_model.build_model import get_model as get_hierarchial_model
from rnn_model.utils import model_load
from beam_search import beam_search, idx2word

device = "cuda:0" if args["cuda"] else "cpu"

with torch.no_grad():
    corpus_hierarchial = load_hierarchial_corpus(data_hierarchial_path)
    model_type, _, _ = get_rnn_model(corpus_hierarchial, args)
    model_type_to_subtype, _, _ = get_double_input_rnn_model(corpus_hierarchial, args)
    model_subtype_to_word, _, _ = get_double_input_rnn_model(corpus_hierarchial, args)

    model_hierarchial, _, _ = get_hierarchial_model(corpus_hierarchial, model_subtype_to_word, model_type_to_subtype, model_type, args)
    model_hierarchial_state_dict, _, _ = model_load(model_hierarchial_save_path, device)
    model_hierarchial.load_state_dict(model_hierarchial_state_dict)

    initial_sentence = 'season chicken with salt and pepper .'
    print()
    print("initial_sentence : ", initial_sentence.replace('_', ' '))
    print()
    initial_sentence = initial_sentence.split()
    hidden_hierarchial = model_hierarchial.init_hidden(1)
    search = ' '.join([idx2word(word, corpus_hierarchial) for word in beam_search(model_hierarchial, corpus_hierarchial, hidden_hierarchial, initial_sentence)[-1]])
    search = search.replace('_', ' ')
    print(search)