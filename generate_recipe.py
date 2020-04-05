import torch
from beam_search import corpus_entity_composite, corpus_type, corpus_awd_lstm, corpus_ori
from beam_search import beam_search, beam_search_entity_composite
from beam_search import get_type, word2idx, idx2word
from main import args, model_awd_lstm_save_path, model_type_save_path, model_entity_composite_save_path
from awd_lstm.build_model import get_model as get_rnn_model, model_load
from entity_composite.build_model import get_model as get_entity_composite_model


with torch.no_grad():
    #  model_awd_lstm, _, _ = get_rnn_model(corpus_awd_lstm, args)
    #  model_awd_lstm_state_dict, _, _ = model_load(model_awd_lstm_save_path, "cpu")
    #  model_awd_lstm.load_state_dict(model_awd_lstm_state_dict)

    #  initial_sentence = "preheat the oven . mix flour and water in a bowl .".split(' ')
    #  hidden = model_awd_lstm.init_hidden(1)
    #  search = [idx2word(word.item(), corpus_awd_lstm) for word in beam_search(model_awd_lstm, corpus_awd_lstm, hidden, initial_sentence)[0]]
    #  print(search)


    model_type, _, _ = get_rnn_model(corpus_type, args)
    model_type_state_dict, _, _ = model_load(model_type_save_path, "cpu")
    model_type.load_state_dict(model_type_state_dict)

    initial_sentence = "preheat the oven . mix flour and water in a bowl .".split(' ')
    hidden = model_type.init_hidden(1)
    search = [idx2word(word.item(), corpus_awd_lstm) for word in beam_search(model_type, corpus_awd_lstm, hidden, initial_sentence)[0]]
    print(search)

    #  model_entity_composite, _, _ = get_entity_composite_model(corpus_entity_composite, args)
    #  model_entity_composite_state_dict, _, _ = model_load(model_entity_composite_save_path, "cpu")
    #  model_entity_composite.load_state_dict(model_entity_composite_state_dict)
    #
    #
    #  model = (model_entity_composite, model_type)
    #  hidden = (model_entity_composite.init_hidden(1), model_type.init_hidden(1))
    #  corpus = (corpus_entity_composite, corpus_type)
    #
    #  #  print(get_type("the"))
    #
    #  print("corpus_ori : ", len(corpus_ori.dictionary))
    #  print("corpus_type : ", len(corpus_type.dictionary))
    #  print("corpus_entity_composite : ", len(corpus_entity_composite.dictionary))
    #
    #  initial_sentence = "melt butter over moderate heat in a medium stockpot or dutch oven .".split(' ')
    #  initial_sentence = [(word, get_type(word)) for word in initial_sentence]
    #  search = [idx2word(word.item(), corpus_awd_lstm) for word in beam_search_entity_composite(model, corpus, hidden, initial_sentence)[0]]
    #  print(search)


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
