import os
import numpy as np
import torch
from rnn_model.build_model import get_model as get_rnn_model, train_and_eval as train_and_eval_rnn_model
from double_input_rnn_model.build_model import get_model as get_double_input_rnn_model, train_and_eval as train_and_eval_double_input_rnn_model
from hierarchial_model.build_model import get_model as get_hierarchial_model, train_and_eval as train_and_eval_hierarchial_model
from data_utils import load_text_corpus, load_hierarchial_corpus, load_text_double_input_corpus
from rnn_model.utils import model_load

data_folder = "recipe_data"
data_hierarchial_path = "recipe_data"

types_folder = os.path.join(data_folder, "types/")

model_folder = "models"
model_awd_lstm_save_path = os.path.join(model_folder, "model_awd_lstm.pt")
model_type_save_path = os.path.join(model_folder, "model_type.pt")
model_type_to_subtype_save_path = os.path.join(model_folder, "model_type_to_subtype.pt")
model_subtype_to_word_save_path = os.path.join(model_folder, "model_subtype_to_word.pt")
model_hierarchial_save_path = os.path.join(model_folder, 'model_hierarchial.pt')

args = {
            "model_type" : "LSTM",
            "embedding_size" : 400, # word embedding size
            "nhid" : 1500, # size of hidden layer
            "num_layers" : 3, # num layers
            "lr" : 0.01, # learning rate
            "clip" : 0.25, # gradient clipping
            "epochs" : 50,
            "batch_size" : 64, "eval_batch_size" : 64, "test_batch_size" : 64,
            "bptt" : 70, # sequence length
            "dropout" : 0.2,
            "dropouth" : 0.3, # dropout for rnn layers
            "dropouti" : 0.4,
            "dropoute" : 0.1,
            "wdrop" : 0.5,
            "tied" : True,
            "seed" : 1882,
            "nonmono" : 5,
            "cuda" : True,
            "log_interval" : 500,
            "alpha" : 2, "beta" : 1, "wdecay" : 1.2e-6,
            "optimizer" : "adam",
        }


device = 'cuda:0' if args["cuda"] else 'cpu'
if __name__ == '__main__':
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if torch.cuda.is_available():
        if not args["cuda"]:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.set_device(0)
            torch.cuda.manual_seed(args["seed"])


    # baseline awd_lstm
    corpus_awd_lstm = load_text_corpus(data_hierarchial_path, 'data_ori', 'data_ori')
    model_awd_lstm, criterion_awd_lstm, params_model_awd_lstm = get_rnn_model(corpus_awd_lstm, args)
    if args["optimizer"] == "sgd":
        optimizer_model_awd_lstm = torch.optim.SGD(params_model_awd_lstm, lr=args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_model_awd_lstm = torch.optim.Adam(params_model_awd_lstm, lr=args["lr"], weight_decay=args["wdecay"])

    if not os.path.exists(model_awd_lstm_save_path):
        train_and_eval_rnn_model(model_awd_lstm, corpus_awd_lstm, optimizer_model_awd_lstm,
                criterion_awd_lstm, params_model_awd_lstm, args,
                model_awd_lstm_save_path)

    corpus_hierarchial = load_hierarchial_corpus(data_hierarchial_path)

    #  type model
    corpus_type = load_text_corpus(data_hierarchial_path, 'data_ori', 'data_type', corpus_hierarchial.dictionary)
    model_type, criterion_type, params_model_type = get_rnn_model(corpus_type, args)
    # model_type_state_dict, _, _ = model_load(model_type_save_path, device)
    # model_type.load_state_dict(model_type_state_dict)

    if args["optimizer"] == "sgd":
        optimizer_model_type = torch.optim.SGD(params_model_type, lr = args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_model_type = torch.optim.Adam(params_model_type, lr = args["lr"], weight_decay=args["wdecay"])

    if not os.path.exists(model_type_save_path):
        train_and_eval_rnn_model(model_type, corpus_type, optimizer_model_type,
                criterion_type, params_model_type, args,
                model_type_save_path)
    

    # type to subtype
    corpus_type_to_subtype = load_text_double_input_corpus(data_hierarchial_path, 'data_ori', 'data_type', 'data_subtype', corpus_hierarchial.dictionary)
    model_type_to_subtype, criterion_type_to_subtype, params_model_type_to_subtype = get_double_input_rnn_model(corpus_type_to_subtype, args)
    #  model_type_to_subtype_state_dict, _, _ = model_load(model_type_to_subtype_save_path, device)
    #  model_type_to_subtype.load_state_dict(model_type_to_subtype_state_dict)

    if args["optimizer"] == "sgd":
        optimizer_model_type_to_subtype = torch.optim.SGD(params_model_type_to_subtype, lr = args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_model_type_to_subtype = torch.optim.Adam(params_model_type_to_subtype, lr = args["lr"], weight_decay=args["wdecay"])

    if not os.path.exists(model_type_to_subtype_save_path):
        train_and_eval_double_input_rnn_model(model_type_to_subtype, corpus_type_to_subtype, optimizer_model_type_to_subtype,
                criterion_type_to_subtype, params_model_type_to_subtype, args,
                model_type_to_subtype_save_path)


    # subtype to word
    corpus_subtype_to_word = load_text_double_input_corpus(data_hierarchial_path, 'data_ori', 'data_subtype', 'data_ori', corpus_hierarchial.dictionary)
    model_subtype_to_word, criterion_subtype_to_word, params_model_subtype_to_word = get_double_input_rnn_model(corpus_subtype_to_word, args)
    # model_subtype_to_word_state_dict, _, _ = model_load(model_subtype_to_word_save_path, device)
    # model_subtype_to_word.load_state_dict(model_subtype_to_word_state_dict)

    if args["optimizer"] == "sgd":
        optimizer_model_subtype_to_word = torch.optim.SGD(params_model_subtype_to_word, lr = args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_model_subtype_to_word = torch.optim.Adam(params_model_subtype_to_word, lr = args["lr"], weight_decay=args["wdecay"])

    if not os.path.exists(model_subtype_to_word_save_path):
        train_and_eval_double_input_rnn_model(model_subtype_to_word, corpus_subtype_to_word, optimizer_model_subtype_to_word,
                criterion_subtype_to_word, params_model_subtype_to_word, args,
                model_subtype_to_word_save_path)

    # # hierarchial model
    # model_type, _, _ = get_rnn_model(corpus_hierarchial, args)
    # model_type_to_subtype, criterion_type_to_subtype, params_model_type_to_subtype = get_double_input_rnn_model(corpus_hierarchial, args)
    # model_subtype_to_word, criterion_subtype_to_word, params_model_subtype_to_word = get_double_input_rnn_model(corpus_hierarchial, args)

    model_hierarchial, criterion_hierarchial, params_model_hierarchial = get_hierarchial_model(corpus_hierarchial, model_subtype_to_word, model_type_to_subtype, model_type, args)
    if args["optimizer"] == "sgd":
        optimizer_model_hierarchial = torch.optim.SGD(params_model_hierarchial, lr = args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_model_hierarchial = torch.optim.Adam(params_model_hierarchial, lr = args["lr"], weight_decay=args["wdecay"])

    if not os.path.exists(model_hierarchial_save_path):
        train_and_eval_hierarchial_model(model_hierarchial, corpus_hierarchial, optimizer_model_hierarchial,
                criterion_hierarchial, params_model_hierarchial, args,
                model_hierarchial_save_path)