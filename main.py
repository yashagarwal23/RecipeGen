import os
import numpy as np
import torch
from rnn_model.build_model import get_model as get_rnn_model, train_and_eval as train_and_eval_rnn_model
from double_input_rnn_model.build_model import get_model as get_double_input_rnn_model, train_and_eval as train_and_eval_double_input_rnn_model
from hierarchical_model.build_model import get_model as get_hierarchical_model, train_and_eval as train_and_eval_hierarchical_model
from data_utils import load_text_corpus, load_hierarchical_corpus, load_text_double_input_corpus
from rnn_model.utils import model_load

data_folder = "recipe_data"
data_hierarchical_path = "recipe_data"

types_folder = os.path.join(data_folder, "types/")

model_folder = "models"
model_type_save_path = os.path.join(model_folder, "model_type.pt")
model_type_to_subtype_save_path = os.path.join(model_folder, "model_type_to_subtype.pt")
model_subtype_to_word_save_path = os.path.join(model_folder, "model_subtype_to_word.pt")
model_hierarchical_save_path = os.path.join(model_folder, 'model_hierarchical.pt')

args = {
            "model_type" : "QRNN",
            "embedding_size" : 400, # word embedding size
            "nhid" : 1500, # size of hidden layer
            "num_layers" : 3, # num layers
            "lr" : 0.001, # learning rate
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

    corpus_hierarchical = load_hierarchical_corpus(data_hierarchical_path)

    #  type model
    corpus_type = load_text_corpus(data_hierarchical_path, 'data_ori', 'data_type', corpus_hierarchical.dictionary)
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
    corpus_type_to_subtype = load_text_double_input_corpus(data_hierarchical_path, 'data_ori', 'data_type', 'data_subtype', corpus_hierarchical.dictionary)
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
    corpus_subtype_to_word = load_text_double_input_corpus(data_hierarchical_path, 'data_ori', 'data_subtype', 'data_ori', corpus_hierarchical.dictionary)
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

    # # hierarchical model
    # model_type, _, _ = get_rnn_model(corpus_hierarchical, args)
    # model_type_to_subtype, criterion_type_to_subtype, params_model_type_to_subtype = get_double_input_rnn_model(corpus_hierarchical, args)
    # model_subtype_to_word, criterion_subtype_to_word, params_model_subtype_to_word = get_double_input_rnn_model(corpus_hierarchical, args)

    model_hierarchical, criterion_hierarchical, params_model_hierarchical = get_hierarchical_model(corpus_hierarchical, model_subtype_to_word, model_type_to_subtype, model_type, args)
    if args["optimizer"] == "sgd":
        optimizer_model_hierarchical = torch.optim.SGD(params_model_hierarchical, lr = args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_model_hierarchical = torch.optim.Adam(params_model_hierarchical, lr = args["lr"], weight_decay=args["wdecay"])

    if not os.path.exists(model_hierarchical_save_path):
        train_and_eval_hierarchical_model(model_hierarchical, corpus_hierarchical, optimizer_model_hierarchical,
                criterion_hierarchical, params_model_hierarchical, args,
                model_hierarchical_save_path)