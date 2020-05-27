import os
import numpy as np
import torch
from rnn_model.build_model import get_model as get_rnn_model, train_and_eval as train_and_eval_rnn_model
from double_input_rnn_model.build_model import get_model as get_double_input_rnn_model, train_and_eval as train_and_eval_double_input_rnn_model
from combined_model.build_model import get_model as get_combined_model, train_and_eval as train_and_eval_combined_model
from data_utils import load_text_corpus, load_text_double_input_corpus, load_combined_corpus

data_folder = "recipe_data"
data_combined_path = "recipe_data"

types_folder = os.path.join(data_folder, "types/")

model_folder = "models"
model_entity_composite_save_path = os.path.join(model_folder, "model_entity_composite.pt")
model_type_save_path = os.path.join(model_folder, "model_type.pt")
model_combined_save_path = os.path.join(model_folder, "model_combined.pt")

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
            "dropout" : 0.4,
            "dropouth" : 0.3, # dropout for rnn layers
            "dropouti" : 0.65,
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


    corpus_combined = load_combined_corpus(data_combined_path)

    # type model
    corpus_type = load_text_corpus(data_combined_path, 'data_ori', 'data_type', corpus_combined.dictionary)
    model_type, criterion_type, params_model_type = get_rnn_model(corpus_type, args)
    
    if args["optimizer"] == "sgd":
        optimizer_model_type = torch.optim.SGD(params_model_type, lr = args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_model_type = torch.optim.Adam(params_model_type, lr = args["lr"], weight_decay=args["wdecay"])

    if not os.path.exists(model_type_save_path):
        train_and_eval_rnn_model(model_type, corpus_type, optimizer_model_type,
                criterion_type, params_model_type, args,
                model_type_save_path)


    # entity composite
    corpus_entity_composite = load_text_double_input_corpus(data_combined_path, 'data_ori', 'data_type', 'data_ori', corpus_combined.dictionary)
    model_entity_composite, criterion_entity_composite, params_entity_composite = get_double_input_rnn_model(corpus_entity_composite, args)
    if args["optimizer"] == "sgd":
        optimizer_model_entity_composite = torch.optim.SGD(params_entity_composite, lr=args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_model_entity_composite = torch.optim.Adam(params_entity_composite, lr=args["lr"], weight_decay=args["wdecay"])

    if not os.path.exists(model_entity_composite_save_path):
        train_and_eval_double_input_rnn_model(model_entity_composite, corpus_entity_composite, optimizer_model_entity_composite,
                criterion_entity_composite, params_entity_composite, args,
                model_entity_composite_save_path)


    # combined
    model_type, criterion_type, params_model_type = get_rnn_model(corpus_combined, args)
    model_entity_composite, criterion_entity_composite, params_entity_composite = get_double_input_rnn_model(corpus_combined, args)
    model_combined, criterion_combined, params_model_combined = get_combined_model(corpus_combined, model_entity_composite, model_type, args)
    if args["optimizer"] == "sgd":
        optimizer_model_combined = torch.optim.SGD(params_model_combined, lr = args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_model_combined = torch.optim.Adam(params_model_combined, lr = args["lr"], weight_decay=args["wdecay"])

    if not os.path.exists(model_combined_save_path):
        train_and_eval_combined_model(model_combined, corpus_combined, optimizer_model_combined,
                criterion_combined, params_model_combined, args,
                model_combined_save_path)