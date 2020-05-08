import os
import numpy as np
import torch
from data_utils import load_text_dataset, load_entity_composite_dataset
from awd_lstm.build_model import get_model as get_rnn_model, train_and_eval as train_and_eval_rnn_model
from entity_composite.build_model import get_model as get_entity_composite_model, train_and_eval as train_and_eval_entity_composite_model
from combined_model.build_model import get_model as get_combined_model, train_and_eval as train_and_eval_combined_model
from awd_lstm.utils import model_load

data_folder = "recipe_data"
data_without_types = os.path.join(data_folder, "data_without_type/")
data_with_type = os.path.join(data_folder, "data_with_type/")
data_entity_composite = os.path.join(data_folder, "data_entity_composite/")

types_folder = os.path.join(data_folder, "types/")

model_folder = "models"
model_awd_lstm_save_path = os.path.join(model_folder, "model_awd_lstm.pt")
model_type_save_path = os.path.join(model_folder, "model_type.pt")
model_entity_composite_save_path = os.path.join(model_folder, "model_entity_composite.pt")
model_combined_save_path = os.path.join(model_folder, "model_combined.pt")

args = {
            "model_type" : "LSTM",
            "embedding_size" : 400, # word embedding size
            "nhid" : 2500, # size of hidden layer
            "num_layers" : 3, # num layers
            "lr" : 2, # learning rate
            "clip" : 0.25, # gradient clipping
            "epochs" : 10,
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
            "log_interval" : 400,
            "alpha" : 2, "beta" : 1, "wdecay" : 1.2e-6,
            "optimizer" : "sgd",
        }

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
    corpus_awd_lstm = load_text_dataset(data_without_types)
    model_awd_lstm, criterion_awd_lstm, params_model_awd_lstm = get_rnn_model(corpus_awd_lstm, args)
    if args["optimizer"] == "sgd":
        optimizer_model_awd_lstm = torch.optim.SGD(params_model_awd_lstm, lr=args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_model_awd_lstm = torch.optim.Adam(params_model_awd_lstm, lr=args["lr"], weight_decay=args["wdecay"])

    if not os.path.exists(model_awd_lstm_save_path):
        train_and_eval_rnn_model(model_awd_lstm, corpus_awd_lstm, optimizer_model_awd_lstm,
                criterion_awd_lstm, params_model_awd_lstm, args,
                model_awd_lstm_save_path)


    # type model
    corpus_type = load_text_dataset(data_with_type)
    model_type, criterion_type, params_model_type = get_rnn_model(corpus_type, args)
    if args["optimizer"] == "sgd":
        optimizer_model_type = torch.optim.SGD(params_model_type, lr = args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_model_type = torch.optim.Adam(params_model_type, lr = args["lr"], weight_decay=args["wdecay"])

    if not os.path.exists(model_type_save_path):
        train_and_eval_rnn_model(model_type, corpus_type, optimizer_model_type,
                criterion_type, params_model_type, args,
                model_type_save_path)


    # entity composite model
    corpus_entity_composite = load_entity_composite_dataset(data_entity_composite)
    model_entity_composite, criterion_entity_composite, params_model_entity_composite = get_entity_composite_model(corpus_entity_composite, args)
    if args["optimizer"] == "sgd":
        optimizer_entity_composite = torch.optim.SGD(params_model_entity_composite, lr=args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_entity_composite = torch.optim.Adam(params_model_entity_composite, lr=args["lr"], weight_decay=args["wdecay"])

    if not os.path.exists(model_entity_composite_save_path):
        train_and_eval_entity_composite_model(model_entity_composite, corpus_entity_composite, optimizer_entity_composite,
                criterion_entity_composite, params_model_entity_composite, args,
                model_entity_composite_save_path)

    # combined
    corpus_combined = corpus_entity_composite, corpus_type
    model_combined, criterion_combined, params_model_combined = get_combined_model(model_entity_composite, model_type, corpus_combined, args)
    if args["optimizer"] == "sgd":
        optimizer_combined = torch.optim.SGD(params_model_combined, lr=args["lr"], weight_decay=args["wdecay"])
    else:
        optimizer_combined = torch.optim.Adam(params_model_combined, lr=args["lr"], weight_decay=args["wdecay"])

    model_type_state_dict, _, _ = model_load(model_type_save_path, "cpu")
    model_type.load_state_dict(model_type_state_dict)
    for p in model_type.parameters():
        if p.requires_grad:
            p.requires_grad = False

    if not os.path.exists(model_combined_save_path):
        train_and_eval_combined_model(model_combined, corpus_combined, optimizer_combined,
                criterion_combined, params_model_combined, args,
                model_combined_save_path)
