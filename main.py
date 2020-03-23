import os
import numpy as np
import torch
from data_utils import load_text_dataset
from awd_lstm.build_model import get_model as get_rnn_model, train_and_eval as train_and_eval_rnn_model

data_without_types = "./data_without_type/"
data_with_type = "./data_with_type"
data_entity_compostite = "./data_entity_composite/"

OPTIMIZER = "sgd"

args = {
            "model_type" : "LSTM",
            "embedding_size" : 100, # word embedding size
            "nhid" : 500, # sie of hidden layer
            "num_layers" : 3, # num layers
            "lr" : 2, # learning rate
            "clip" : 0.25, # gradient clipping
            "epochs" : 10,
            "batch_size" : 128, "eval_batch_size" : 10, "test_batch_size" : 1,
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
            "log_interval" : 200,
            "alpha" : 2, "beta" : 1, "wdecay" : 1.2e-6
        }

args["cuda"] = False
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
if OPTIMIZER == "sgd":
    optimizer_model_awd_lstm = torch.optim.SGD(params_model_awd_lstm, lr=args["lr"], weight_decay=args["wdecay"])
else:
    optimizer_model_awd_lstm = torch.optim.Adam(params_model_awd_lstm, lr=args["lr"], weight_decay=args["wdecay"])

if not os.path.exists('model_awd_lstm.pt'):
    train_and_eval_rnn_model(model_awd_lstm, corpus_awd_lstm, optimizer_model_awd_lstm,
            criterion_awd_lstm, params_model_awd_lstm, args,
            'model_awd_lstm.pt')
