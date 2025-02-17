import os
import math
import numpy as np
import torch.nn as nn
import torch.nn.utils
import torch
import time
from hierarchical_model.model import hierarchical_model
from rnn_model.utils import batchify, get_batch, repackage_hidden, model_save, model_load
from double_input_rnn_model.build_model import get_model as get_double_input_rnn_model
from rnn_model.build_model import get_model as get_rnn_model

def shuffle(train_data):
    idx = torch.randperm(train_data.size(1))
    return train_data[:, idx]

def get_model(corpus, model_subtype_to_word, model_type_to_subtype, model_type, args, attention_model = False):
    if model_subtype_to_word == None:
        model_subtype_to_word = get_double_input_rnn_model(corpus, args)
    if model_type_to_subtype == None:
        model_type_to_subtype = get_double_input_rnn_model(corpus, args)
    if model_type == None:
        model_type = get_rnn_model(corpus, args)
  
    model = hierarchical_model(model_subtype_to_word, model_type_to_subtype, model_type)
    criterion = nn.CrossEntropyLoss()
    if args["cuda"]:
        model = model.cuda()
        criterion = criterion.cuda()
    params = list(model.parameters()) + list(criterion.parameters())
    return model, criterion, params

def evaluate(model, criterion, args, data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args["model_type"]== 'QRNN':
        model.reset()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args["bptt"]):
        if model.is_attention_model():
            model.reset_last_layer()
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        #  total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        total_loss += len(data) * criterion(output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

def train(model, corpus, optimizer, criterion, params, epoch, args):
    # Turn on training mode which enables dropout.
    if args["model_type"]== 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    train_data = shuffle(batchify(corpus.train, args["batch_size"], args))
    hidden = model.init_hidden(args["batch_size"])
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        if model.is_attention_model():
            model.reset_last_layer()
        bptt = args["bptt"] if np.random.random() < 0.95 else args["bptt"]/ 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args["bptt"]
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(output, targets)
        loss = raw_loss
        # Activiation Regularization
        if args["alpha"]: loss = loss + sum(args["alpha"]* dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args["beta"]: loss = loss + sum(args["beta"]* (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args["clip"]: torch.nn.utils.clip_grad_norm_(params, args["clip"])
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args["log_interval"]== 0 and batch > 0:
            cur_loss = total_loss.item() / args["log_interval"]
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args["bptt"], optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args["log_interval"], cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

        del data, targets, raw_loss


def train_and_eval(model, corpus, optimizer, criterion, params, args, save_path):
    val_data = batchify(corpus.valid, args["eval_batch_size"], args)
    test_data = batchify(corpus.test, args["test_batch_size"], args)

    best_val_loss = []
    stored_loss = 100000000

    try:
        for epoch in range(1, args["epochs"]+ 1):
            epoch_start_time = time.time()
            train(model, corpus, optimizer, criterion, params, epoch, args)
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2 = evaluate(model, criterion, args, val_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                print('-' * 89)

                if val_loss2 < stored_loss:
                    model_save(save_path, model, criterion, optimizer)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

            else:
                val_loss = evaluate(model, criterion, args, val_data, args["eval_batch_size"])
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('-' * 89)

                if val_loss < stored_loss:
                    model_save(save_path, model, criterion, optimizer)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if args["optimizer"] == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                       len(best_val_loss) > args["nonmono"] and val_loss > min(best_val_loss[:-args["nonmono"]])):
                   print('Switching to ASGD')
                   optimizer = torch.optim.ASGD(model.parameters(), lr=args["lr"], t0=0, lambd=0.,
                                                weight_decay=args["wdecay"])

                if "when" in args and epoch in args["when"]:
                    print('Saving model before learning rate decreased')
                    #  model_save('{}.e{}'.format(save_path, epoch))
                    model_save('{}.e{}'.format(save_path, epoch) , model, criterion, optimizer)
                    print('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    if os.path.exists(save_path):
        model_state_dict, criterion, params = model_load(save_path)
        model.load_state_dict(model_state_dict)
        # Run on test data.
        test_loss = evaluate(model, criterion, args, test_data, args["test_batch_size"])
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
            test_loss, math.exp(test_loss), test_loss / math.log(2)))
        print('=' * 89)
