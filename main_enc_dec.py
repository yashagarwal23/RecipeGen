import os
import math
import torch
import torch.nn as nn
from EncoderDecoder.data import get_data_loader
from EncoderDecoder import Encoder, get_encoder_decoder_model
from main_combined import data_combined_path
from main import args as lm_args
from rnn_model.build_model import get_model as get_rnn_model
from double_input_rnn_model.build_model import get_model as get_double_input_rnn_model
from combined_model.build_model import get_model as get_combined_model
from data_utils import load_combined_corpus

args = {
            "encoder_model_type" : "GRU",
            "encoder_nhid" : 512,
            "encoder_num_layers" : 2,
            "encoder_bidirectional" : True,
            "ingred_embeddiing_size" : 200, # word embedding size
            "encoder_dropout" : 0.3,
            "decoder_model_type" : "QRNN",
            "decoder_embedding_size" : lm_args['embedding_size'],
            "decoder_nhid" : lm_args['nhid'], # size of hidden layer
            "decoder_num_layers" : lm_args['num_layers'], # num layers
            "decoder_num_stages" : 2,
            "lr" : lm_args['lr'], # learning rate
            "epochs" : lm_args['epochs'],
            "batch_size" : lm_args['batch_size'], 
            "eval_batch_size" : lm_args['eval_batch_size'], 
            "test_batch_size" : lm_args['test_batch_size'],
            "seed" : lm_args['seed'],
            "cuda" : lm_args['cuda'],
            "log_interval" : 500,
            "optimizer" : "adam",
        }

def train(model, data_loader, criterion, optimizer, log_interval):
    batch = 0
    total_loss = 0
    model.train()
    for input, target in data_loader:
        batch += 1
        optimizer.zero_grad()
        output, hidden = model(input, target)
        loss = criterion(output.view(-1, output.size(-1)), target[0].view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.data

        if batch%log_interval == 0:
            curr_loss = total_loss.data/log_interval
            print("batch {}/{}".format(batch, len(data_loader)))
            print("loss : ", curr_loss)
            print("ppl : ", math.exp(curr_loss))
            total_loss = 0

def evaluate(model, data_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for input, target in data_loader:
            output, hidden = model(input, target)
            loss = criterion(output.view(-1, output.size(-1)), target[0].view(-1))
            total_loss += loss.item()
        return total_loss/len(data_loader)

def train_and_eval(model, train_data_loader, valid_dataloader, test_datalaoder, num_epochs, criterion, optimizer, log_interval):
    for e in range(1, num_epochs+1):
        print('-'*89)
        print("epoch : ", e)
        train(model, train_dataloader, criterion, optimizer, log_interval)
        valid_loss = evaluate(model, valid_dataloader)
        print("valid loss : ", valid_loss)
        print('-'*89)
    test_loss = evaluate(model, test_datalaoder)
    print("test loss : ", test_loss)


if __name__ == '__main__':
    recipe_data_path = os.path.join('recipe_data', 'data_with_ingredients')

    train_dataloader , valid_dataloader, test_datalaoder, ingred_vocab_size, recipe_vocab_size= get_data_loader(recipe_data_path, args)

    corpus_combined = load_combined_corpus(data_combined_path)
    model_type, _, _ = get_rnn_model(corpus_combined, lm_args)
    model_entity_composite, _, _ = get_double_input_rnn_model(corpus_combined, lm_args)
    model_combined, _, _ = get_combined_model(corpus_combined, model_entity_composite, model_type, lm_args)

    model = get_encoder_decoder_model(ingred_vocab_size, recipe_vocab_size, model_combined, args)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    train_and_eval(model, train_dataloader , valid_dataloader, test_datalaoder, args['num_epochs'], criterion, optimizer, args['log_interval'])