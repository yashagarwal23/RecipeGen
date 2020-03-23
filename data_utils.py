import torch
import os
import hashlib
import awd_lstm.data as data
import entity_composite.data_entity_composite as data_entity_composite


def load_text_dataset(path):
    encoded_path = 'corpus.{}.data'.format(hashlib.md5(path.encode()).hexdigest())
    if os.path.exists(encoded_path):
        print('Loading cached dataset of raw data with types...')
        corpus = torch.load(encoded_path)
    else:
        print('Producing dataset of raw data with types...')
        corpus = data.Corpus(path)
        torch.save(corpus, encoded_path)
    return corpus

def load_entity_composite_dataset(path):
    encoded_path = 'corpus.{}.data'.format(hashlib.md5(path.encode()).hexdigest())
    if os.path.exists(encoded_path):
        print('Loading cached dataset of raw data for entity composite...')
        corpus_EC = torch.load(encoded_path)
    else:
        print('Producing dataset of raw data for entity composite...')
        corpus_EC = data_entity_composite.Corpus(path)
        torch.save(corpus_EC, encoded_path)
    return corpus_EC
