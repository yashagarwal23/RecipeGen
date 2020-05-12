import torch
import os
import hashlib
import rnn_model.data as rnn_data
import hierarchial_model.data as hierarchial_data
import double_input_rnn_model.data as double_input_rnn_data


def load_text_corpus(path, input_folder, target_folder, dictionary = None):
    print("Loading rnn text corpus")
    print('input path : ', os.path.join(path, input_folder))
    print('target path : ', os.path.join(path, target_folder))
    encoded_path = 'corpus/corpus.{}.data'.format(hashlib.md5((path+input_folder+target_folder).encode()).hexdigest())
    if os.path.exists(encoded_path):
        print('Loading saved corpus')
        corpus = torch.load(encoded_path)
    else:
        print('preparing corpus')
        corpus = rnn_data.Corpus(path, input_folder, target_folder, dictionary)
        print('Saving corpus')
        torch.save(corpus, encoded_path)
    return corpus

def load_text_double_input_corpus(path, input_folder, input2_folder, target_folder, dictionary = None):
    print("Loading rnn text corpus")
    print('input path : ', os.path.join(path, input_folder))
    print('input2 path : ', os.path.join(path, input2_folder))
    print('target path : ', os.path.join(path, target_folder))
    encoded_path = 'corpus/corpus.{}.data'.format(hashlib.md5((path+input_folder+input2_folder+target_folder).encode()).hexdigest())
    if os.path.exists(encoded_path):
        print('Loading saved corpus')
        corpus_data_with_type = torch.load(encoded_path)
    else:
        print('preparing corpus')
        corpus_data_with_type = double_input_rnn_data.Corpus(path, input_folder, input2_folder, target_folder, dictionary)
        print('Saving corpus')
        torch.save(corpus_data_with_type, encoded_path)
    return corpus_data_with_type

def load_hierarchial_corpus(path, dictionary = None):
    print("Loading Hierarchial text corpus")
    print("path : ", path)
    encoded_path = 'corpus/corpus.{}.data'.format(hashlib.md5(path.encode()).hexdigest())
    if os.path.exists(encoded_path):
        print('Loading saved corpus')
        corpus_hierarchial = torch.load(encoded_path)
    else:
        print('preparing corpus')
        corpus_hierarchial = hierarchial_data.Corpus(path)
        print('Saving corpus')
        torch.save(corpus_hierarchial, encoded_path)
    return corpus_hierarchial