import os
import torch

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, input_folder, target_folder, dictionary = None):
        if dictionary == None:
            self.dictionary = Dictionary()
        else:
            self.dictionary = dictionary
        self.train_input = self.tokenize(os.path.join(path, input_folder, 'train.txt'))
        self.valid_input = self.tokenize(os.path.join(path, input_folder, 'valid.txt'))
        self.test_input = self.tokenize(os.path.join(path, input_folder, 'test.txt'))
        
        self.train_target = self.tokenize(os.path.join(path, target_folder, 'train.txt'))
        self.valid_target = self.tokenize(os.path.join(path, target_folder, 'valid.txt'))
        self.test_target = self.tokenize(os.path.join(path, target_folder, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids