import os
import torch
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)

def make_dictionary(data):
    dictionary = Dictionary()
    dictionary.add_word('<pad>')
    dictionary.add_word('<sos>')
    dictionary.add_word('<eos>')
    for word in data:
        dictionary.add_word(word)
    return dictionary

class RecipeData(Dataset):
    def __init__(self, df, ingrd_dictionary, recipe_dictionary):
        self.df = df
        # self.max_ing = max_ing
        self.ingrd_dictionary = ingrd_dictionary
        self.recipe_dictionary = recipe_dictionary

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        ingrds = self.df.iloc[idx]['ingredients'].split()
        ingrds_len = torch.LongTensor([len(ingrds)])
        recipe = self.df.iloc[idx]['recipe'].split()

        ingrd_idx = [self.ingrd_dictionary.word2idx[ingrd] for ingrd in ingrds]
        recipe_idx = [self.recipe_dictionary.word2idx[word] for word in ['<sos>']+recipe+['<eos>']]

        ingrds_labels = torch.LongTensor(ingrd_idx)
        ingrds_len = torch.LongTensor([len(ingrds_labels)])

        recipe_labels = torch.LongTensor(recipe_idx)   
        recipe_len = torch.LongTensor([len(recipe_labels)])

        return ingrds_labels, ingrds_len, recipe_labels, recipe_len, self.ingrd_dictionary.word2idx['<pad>'] , self.recipe_dictionary.word2idx['<pad>']


def collate_fn(data):
    ingrds_labels, ingrds_len, recipe_labels, recipe_len, ingrd_pad_value, recipe_pad_value = zip(*data)

    input_lengths = torch.stack(ingrds_len, 0).squeeze(1)
    lengths = list(map(len, ingrds_labels))
    inputs = torch.ones(len(ingrds_labels), max(lengths)).long()*ingrd_pad_value[0]
    for i, ingrd in enumerate(ingrds_labels):
        end = lengths[i]
        inputs[i, :end] = ingrd[:end]
    input_masks = (inputs != ingrd_pad_value[0]).unsqueeze(1)

    target_lengths = torch.stack(recipe_len, 0).squeeze(1)
    lengths = list(map(len, recipe_labels))
    targets = torch.ones(len(recipe_labels), max(lengths)).long()*recipe_pad_value[0]
    for i, recipe in enumerate(recipe_labels):
        end = lengths[i]
        targets[i, :end] = recipe[:end]
    target_masks = (targets != recipe_pad_value[0]).unsqueeze(1)
    return (inputs, input_lengths, input_masks), (targets, target_lengths, target_masks)

def get_data_loader(path, args):
    train_df = pd.read_csv(os.path.join(path, 'train.tsv'), delimiter='\t')
    valid_df = pd.read_csv(os.path.join(path, 'valid.tsv'), delimiter='\t')
    test_df = pd.read_csv(os.path.join(path, 'test.tsv'), delimiter='\t')

    ingr_dict_path = os.path.join('corpus', 'ingredient_dictionary.pkl')
    if not os.path.exists(ingr_dict_path):
        ingr_list = train_df['ingredients'].tolist() + valid_df['ingredients'].tolist() + test_df['ingredients'].tolist()
        flat_ingr_list = []
        for ingrs in ingr_list:
            flat_ingr_list.extend(ingrs.strip().split())
        ingrd_dictionary = make_dictionary(flat_ingr_list)
        pickle.dump(ingrd_dictionary, open(ingr_dict_path, 'wb'))
    else:
        ingrd_dictionary = pickle.load(open(ingr_dict_path, 'rb'))

    ingrd_vocab_size = len(ingrd_dictionary)

    recipe_dict_path = os.path.join('corpus', 'recipe_dictionary.pkl')
    if not os.path.exists(recipe_dict_path):
        recipe_list = train_df['recipe'].tolist() + valid_df['recipe'].tolist() + test_df['recipe'].tolist()
        flat_recipe_list = []
        for recipe in recipe_list:
            flat_recipe_list.extend(recipe.strip().split())
        recipe_dictionary = make_dictionary(flat_recipe_list)
        pickle.dump(recipe_dictionary, open(recipe_dict_path, 'wb'))
    else:
        recipe_dictionary = pickle.load(open(recipe_dict_path, 'rb'))

    recipe_vocab_size = len(recipe_dictionary)

    train_dataset = RecipeData(train_df, ingrd_dictionary, recipe_dictionary)
    valid_dataset = RecipeData(valid_df, ingrd_dictionary, recipe_dictionary)
    test_dataset = RecipeData(test_df, ingrd_dictionary, recipe_dictionary)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'],
                                 shuffle=True, num_workers=4, drop_last=True,
                                 collate_fn=collate_fn, pin_memory=True)

    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args['eval_batch_size'],
                                 shuffle=True, num_workers=4, drop_last=True,
                                 collate_fn=collate_fn, pin_memory=True)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args['test_batch_size'],
                                 shuffle=True, num_workers=4, drop_last=True,
                                 collate_fn=collate_fn, pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader, ingrd_vocab_size, recipe_vocab_size                                