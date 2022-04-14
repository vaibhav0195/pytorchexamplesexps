import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def save(self,path):
        import json
        with open(path+"/idx2word.json", 'w') as f:
            # indent=2 is not needed but makes the file human-readable
            # if the data is nested
            json.dump(self.idx2word, f, indent=2)

    def load(self,path):
        import json
        with open(path, 'r') as f:
            self.idx2word = json.load(f)

        for idx,word in enumerate(self.idx2word):
            self.word2idx[word] = idx

class Corpus(object):
    def __init__(self, path,eval=None):
        if eval is None:
            self.dictionary = Dictionary()
            self.train = self.tokenize(os.path.join(path, 'traindata.txt'))
            self.valid = self.tokenize(os.path.join(path, 'validsubset.txt'))
        else:
            self.dictionary = Dictionary()
            self.dictionary.load(eval)
        # self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
