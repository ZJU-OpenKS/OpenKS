import sys
import numpy as np
import logging
import torch
import gensim

logging.basicConfig(level = logging.DEBUG)

class Glove:
    """
    Stores pretrained word embeddings for GloVe.
    """
    def __init__(self, WORDEMB_PATH, dim = 50):
        """
        Load a GloVe pretrained embeddings model.
        WORDEMB_PATH - File Path from which to load the embeddings
        dim - Dimension of expected word embeddings.
        """
        self.fn = WORDEMB_PATH+str(dim)+'d.txt'
        self.dim = dim
        logging.debug("Loading GloVe embeddings from: {} ...".format(self.fn))
        self._load(self.fn)
        logging.debug("Done!")

    def _load(self, fn):
        """
        Load glove embedding from a given filename
        """
        self.word2vec_dict = dict()
        with open(fn, 'r') as fp:
            count = 0
            for line in fp:
                line_list = line.strip().split(' ')
                word = line_list[0]
                vector = []
                for index in range(1, len(line_list)):
                    vector.append(float(line_list[index]))
                vector = torch.FloatTensor(vector)
                self.word2vec_dict[word] = vector / torch.sum(vector)
                count += 1
                if (count % 100000) == 0:
                    print(count, 'done.')

        vector = torch.randn(self.dim)
        self.word2vec_dict['<pad>'] = vector / torch.sum(vector)

class Gensim(object):
    """docstring for Gensim"""
    def __init__(self, WORDEMB_PATH, dim = 50):
        super(Gensim, self).__init__()
        self.fn = WORDEMB_PATH       
        self.dim = dim
        self.word2vec_dict = dict()
        logging.debug("loading word embeddings from: {} ...".format(self.fn))
        wv_model = gensim.models.KeyedVectors.load_word2vec_format(self.fn, binary=True)
        for word in wv_model.vocab:
            self.word2vec_dict[word] = torch.tensor(wv_model.wv[word], dtype=torch.float32)
        assert len(self.word2vec_dict) == len(wv_model.vocab)
        vector = torch.randn(self.dim, dtype=torch.float32)
        self.word2vec_dict['<pad>'] = vector
        vector = torch.randn(self.dim, dtype=torch.float32)
        assert not '<unk>' in self.word2vec_dict
        self.word2vec_dict['<unk>'] = vector

if __name__ == '__main__':
    Gensim('../preprocessing/pubmed-vectors=50.bin')
