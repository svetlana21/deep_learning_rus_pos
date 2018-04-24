import os

import numpy
import gensim

from config.Config_worker import Config
from data_preparation.utils import *


class W2vEmb:
    def __init__(self):

        self.config = Config(model_type='bilstm')

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/')

        sent = load_data(self.data_path + '/' + self.config.get('Corpora') + '/test') + \
               load_data(self.data_path + '/' + self.config.get('Corpora') + '/train')

        self.model = gensim.models.Word2Vec.load_word2vec_format(
            self.data_path + '/' + "w2v_models/mix_corpora_5_10_300_skip_neg.bin", binary=True)
        unique_tokens = self.unique_tokens(sent)
        w2v_embeddings = self.form_emb_vocab(unique_tokens)

        print('vocabulary:', len(unique_tokens))
        print('char embeddings:', w2v_embeddings.shape)

        self.save_emb(
            ('w2v_matrix', w2v_embeddings)
        )

    def form_emb_vocab(self, unique_tokens):
        tokens_with_zero_vector = 0
        embed_vocab = list()
        base_vector = numpy.zeros(300, dtype='int64')
        embed_vocab.append(base_vector)
        for tokens in unique_tokens:
            feature_vector = base_vector
            try:
                feature_vector = self.model[tokens.lower()]
            except KeyError:
                tokens_with_zero_vector += 1
            embed_vocab.append(feature_vector)
        print('tokens_with_zero_vector:', tokens_with_zero_vector)
        return numpy.array(embed_vocab, dtype='int64')

    def save_emb(self, w2v_matrix=None):
        """
        Сохранение данных по признаку.
        :param w2v_matrix:
        :return:
        """

        emb_feature_data = dict()
        emb_feature_data[w2v_matrix[0]] = w2v_matrix[1]
        self.save_binary(emb_feature_data, 'w2v_emb_feature_data')

    def save_binary(self, data, file_name):
        """
        Сохранение данных в бинарном формате.
        :param data:
        :param file_name:
        :return:
        """

        with open(self.data_path + '/%s/' % (self.config.get('Corpora'), ) + '/bilstm/%s.pkl' % (file_name,), 'wb') as file:
            pickle.dump(data, file)

    def unique_tokens(self, data):
        return sorted(set([tokens[1] for sent in data for tokens in sent]))


if __name__ == '__main__':
    W2vEmb()
