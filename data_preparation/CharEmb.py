import os
import pickle

import numpy

from config.Config_worker import Config
from data_preparation.utils import *


class CharEmb:
    def __init__(self, model_type='bilstm'):
        """
        Stata per corpora:
            gicrya: self.sent_max_len = 55  # optimal:55; max: 110;
            gicrya: self.max_token_length = 35;

        network_type: cnn or lstm;
            
        """

        self.config = Config(model_type=model_type)
        self.sent_max_len = self.config.get('Sent_max_length')

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/')
        sent = load_data(self.data_path + '/' + self.config.get('Corpora') + '/test') + \
               load_data(self.data_path + '/' + self.config.get('Corpora') + '/train')

        x_set = seq_form(sent, data_type='x')
        unique_tokens = unique_elements(x_set)
        self.unique_symbols = unique_chars(x_set)
        self.max_token_length = max([len(token) for token in unique_tokens])
        self.word2ind, self.ind2word = self.token_encode(unique_tokens)

        char_embeddings = self.char_matrix(unique_tokens)

        print('vocabulary:', len(unique_tokens))
        print('unique_symbols:', len(self.unique_symbols))
        print('Maximum sequence length:', self.sent_max_len)
        print('Maximum token length:', self.max_token_length)
        print('char embeddings:', char_embeddings.shape)

        self.save_emb(
            ('unique_symbols', self.unique_symbols),
            ('unique_tokens', unique_tokens),
            ('word2index', self.word2ind),
            ('max_sent_length', self.sent_max_len),
            ('char_matrix', char_embeddings)
        )

    def save_emb(self,
                 unique_symbols=None,
                 unique_tokens=None,
                 word2index=None,
                 max_sent_length=None,
                 char_matrix=None
                 ):
        """
        Save data.
        """

        emb_feature_data = dict()
        emb_feature_data[unique_symbols[0]] = unique_symbols[1]
        emb_feature_data[unique_tokens[0]] = unique_tokens[1]
        emb_feature_data[word2index[0]] = word2index[1]
        emb_feature_data[max_sent_length[0]] = max_sent_length[1]
        emb_feature_data[char_matrix[0]] = char_matrix[1]
        save_binary(emb_feature_data,
                    self.data_path + '/%s/' % (self.config.get('Corpora'),) + '/bilstm/char_emb_rnn_feature_data.pkl')

    def token_encode(self, uniq_tokens):
        """
        Create vocabulary.
        :param x_data:
        :return: {'heeft': 0, 'leveranciers': 4112, 'SGR': 1, 'revolutie': 4113, ...}
        """

        return {word: index+1 for index, word in enumerate(uniq_tokens)}, \
               {index+1: word for index, word in enumerate(uniq_tokens)}

    def char_matrix(self, unique_tokens):
        """
        Creating matrix with char embedding.
        :param data:
        :return:
        """

        embed_vocab = list()
        base_vector = numpy.zeros(len(self.unique_symbols) * self.max_token_length)
        embed_vocab.append(base_vector)
        for tokens in unique_tokens:
            features_per_token = numpy.array([], dtype='int8')
            for index_chars in range(0, self.max_token_length):
                array_char = numpy.zeros((len(self.unique_symbols),))
                try:
                    array_char[self.unique_symbols.index(tokens[index_chars])] = 1
                    # print(word[index_chars], array_char)
                except IndexError:
                    pass
                features_per_token = numpy.append(features_per_token, array_char)
            embed_vocab.append(features_per_token)
        return numpy.array(embed_vocab).astype('int8')


if __name__ == '__main__':
    CharEmb(model_type='bilstm')
