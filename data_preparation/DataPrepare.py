"""
Module for creating x train and test data.
"""
import pickle
import os

from keras.preprocessing.sequence import pad_sequences
import numpy as np

from config.Config_worker import Config
from data_preparation.utils import *


class DataPrepare:
    """
    Class for creating x train and test data.
    """
    def __init__(self):

        self.config = Config(model_type='bilstm')
        self.sent_max_len = self.config.get('Sent_max_length')
        self.corpora_limit = self.config.get('Corpora_sent_limit')

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/')

        char_emb_feature = load_bin_data(
            self.data_path + '/' + self.config.get('Corpora') + '/bilstm/char_emb_rnn_feature_data.pkl')
        self.word2ind = char_emb_feature['word2index']

        self.preparator(data_name='train')
        self.preparator(data_name='test')

    def preparator(self, data_name='test'):
        """
        Prepare data: encoding, padding.
        """

        if self.corpora_limit != 'False':
            sent = load_data(self.data_path + '/' + self.config.get('Corpora') + '/%s' % (data_name, ))[
                   :self.corpora_limit]
        else:
            sent = load_data(self.data_path + '/' + self.config.get('Corpora') + '/%s' % (data_name,))
        x_data = seq_form(sent, data_type='x')
        X_data = self.data_prepare(x_data, data_name)
        del (x_data, sent)
        print('X_%s' % (data_name, ), X_data.dtype)
        save_binary(X_data,
                    self.data_path + '/%s/' % (self.config.get('Corpora'),) + '/bilstm/x_%s.pkl' % (data_name,))

    def data_prepare(self, x_set, name):
        """
        Подготовка данных.
        :param x:
        :param y:
        :return:
        """

        x_enc = [[self.word2ind[c] for c in x] for x in x_set]
        x_train = pad_sequences(x_enc, maxlen=self.sent_max_len)
        print('\nTraining tensor shapes:')
        print('x_%s_forward: %s;' % (name, x_train.shape,))
        return x_train


if __name__ == '__main__':
    DataPrepare()
