import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(1024)
rn.seed(1024)
tf.set_random_seed(1024)

import pickle
from collections import OrderedDict
from config.Config_worker import Config
from keras.models import *

import gc
gc.collect()


class CNNProbEmbeddings:
    def __init__(self, use_config=True, corpora='UD_Russian-SynTagRus', task_type='POS', class_index=3, dev=True,
                 verbose=1, prob_cnn_emb_layer_name="dense_3"):

        if use_config:
            self.config = Config(model_type='cnn')
            self.task_type = self.config.get('Task_type')
            self.class_index = self.config.get('Classification_tasks')['UD2']['POS'][0]
            self.corpora = self.config.get('Corpora')
            self.prob_cnn_emb_layer_name = self.config.get('Network_options').get('prob_cnn_emb_layer_name')
        else:
            self.task_type = task_type
            self.corpora = corpora
            self.class_index = class_index
            self.prob_cnn_emb_layer_name = prob_cnn_emb_layer_name

        print('CNNProbEmbeddings for 2 level model.')
        print('Task:', self.task_type)
        print('Corpora:', self.corpora)
        print('Label index:', class_index)

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/%s/cnn/model_level_1/' % (self.corpora,))
        self.model_path = os.path.abspath(file_path + '/../tagger_models/')

        char_emb_feature = self.load_binary_data(self.data_path + '/char_emb_cnn1_feature_data_%s.pkl' % (self.task_type, ))

        self.ind2symbol = char_emb_feature['ind2symbol']
        self.max_token_length = char_emb_feature['max_token_length']
        self.x_train = self.load_binary_data(self.data_path + '/x_train_cnn1level.pkl')
        self.x_test = self.load_binary_data(self.data_path + '/x_test_cnn1level.pkl')
        if dev:
            self.x_dev = self.load_binary_data(self.data_path + '/x_dev_cnn1level.pkl')

        if verbose == 1:
            print("Loading char_emb_cnn1_feature_data_%s ..." % (self.task_type,))
            print('x_train shape:', self.x_train.shape)
            print('x_test shape:', self.x_test.shape)
            print('x_dev shape:', self.x_dev.shape)

        ################################################################################################################

        str2vector = {}
        for el_ in self.x_train:
            str2vector[''.join([self.ind2symbol[s] for s in el_ if s != 0])] = el_
        for _el in self.x_test:
            str2vector[''.join([self.ind2symbol[s] for s in _el if s != 0])] = _el
        if dev:
            for el in self.x_dev:
                str2vector[''.join([self.ind2symbol[s] for s in el if s != 0])] = el
        str2vector = OrderedDict(str2vector)
        if verbose == 1:
            print("Unique_tokens:", len(str2vector))

        ################################################################################################################

        null_word = [0 for i in range(self.max_token_length)]
        null_vector = np.array(null_word)

        str2vector.update({'_null_': null_vector})
        str2vector.move_to_end('_null_', last=False)
        str2vector = [(el, str2vector[el]) for el in str2vector]
        if verbose == 1:
            print("Checking null word:", str2vector[0])

        ################################################################################################################

        non_tuned_embeddings = np.array([el[1] for el in str2vector])
        if verbose == 1:
            print("Non tune embeddings:", non_tuned_embeddings.shape)

        ################################################################################################################

        if verbose == 1:
            print('Loading cnn_1level_model_%s_%s.pkl' % (self.corpora, self.task_type,))
        self.model = load_model(
            self.model_path + '/cnn_1level_model_%s_%s.pkl' % (self.corpora, self.task_type,))

        activation_values = self.get_prob_from_layer(
            layer_name=self.prob_cnn_emb_layer_name,
            data=non_tuned_embeddings)
        if verbose == 1:
            print('activity_values_train shape:', activation_values.shape)

        ################################################################################################################

        if self.task_type == "All":
            model_pos = load_model(
               self.model_path + '/cnn_1level_model_%s_%s.pkl' % (self.corpora, "POS",))
            pr = model_pos.predict(non_tuned_embeddings, verbose=1)
            activation_values = np.concatenate((activation_values, pr), axis=1)
            if verbose == 1:
                print("Predictons shape:", pr.shape)
                print("Predictons shape + activations:", activation_values.shape)
                # TODO проверить предсказание по _null_

        ################################################################################################################

        result_train = OrderedDict(list(zip([el[0] for el in str2vector], activation_values)))
        if verbose == 1:
            print("Checking null word:", len(result_train['_null_']))
        self.save_binary(result_train, '_%s_%s' % (self.prob_cnn_emb_layer_name, self.task_type))

        if dev:
            del (result_train, activation_values, self.model, str2vector, null_vector, null_word, char_emb_feature,
                 self.ind2symbol, self.max_token_length, self.x_train, self.x_test, self.x_dev, non_tuned_embeddings)
        else:
            del (result_train, activation_values, self.model, str2vector, null_vector, null_word, char_emb_feature,
                 self.ind2symbol, self.max_token_length, self.x_train, self.x_test, non_tuned_embeddings)

    def load_binary_data(self, path_to_data):
        """
        Load data
        :return:
        """

        with open(path_to_data, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_prob_from_layer(self, layer_name=None, data=None):
        """
        The output of an intermediate layer.
        :param layer_name: 
        :return: 
        """

        intermediate_layer_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(data)
        return intermediate_output

    def load_data(self, path_to_data):
        """
        Data loader.
        :param path_to_data:
        :return:
        """

        # print('Loading:', path_to_data, '\n')
        raw = open(path_to_data, 'r').readlines()
        all_x = []
        point = []
        for line in raw:
            stripped_line = line.strip().split('\t')
            point.append(stripped_line)
            if line == '\n':
                all_x.append(point[:-1])
                point = []
        all_x = all_x[:-1]
        return all_x

    def save_binary(self, data, file_name):
        """
        Сохранение данных в бинарном формате.
        :param data:
        :param file_name:
        :return:
        """

        with open(self.data_path + '/cnn_prob_emb%s.pkl' % (file_name, ), 'wb') as file:
            pickle.dump(data, file)
