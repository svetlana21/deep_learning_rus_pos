import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(1024)
rn.seed(1024)
tf.set_random_seed(1024)

import pickle
import math
from config.Config_worker import Config
from keras.models import *
from keras.layers import *
from keras.utils import to_categorical
from keras.callbacks import *
import matplotlib.pyplot as plt

import gc
gc.collect()


class CNN2levelTagger:
    """
    Mean of option "Task_type". Grammem: "Grammem_tag_Animacy", POS, All (all morphology properties.).
    """

    def __init__(self,
                 use_config=True,
                 corpora='UD_Russian-SynTagRus',
                 task_type='POS',
                 class_index=3,
                 verbose=1,
                 batch_size=512,
                 epoch=300,
                 dev=False):

        if use_config:
            self.config = Config(model_type='cnn')
            self.task_type = self.config.get('Task_type')
            self.class_index = self.config.get('Classification_tasks')['UD2']['POS'][0]
            self.corpora = self.config.get('Corpora')
            self.batch_size_level_2 = self.config.get('Network_options').get('batch_size_level_2')
            self.epoch = self.config.get('Network_options').get('training_epoch')
        else:
            self.task_type = task_type
            self.corpora = corpora
            self.class_index = class_index
            self.batch_size_level_2 = batch_size
            self.epoch = epoch

        print('\nModel train for 2 level model.')
        print('Task:', self.task_type)
        print('Corpora:', self.corpora)
        print('Label index:', class_index)

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/%s/cnn/model_level_2/' % (self.corpora,))
        self.model_path = os.path.abspath(file_path + '/../tagger_models/')

        self.tune_char_emb_matrix = self.load_binary_data(self.data_path + '/char_emb_cnn2_feature_data_%s.pkl' % (self.task_type, ))
        self.sent_max_len = self.tune_char_emb_matrix['max_sent_length']

        self.x_train = self.load_binary_data(self.data_path + '/x_train_cnn2level_%s.pkl' % (self.task_type,))
        self.y_train, self.out_size = self.load_grammatical_cat(verbose=verbose)
        if verbose == 1:
            print('x_train shape:', self.x_train.shape)
            print('y_train shape:', self.y_train.shape)

        if dev:
            self.x_dev = self.load_binary_data(self.data_path + '/x_dev_cnn2level_%s.pkl' % (self.task_type,))
            self.y_dev, _ = self.load_grammatical_cat(y_data_name='dev', verbose=verbose)
            if verbose == 1:
                print('x_dev shape:', self.x_dev.shape)
                print('y_dev shape:', self.y_dev.shape)

        self.max_features = len(self.tune_char_emb_matrix['word2ind']) + 2
        self.data_for_emb_layers = {'tune_char_emb_matrix': self.tune_char_emb_matrix['tune_char_emb_matrix']}

        self.num_batches_per_epoch_train = math.ceil(self.x_train.shape[0] / self.batch_size_level_2)
        if dev:
            self.num_batches_per_epoch_valid = math.ceil(self.x_dev.shape[0] / self.batch_size_level_2)

        if verbose == 1:
            print("num_batches_per_epoch_train:", self.num_batches_per_epoch_train)
            print("num_batches_per_epoch_valid:", self.num_batches_per_epoch_valid)

        self.model = None

    def load_grammatical_cat(self, y_data_name='train', verbose=1):
        """
        Loading y data for each grammatical category.
        """

        labels = None
        labels_2indexes = None

        for files in os.listdir(self.data_path):
            if self.task_type == 'All':
                if 'y_%s_cnn2level_All' % (y_data_name,) in files:
                    labels = self.load_binary_data(self.data_path + '/' + files)
                if 'y_label2ind_cnn2level_All' in files:
                    labels_2indexes = len(self.load_binary_data(self.data_path + '/' + files))
            else:
                if 'y_%s_cnn2level_%s' % (y_data_name, self.task_type) in files:
                    labels = self.load_binary_data(self.data_path + '/' + files)
                if 'y_label2ind_cnn2level_%s' % (self.task_type, ) in files:
                    labels_2indexes = len(self.load_binary_data(self.data_path + '/' + files))
        if verbose == 1:
            print('y_data:', labels.shape)
        return labels, labels_2indexes

    def __to_categorical(self, data):
        return np.array([[to_categorical(t, self.out_size+2)[0] for t in s] for s in data])

    def data_generator(self, batch_size, x, y, num_batches_per_epoch):
        """
        A generator or an instance of Sequence (keras.utils.Sequence) object in order to avoid duplicate data when
        using multiprocessing. The output of the generator must be either a tuple (inputs, targets)
        :return:
        """

        while True:
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = (batch_num + 1) * batch_size
                x_batch = x[start_index:end_index]
                y_batch = self.__to_categorical(y[start_index:end_index])
                if x_batch.shape[0] != 0:
                  yield x_batch, y_batch

    def load_binary_data(self, path_to_data):
        """
        Load data
        :return:
        """

        with open(path_to_data, 'rb') as f:
            data = pickle.load(f)
        return data

    def network_initialization(self, verbose=1):
        """
        Network compilation.            
        :return:
        """

        ###################################################Network_level_2##############################################

        seq_input = Input((self.sent_max_len,))
        seq_emb = Embedding(
            input_dim=self.max_features,
            output_dim=self.data_for_emb_layers.get('tune_char_emb_matrix').shape[1],
            input_length=self.sent_max_len,
            weights=[self.data_for_emb_layers.get('tune_char_emb_matrix')],
            # https://github.com/fchollet/keras/issues/3335
            # https://groups.google.com/forum/#!topic/keras-users/KfoTsCHldM4
            mask_zero=False,
            trainable=True
        )(seq_input)
        seq_dropout = Dropout(0.5)(seq_emb)

        seq_conv1d_0 = Conv1D(
            filters=256,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='conv1d_7',
        )(seq_dropout)

        seq_conv1d_1 = Conv1D(
            filters=256,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='conv1d_8',
        )(seq_conv1d_0)

        seq_conv1d_2 = Conv1D(
            # the PoS-classes number + 1 (for the zero padding)
            filters=self.out_size+2,
            kernel_size=3,
            padding='same',
            activation='softmax',
            name='conv1d_9',
        )(seq_conv1d_1)

        model = Model(inputs=seq_input, outputs=seq_conv1d_2)
        model.compile(optimizer='adamax', loss="mean_squared_error", metrics=["accuracy"])
        self.model = model
        print(self.model.summary())

        # plot_model(model, to_file=self.model_path + '/cnn_2level_model_schema_%s_%s.png' %
        #                                             (self.config.get('Corpora'),
        #                                              self.task_type,), show_shapes=True)
        if verbose == 1:
            print(model.summary())
            print('Model comp done.')

    def training(self, verbose=1, dev=True):
        """
        Training.

        model_checkpoint
            For `val_acc`, this should be `max`, for `val_loss` this should  be `min`, etc. In `auto` mode, the
            direction is automatically inferred from the name of the monitored quantity.

        early_stopping
                In `min` mode, training will stop when the quantity monitored has stopped decreasing; in `max` mode it
                will stop when the quantity  monitored has stopped increasing; in `auto` mode, the direction is
                automatically inferred from the name of the monitored quantity.

        :return:
        """

        if self.task_type == "All":
            __save_best_only = False
        else:
            __save_best_only = True

        model_checkpoint = ModelCheckpoint(filepath=os.path.join(self.model_path + '/cnn_2level_model_%s_%s.pkl' %
                                                                 (self.corpora,
                                                                  self.task_type,)),
                                           monitor='val_loss',
                                           verbose=0,
                                           save_weights_only=False,
                                           save_best_only=__save_best_only,
                                           mode='auto')

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=10,
                                       verbose=0,
                                       mode='auto')

        if dev:
            self.model.fit_generator(
                generator=self.data_generator(
                    batch_size=self.batch_size_level_2,
                    x=self.x_train,
                    y=self.y_train,
                    num_batches_per_epoch=self.num_batches_per_epoch_train),
                steps_per_epoch=self.num_batches_per_epoch_train,
                epochs=self.epoch,
                validation_data=self.data_generator(
                    batch_size=self.batch_size_level_2,
                    x=self.x_dev,
                    y=self.y_dev,
                    num_batches_per_epoch=self.num_batches_per_epoch_valid),
                validation_steps=self.num_batches_per_epoch_valid,
                verbose=2,
                shuffle=True,
                callbacks=[model_checkpoint, early_stopping],
                workers=1,
                use_multiprocessing=False
            )

            del (self.x_train, self.y_train, self.out_size, self.x_dev, self.y_dev, self.tune_char_emb_matrix,
                 self.sent_max_len, self.max_features, self.data_for_emb_layers, self.model)

        else:
            self.model.fit(
                x=self.x_train,
                y=self.__to_categorical(self.y_train),
                batch_size=self.batch_size_level_2,
                epochs=self.epoch,
                validation_split=0.1,
                verbose=2,
                shuffle=True,
                callbacks=[model_checkpoint, early_stopping]
            )

            del (self.x_train, self.y_train, self.out_size, self.tune_char_emb_matrix,
                 self.sent_max_len, self.max_features, self.data_for_emb_layers, self.model)

    def plot_report(self, model_history):
        """
        Plot of loss function.
        :param model_history:
        :return:
        """

        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.data_path + '/nn_report/' + 'cnn_1level_model_loss.jpeg')
