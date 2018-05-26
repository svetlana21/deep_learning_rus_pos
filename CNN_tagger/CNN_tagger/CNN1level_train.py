import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(1024)
rn.seed(1024)
tf.set_random_seed(1024)

import pickle
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.utils import plot_model
import matplotlib.pyplot as plt
from config.Config_worker import Config

import gc
gc.collect()


class CNN1levelTagger:
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
                 dev=False,
                 ):

        if use_config:
            self.config = Config(model_type='cnn')
            self.task_type = self.config.get('Task_type')
            self.class_index = self.config.get('Classification_tasks')['UD2']['POS'][0]
            self.corpora = self.config.get('Corpora')
            self.batch_size_level_1 = self.config.get('Network_options').get('batch_size_level_1')
            self.epoch = self.config.get('Network_options').get('training_epoch')
        else:
            self.task_type = task_type
            self.corpora = corpora
            self.class_index = class_index
            self.batch_size_level_1 = batch_size
            self.epoch = epoch

        print('\nModel train for 1 level model.')
        print('Task:', self.task_type)
        print('Corpora:', self.corpora)
        print('Label index:', class_index)

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/%s/cnn/model_level_1/' % (self.corpora,))
        self.model_path = os.path.abspath(file_path + '/../tagger_models/')

        self.char_emb_feature = self.load_binary_data(self.data_path + '/char_emb_cnn1_feature_data_%s.pkl' % (self.task_type,))

        self.symbol2ind = self.char_emb_feature['symbol2ind']
        self.max_token_length = self.char_emb_feature['max_token_length']

        self.x_train = self.load_binary_data(self.data_path + '/x_train_cnn1level.pkl')
        self.y_train, self.out_size = self.load_grammatical_cat(verbose=verbose)
        if verbose == 1:
            print('x_train shape:', self.x_train.shape)
            print('y_train shape:', self.y_train.shape)

        if dev:
            self.x_dev = self.load_binary_data(self.data_path + '/x_dev_cnn1level.pkl')
            self.y_dev, _ = self.load_grammatical_cat(y_data_name='dev', verbose=verbose)
            if verbose == 1:
                print('x_dev shape:', self.x_dev.shape)
                print('y_dev shape:', self.y_dev.shape)

        self.max_features = max(self.symbol2ind.values()) + 1
        self.data_for_emb_layers = {
            'char': self.char_emb_feature['char_matrix']
        }

        if verbose == 1:
            print('data embedding char shape:',
                  self.data_for_emb_layers['char'].shape,
                  self.data_for_emb_layers['char'].dtype)
        
        self.model = None

    def load_grammatical_cat(self, y_data_name='train', verbose=1):
        """
        Loading y data for each grammatical category.
        """

        labels = None
        labels_2indexes = None

        for files in os.listdir(self.data_path):
            if self.task_type == 'All':
                if 'y_%s_cnn1level_All' % (y_data_name,) in files:
                    labels = self.load_binary_data(self.data_path + '/' + files)
                if 'y_label2ind_cnn1level_All' in files:
                    labels_2indexes = len(self.load_binary_data(self.data_path + '/' + files))
            else:
                if 'y_%s_cnn1level_%s' % (y_data_name, self.task_type) in files:
                    labels = self.load_binary_data(self.data_path + '/' + files)
                if 'y_label2ind_cnn1level_%s' % (self.task_type, ) in files:
                    labels_2indexes = len(self.load_binary_data(self.data_path + '/' + files))
        if verbose == 1:
            print('y_data:', labels.shape)
        return labels, labels_2indexes

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

        ###################################################Network_level_1##############################################

        model = Sequential()

        model.add(Embedding(
            input_dim=self.max_features,
            output_dim=self.data_for_emb_layers.get('char').shape[1],
            input_length=self.max_token_length,
            weights=[self.data_for_emb_layers.get('char')],
            mask_zero=False,
            trainable=False
            ))

        model.add(Conv1D(
            filters=1024,
            kernel_size=5,
            padding='valid',
            activation='relu',
            strides=1,
            name='conv1d',
        ))
        model.add(GlobalMaxPooling1D(name='global_max_pooling1d'))

        model.add(BatchNormalization())

        model.add(Dense(256, activation='relu', name='dense_3'))

        model.add(Dense(256, activation='relu', name='dense_4'))

        model.add(BatchNormalization())

        model.add(Dense(self.out_size, activation='softmax', name='dense_5'))

        model.compile(optimizer='adamax', loss="mean_squared_error", metrics=["accuracy"])

        self.model = model
        print(model.summary())

        if verbose == 1:
            plot_model(model, to_file=self.model_path + '/cnn_1level_model_schema_%s_%s.png' %
                                                                 (self.corpora,
                                                                  self.task_type,), show_shapes=True)
            print(model.summary())

    def training(self, verbose=1, dev=False):
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

        model_checkpoint = ModelCheckpoint(filepath=os.path.join(self.model_path + '/cnn_1level_model_%s_%s.pkl' %
                                                                 (self.corpora,
                                                                  self.task_type,)),
                                           monitor='val_loss',
                                           verbose=0,
                                           save_weights_only=False,
                                           save_best_only=True,
                                           mode='min')

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=15,
                                       verbose=0,
                                       mode='auto')

        if dev:
            self.model.fit(
                x=self.x_train,
                y=self.y_train,
                batch_size=self.batch_size_level_1,
                epochs=self.epoch,
                validation_data=(self.x_dev, self.y_dev),
                verbose=2,
                shuffle=True,
                callbacks=[model_checkpoint, early_stopping]
            )

            del (self.char_emb_feature, self.x_train, self.y_train, self.out_size,
                 self.data_for_emb_layers, self.symbol2ind, self.max_token_length, self.model)
        else:
            self.model.fit(
                x=self.x_train,
                y=self.y_train,
                batch_size=self.batch_size_level_1,
                epochs=self.epoch,
                validation_split=0.1,
                verbose=2,
                shuffle=True,
                callbacks=[model_checkpoint, early_stopping]
            )

            del (self.char_emb_feature, self.x_train, self.y_train, self.out_size,
                 self.data_for_emb_layers, self.symbol2ind, self.max_token_length, self.model)


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
