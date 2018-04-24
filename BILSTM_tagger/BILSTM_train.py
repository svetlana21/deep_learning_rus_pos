# Keras==2.0.3 tensorflow=1.1.0rc1
import pickle

from config.Config_worker import Config

import numpy
numpy.random.seed(1024)


from keras.models import *
from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import *
from keras.utils import plot_model
import matplotlib.pyplot as plt
import gc; gc.collect()


class LSTMTagger:
    """
    Mean of option "Task_type". Grammem: "Grammem_tag_Animacy", POS, All (all morphology properties.).
    """
    def __init__(self'):
        self.config = Config(model_type='bilstm')
        self.task_type = self.config.get('Task_type')

        print('#' * 100)
        print('Task:', self.task_type)
        print('Corpora:', self.config.get('Corpora'))
        print('#' * 100)

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/%s/' % (self.config.get('Corpora'),))
        self.model_path = os.path.abspath(file_path + '/../tagger_models/')

        char_emb_feature = self.load_binary_data(self.data_path + '/char_emb_feature_data.pkl')
        w2v_emb_feature = self.load_binary_data(self.data_path + '/w2v_emb_feature_data.pkl')

        w2v_emb_feature['w2v_matrix'] = w2v_emb_feature['w2v_matrix'].astype('int64')
        self.word2ind = char_emb_feature['word2index']
        self.sent_max_len = char_emb_feature['max_sent_length']

        self.x_train = self.load_binary_data(self.data_path + '/x_train.pkl')

        self.max_features = len(self.word2ind) + 1
        self.random_embedding_size = self.config.get('Network_options').get('random_embedding_size')
        self.lstm_hidden_size = self.config.get('Network_options').get('lstm_hidden_size')
        self.dense_hidden_size = self.config.get('Network_options').get('dense_hidden_size')
        self.batch_size = self.config.get('Network_options').get('batch_size')
        self.epoch = self.config.get('Network_options').get('training_epoch')

        self.data_for_emb_layers = {
            'char': char_emb_feature['char_matrix'],
            'w2v':  w2v_emb_feature['w2v_matrix']
        }

        self.y_train, self.out_size = self.load_grammatical_cat()

        print('data embedding char shape:', self.data_for_emb_layers['char'].shape, self.data_for_emb_layers['char'].dtype)
        print('data embedding w2v shape:', self.data_for_emb_layers['w2v'].shape, self.data_for_emb_layers['w2v'].dtype)
        
        self.model = None

    def load_grammatical_cat(self):
        """
        Loading y data for each grammatical category.
        """

        labels = []
        labels_2indexes = []
        for files in os.listdir(self.data_path):

            if 'y_train_Grammem_tag' in files:
                if self.task_type != 'All': 
                    if self.task_type in files:
                        labels.append(
                            [files.split('.')[0].split('y_train_Grammem_tag_')[1], 
                            self.load_binary_data(self.data_path + '/' + files)])
                else:
                    labels.append(
                        [files.split('.')[0].split('y_train_Grammem_tag_')[1], 
                        self.load_binary_data(self.data_path + '/' + files)])

            if 'y_label2ind_Grammem_tag' in files:
                if self.task_type != 'All': 
                    if self.task_type in files:
                        labels_2indexes.append(
                            [files.split('.')[0].split('y_label2ind_Grammem_tag_')[1], 
                            len(self.load_binary_data(self.data_path + '/' + files)) + 1])
                else:
                    labels_2indexes.append(
                        [files.split('.')[0].split('y_label2ind_Grammem_tag_')[1], 
                        len(self.load_binary_data(self.data_path + '/' + files)) + 1])
        return sorted(labels), sorted(labels_2indexes)

    def load_binary_data(self, path_to_data):
        """
        Загрузка признаков
        :return:
        """

        with open(path_to_data, 'rb') as f:
            data = pickle.load(f)
        return data

    def network_initialization(self):
        """
        Инициализация сети.

        MaskLambda
            The next crucial building block is a way to reverse sequences, and also their masks. One way to reverse
            sequences in Keras is with a Lambda layer that wraps x[:,::-1,:] on the input tensor. Unfortunately I
            couldn't find a way in straight  Keras that will also reverse the mask, but @braingineer created the perfect
            custom lambda layer that allows us to manipulate the mask with an arbitrary function.
            
        
        LSTM vs GRU 
        The GRU unit controls the flow of information like the LSTM unit, but without having to use a memory unit. 
        It just exposes the full hidden content without any control.
        
        GRU is relatively new, and from my perspective, the performance is on par with LSTM, but computationally 
        more efficient (less complex structure as pointed out). So we are seeing it being used more and more.
        
        https://arxiv.org/pdf/1412.3555v1.pdf
            
        :return:
        """

        #####################################################################################################################
        input_char_emb = Input((self.sent_max_len,), name='input_char_emb')
        char_emb = Embedding(
            input_dim=self.max_features,
            output_dim=self.data_for_emb_layers.get('char').shape[1],
            input_length=self.sent_max_len,
            weights=[self.data_for_emb_layers.get('char')],
            mask_zero=True,
            trainable=False
            )(input_char_emb)
        bilstm_layer_char_emb_0 = Bidirectional(LSTM(
            self.lstm_hidden_size,
            return_sequences=True,
            activation='tanh',
            recurrent_activation="hard_sigmoid",))(char_emb)
            # dropout=0.2, recurrent_dropout=0.2)))
        bilstm_layer_char_emb_1 = Bidirectional(LSTM(
            self.lstm_hidden_size,
            return_sequences=True,
            activation='tanh',
            recurrent_activation="hard_sigmoid",))(bilstm_layer_char_emb_0)
            # dropout=0.2, recurrent_dropout=0.2)))
        char_emb_out = Dropout(0.5)(bilstm_layer_char_emb_1)
        #####################################################################################################################
        input_w2v_emb = Input((self.sent_max_len,), name='input_w2v_emb')
        mw2v_emb = Embedding(
            input_dim=self.max_features,
            output_dim=self.data_for_emb_layers.get('w2v').shape[1],
            input_length=self.sent_max_len,
            weights=[self.data_for_emb_layers.get('w2v')],
            mask_zero=True,
            trainable=False
        )(input_w2v_emb)
        bilstm_layer_w2v_emb_0 = Bidirectional(LSTM(
            self.lstm_hidden_size,
            return_sequences=True,
            activation='tanh',
            recurrent_activation="hard_sigmoid", ))(mw2v_emb)
            # dropout=0.2, recurrent_dropout=0.2)))
        bilstm_layer_w2v_emb_1 = Bidirectional(LSTM(
            self.lstm_hidden_size,
            return_sequences=True,
            activation='tanh',
            recurrent_activation="hard_sigmoid", ))(bilstm_layer_w2v_emb_0)
            # dropout=0.2, recurrent_dropout=0.2)))
        w2v_emb_out = Dropout(0.5)(bilstm_layer_w2v_emb_1)
        #####################################################################################################################
        dense_network = concatenate([char_emb_out, w2v_emb_out])
        dense_network = TimeDistributed(Dense(self.dense_hidden_size, activation='relu'))(dense_network)
        dense_network = Dropout(0.5)(dense_network)
        dense_network = TimeDistributed(Dense(self.dense_hidden_size, activation='relu'))(dense_network)
        dense_network = Dropout(0.5)(dense_network)

        output_layers_massive = []
        for i in range(len(self.y_train)):
            output = Dense(self.out_size[i][1], activation='softmax', name=self.out_size[i][0])(dense_network)
            output_layers_massive.append(output)

        #####################################################################################################################

        model = Model(inputs=[input_char_emb, input_w2v_emb], outputs=output_layers_massive)
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
        self.model = model

        plot_model(model, to_file=self.model_path + '/model_schema_%s_%s.png' %
                                                                 (self.config.get('Corpora'),
                                                                  self.task_type,), show_shapes=True)
        print(model.summary())
        print('Model comp done.')

    def training(self):
        """
        Обучение.

        model_checkpoint
            For `val_acc`, this should be `max`, for `val_loss` this should  be `min`, etc. In `auto` mode, the
            direction is automatically inferred from the name of the monitored quantity.

        early_stopping
                In `min` mode, training will stop when the quantity monitored has stopped decreasing; in `max` mode it
                will stop when the quantity  monitored has stopped increasing; in `auto` mode, the direction is
                automatically inferred from the name of the monitored quantity.

        :return:
        """

        model_checkpoint = ModelCheckpoint(filepath=os.path.join(self.model_path + '/model_%s_%s.pkl' %
                                                                 (self.config.get('Corpora'),
                                                                  self.task_type,)),
                                           monitor='val_loss',
                                           verbose=0,
                                           save_weights_only=False,
                                           save_best_only=True,
                                           mode='min')

        # TODO Early stopping in multi-task learning
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       verbose=1,
                                       mode='auto')

        train_data = [self.x_train, self.x_train]

        history = self.model.fit(train_data,
                                 [el[1] for el in self.y_train],
                                 batch_size=self.batch_size,
                                 epochs=self.epoch,
                                 validation_split=0.1,
                                 verbose=2,
                                 shuffle=True,
                                 callbacks=[model_checkpoint, early_stopping])

    def plot_report(self, model_history):
        """
        График loss function.
        :param model_history:
        :return:
        """

        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.data_path + '/nn_report/' + 'model_loss.jpeg')

if __name__ == '__main__':
    lstm_tagger = LSTMTagger()
    lstm_tagger.network_initialization()
    lstm_tagger.training()
