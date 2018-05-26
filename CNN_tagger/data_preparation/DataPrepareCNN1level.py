from config.Config_worker import Config
from data_preparation.utils import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import gc
gc.collect()



class DataCNN1Level:
    def __init__(self, use_config=True, corpora='UD_Russian-SynTagRus', task_type='POS', class_index=3, dev=True, verbose=1):

        if use_config:
            self.config = Config(model_type='cnn')
            self.task_type = self.config.get('Task_type')
            self.class_index = self.config.get('Classification_tasks')['UD2']['POS'][0]
            self.corpora = self.config.get('Corpora')
        else:
            self.task_type = task_type
            self.corpora = corpora
            self.class_index = class_index

        print('Data preparation for 1 level model.')
        print('Task:', self.task_type)
        print('Corpora:', self.corpora)
        print('Label index:', class_index)

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/')

        sent_test = load_data(self.data_path + '/' + self.corpora + '/test')
        sent_train = load_data(self.data_path + '/' + self.corpora + '/train')
        if dev:
            sent_valid = load_data(self.data_path + '/' + self.corpora + '/dev')
        else:
            sent_valid = []

        test_tokens_data_seq = seq_form(sent_test, task_type=self.task_type)
        train_tokens_data_seq = seq_form(sent_train, task_type=self.task_type)
        if dev:
            dev_tokens_data_seq = seq_form(sent_valid, task_type=self.task_type)
        else:
            dev_tokens_data_seq = []

        test_labels_data_seq = seq_form(sent_test, data_type='y', task_type=self.task_type, task_index=self.class_index)
        train_labels_data_seq = seq_form(sent_train, data_type='y', task_type=self.task_type, task_index=self.class_index)
        if dev:
            dev_labels_data_seq = seq_form(sent_valid, data_type='y', task_type=self.task_type, task_index=self.class_index)
        else:
            dev_labels_data_seq = []

        test_tokens_data = [tokens for sent in test_tokens_data_seq for tokens in sent]
        train_tokens_data = [tokens for sent in train_tokens_data_seq for tokens in sent]
        if dev:
            dev_tokens_data = [tokens for sent in dev_tokens_data_seq for tokens in sent]
        else:
            dev_tokens_data = []

        test_labels_data = [labels for sent in test_labels_data_seq for labels in sent]
        train_labels_data = [labels for sent in train_labels_data_seq for labels in sent]
        if dev:
            dev_labels_data = [labels for sent in dev_labels_data_seq for labels in sent]
        else:
            dev_labels_data = []

        # After we can encode y test and train data.
        self.ADDING_INDEX = 1
        self.PADDING_VALUE = 0

        UNIQUE_LABELS = sorted(set(test_labels_data + train_labels_data + dev_labels_data))
        self.label2ind_with_adding, self.ind2label_with_adding = labels_encode(UNIQUE_LABELS, 0)
        self.max_label_numbers = max(self.label2ind_with_adding.values())
        if verbose == 1:
            print('Unique labels:', self.max_label_numbers)
            print("\nLabels:", self.label2ind_with_adding.keys())

        y_train = self.label_data_prepare(train_labels_data, verbose=verbose)
        y_test = self.label_data_prepare(test_labels_data, verbose=verbose)
        if dev:
            y_dev = self.label_data_prepare(dev_labels_data, verbose=verbose)
        else:
            y_dev = []

        save_binary(y_test,
                    self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_1/y_test_cnn1level_%s' % (self.task_type,))
        save_binary(y_train,
                    self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_1/y_train_cnn1level_%s' % (self.task_type,))
        save_binary(y_dev,
                    self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_1/y_dev_cnn1level_%s' % (self.task_type,))
        save_binary(self.label2ind_with_adding,
                    self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_1/y_label2ind_cnn1level_%s' % (self.task_type,))

        del (y_train,
             y_test,
             y_dev,
             self.label2ind_with_adding,
             self.ind2label_with_adding,
             self.max_label_numbers,
             UNIQUE_LABELS)

        # After we can encode x test, dev, train data.
        unique_tokens = sorted(set(test_tokens_data + train_tokens_data + dev_tokens_data))
        if verbose == 1:
            print("\nUnique tokens:", len(unique_tokens))

        self.unique_symbols = unique_chars(unique_tokens)
        self.max_token_length = max([len(token) for token in unique_tokens])
        self.symbol2ind_with_adding, self.ind2symbol_with_adding = symbols_encode(self.unique_symbols, self.ADDING_INDEX)
        if verbose == 1:
            print("\nUnique symbols:", self.symbol2ind_with_adding.keys())

        x_test = self.data_prepare(test_tokens_data, verbose=verbose)
        x_train = self.data_prepare(train_tokens_data, verbose=verbose)
        x_dev = self.data_prepare(dev_tokens_data, verbose=verbose)

        save_binary(x_test, self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_1/x_test_cnn1level.pkl')
        save_binary(x_train, self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_1/x_train_cnn1level.pkl')
        save_binary(x_dev, self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_1/x_dev_cnn1level.pkl')

        char_embeddings = self.char_matrix_cnn()
        if verbose == 1:
            print('\nChar_embeddings shape:', char_embeddings.shape)

        self.save_emb(
            ('symbol2ind', self.symbol2ind_with_adding),
            ('ind2symbol', self.ind2symbol_with_adding),
            ('max_token_length', self.max_token_length),
            ('char_matrix', char_embeddings)
        )

        del (self.symbol2ind_with_adding, self.ind2symbol_with_adding, self.max_token_length, char_embeddings,
             self.unique_symbols, x_test, x_train, x_dev, sent_test, sent_valid, sent_train, test_tokens_data_seq,
             train_tokens_data_seq, dev_tokens_data_seq, test_labels_data_seq, train_labels_data_seq,
             dev_labels_data_seq, test_tokens_data, train_tokens_data, dev_tokens_data,
             test_labels_data, train_labels_data, dev_labels_data, unique_tokens)

    def data_prepare(self, x_set, verbose=1):
        """
        Encoding symbols using dict symbols per digit and padding symbols sequence.
        :param x:
        :param y:
        :return:
        """

        x_enc = [[self.symbol2ind_with_adding[char] for char in token] for token in x_set]
        x = pad_sequences(x_enc, maxlen=self.max_token_length, value=self.PADDING_VALUE)
        if verbose == 1:
            print('x tensor shapes: %s' % (x.shape,), x[0])
        return x

    def save_emb(self,
                 symbol2ind=None,
                 ind2symbol=None,
                 max_token_length=None,
                 char_matrix=None
                 ):
        """
        Сохранение данных по признаку.
        """

        emb_feature_data = dict()
        emb_feature_data[symbol2ind[0]] = symbol2ind[1]
        emb_feature_data[ind2symbol[0]] = ind2symbol[1]
        emb_feature_data[max_token_length[0]] = max_token_length[1]
        emb_feature_data[char_matrix[0]] = char_matrix[1]
        save_binary(emb_feature_data, self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_1/char_emb_cnn1_feature_data_%s.pkl' % (self.task_type,))

    def char_matrix_cnn(self):
        """
        Creating matrix with char embedding for cnn network.
        Example:
            0 [1 0 0 ... 0 ]
            ! [0 1 0 0 ...]
        """

        char_emb_vocab = list()
        null_vector = np.zeros(len(self.symbol2ind_with_adding) + 1, dtype='int8')
        null_vector[0] = 1
        char_emb_vocab.append(null_vector)
        for symbols in self.unique_symbols:
            features_per_symbol = np.zeros(len(self.symbol2ind_with_adding) + 1, dtype='int8')
            features_per_symbol[self.symbol2ind_with_adding[symbols]] = 1
            char_emb_vocab.append(features_per_symbol)
        return np.array(char_emb_vocab).astype('int8')

    def label_data_prepare(self, y_set, verbose=1):
        """
        Creating one-hot vector for encoding labels.
        :param y:
        :return:
        """

        y_set = to_categorical([self.label2ind_with_adding[l] for l in y_set], self.max_label_numbers+1)
        if verbose == 1:
            print('y tensor shapes: %s' % (y_set.shape,), y_set[0])
        return y_set
