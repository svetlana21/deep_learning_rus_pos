from config.Config_worker import Config
from data_preparation.utils import *
from keras.preprocessing.sequence import pad_sequences
import gc
gc.collect()


class DataCNN2Level:
    def __init__(self, use_config=True, corpora='UD_Russian-SynTagRus', task_type='POS', class_index=3, dev=False,
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

        print('\nData preparation for 2 level model.')
        print('Task:', self.task_type)
        print('Corpora:', self.corpora)
        print('Label index:', class_index)

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/')
        self.tuned_vectors_path = os.path.abspath(file_path + '/../data/%s/cnn/model_level_1/' % (self.corpora,))

        tokens_tune_vectors = load_bin_data(
            self.tuned_vectors_path + '/cnn_prob_emb%s.pkl' % ('_%s_%s' % (
                self.prob_cnn_emb_layer_name, self.task_type)))

        sent_test = load_data(self.data_path + '/' + self.corpora + '/test')
        sent_train = load_data(self.data_path + '/' + self.corpora + '/train')
        if dev:
            sent_valid = load_data(self.data_path + '/' + self.corpora + '/dev')
        else:
            sent_valid = []

        test_tokens_data_seq = seq_form(sent_test)
        train_tokens_data_seq = seq_form(sent_train)
        if dev:
            dev_tokens_data_seq = seq_form(sent_valid)
        else:
            dev_tokens_data_seq = []

        test_labels_data_seq = seq_form(sent_test, data_type='y', task_type=self.task_type, task_index=class_index)
        train_labels_data_seq = seq_form(sent_train, data_type='y', task_type=self.task_type, task_index=class_index)
        if dev:
            dev_labels_data_seq = seq_form(sent_valid, data_type='y', task_type=self.task_type, task_index=class_index)
        else:
            dev_labels_data_seq = []

        self.MAX_SENT_LENGTH = max([len(s) for s in test_tokens_data_seq] + [len(s) for s in train_tokens_data_seq] +
                                   [len(s) for s in dev_tokens_data_seq])
        if verbose == 1:
            print('Max sent length:', self.MAX_SENT_LENGTH)

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
        self.label2ind_with_adding, self.ind2label_with_adding = self.labels_encode_cnn2(UNIQUE_LABELS)
        self.max_label_number = max(self.label2ind_with_adding.values())
        if verbose == 1:
            print('max_label_number:', self.max_label_number)

        y_train = self.label_data_prepare(train_labels_data_seq, verbose=verbose)
        y_test = self.label_data_prepare(test_labels_data_seq, verbose=verbose)
        if dev:
            y_dev = self.label_data_prepare(dev_labels_data_seq, verbose=verbose)
        else:
            y_dev = []

        save_binary(y_test, self.data_path + '/%s/' % (
            self.corpora,) + 'cnn/model_level_2/y_test_cnn2level_%s' % (self.task_type,))
        save_binary(y_train, self.data_path + '/%s/' % (
            self.corpora,) + 'cnn/model_level_2/y_train_cnn2level_%s' % (self.task_type,))
        if dev:
            save_binary(y_dev, self.data_path + '/%s/' % (
                self.corpora,) + 'cnn/model_level_2/y_dev_cnn2level_%s' % (self.task_type,))

        save_binary(self.label2ind_with_adding, self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_2/y_label2ind_cnn2level_%s' % (self.task_type,))
        del (y_train, y_test, y_dev, self.label2ind_with_adding, self.ind2label_with_adding, UNIQUE_LABELS)

        # After we can encode x test and train data.
        unique_tokens = sorted(set([k for k in tokens_tune_vectors]))
        self.word2ind_with_adding = {token: (index + 2) for index, token in enumerate(unique_tokens)}
        if verbose == 1:
            print("\nUnique tokens:", len(unique_tokens))

        x_test = self.data_prepare(test_tokens_data_seq, name="test", verbose=verbose)
        x_train = self.data_prepare(train_tokens_data_seq, name="train", verbose=verbose)
        if dev:
            x_dev = self.data_prepare(dev_tokens_data_seq, name="dev", verbose=verbose)
        else:
            x_dev = []

        tune_char_emb_matrix = self.matrix_creating(unique_tokens, tokens_tune_vectors)
        if verbose == 1:
            print("Tune embedding matrix:", tune_char_emb_matrix.shape)

        save_binary(
            x_test, self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_2/x_test_cnn2level_%s.pkl' % (self.task_type,))
        save_binary(
            x_train, self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_2/x_train_cnn2level_%s.pkl' % (self.task_type,))
        if dev:
            save_binary(
                x_dev, self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_2/x_dev_cnn2level_%s.pkl' % (self.task_type,))

        self.save_emb(
            ('max_label_numbers', self.max_label_number),
            ('max_sent_length', self.MAX_SENT_LENGTH),
            ('tune_char_emb_matrix', tune_char_emb_matrix),
            ('word2ind', self.word2ind_with_adding))

        del (self.max_label_number, self.MAX_SENT_LENGTH, tune_char_emb_matrix, self.word2ind_with_adding, x_test,
             x_train, x_dev, sent_test, sent_valid, sent_train, test_tokens_data_seq,
             train_tokens_data_seq, dev_tokens_data_seq, test_labels_data_seq, train_labels_data_seq,
             dev_labels_data_seq, test_labels_data, train_labels_data, dev_labels_data, unique_tokens)

    def save_emb(self,
                 max_label_numbers=None,
                 max_sent_length=None,
                 tune_char_emb_matrix=None,
                 word2ind=None
                 ):
        """
        Сохранение данных по признаку.
        """

        emb_feature_data = dict()
        emb_feature_data[max_label_numbers[0]] = max_label_numbers[1]
        emb_feature_data[max_sent_length[0]] = max_sent_length[1]
        emb_feature_data[tune_char_emb_matrix[0]] = tune_char_emb_matrix[1]
        emb_feature_data[word2ind[0]] = word2ind[1]
        save_binary(emb_feature_data,
                    self.data_path + '/%s/' % (self.corpora,) + 'cnn/model_level_2/char_emb_cnn2_feature_data_%s.pkl' % (self.task_type,))

    def matrix_creating(self, unique_tokens, tokens_tune_vectors):
        emb_vocab = list()
        zero_vector = np.zeros(len(tokens_tune_vectors['_null_']))
        emb_vocab.append(zero_vector)
        emb_vocab.append(tokens_tune_vectors['_null_'])
        for tokens in unique_tokens:
            emb_vocab.append(tokens_tune_vectors[tokens])
        return np.array(emb_vocab)

    def data_prepare(self, x_set, name, verbose=1):
        """
        Подготовка данных.
        :param x:
        :param y:
        :return:
        """

        x_enc = [[self.word2ind_with_adding[c] for c in x] for x in x_set]
        x = pad_sequences(x_enc, maxlen=self.MAX_SENT_LENGTH, value=1)
        if verbose == 1:
            print('x_shape: %s;' % (name,), x.shape)
            print('sequence example:', x[0])
        return x

    def labels_encode_cnn2(self, unique_elements):
        """
        Encoding labels by numbers. 1 for null;
        Short sentences are extended from the beginning with “null words” consisting of “null” label characters.
        Such “null words” belong to special null class.
        :param unique_labels:
        :return:
        """

        return {label: (index + 2) for index, label in enumerate(unique_elements)}, \
               {(index + 2): label for index, label in enumerate(unique_elements)}

    def label_data_prepare(self, y_set, verbose=1):
        """
        Creating one-hot vector for encoding labels.
        :param y:
        :return:
        """

        y_set = [[self.label2ind_with_adding[t] for t in s] for s in y_set]
        y_set = pad_sequences(y_set, maxlen=self.MAX_SENT_LENGTH, value=1)
        if verbose == 1:
            print('y_shape:', y_set.shape, y_set[0])
        return y_set
