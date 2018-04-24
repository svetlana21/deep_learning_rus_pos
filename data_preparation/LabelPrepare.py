import os
import pickle
from collections import Counter

from config.Config_worker import Config
from data_preparation.utils import *

from keras.preprocessing.sequence import pad_sequences
import numpy as np


class LabelPreparation:
    def __init__(self):
        self.config = Config(model_type='bilstm')

        print('#' * 100)
        print('Task:', self.config.get('Task_type'))
        print('Corpora:', self.config.get('Corpora'))
        print('Label encoding')
        print('#' * 100)

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/')
        self.sent_max_len = self.config.get('Sent_max_length')
        self.corpora_limit = self.config.get('Corpora_sent_limit')

        sent_test = load_data(self.data_path + '/' + self.config.get('Corpora') + '/test')
        if self.corpora_limit != 'False':
            sent_train = load_data(self.data_path + '/' + self.config.get('Corpora') + '/train')[
                         :self.config.get('Corpora_sent_limit')]
        else:
            sent_train = load_data(self.data_path + '/' + self.config.get('Corpora') + '/train')

        for classification_task in self.config.get('Classification_tasks')[self.config.get('Corpora')]:
            print('\nClassification tasks:', classification_task)
            # We must find all unique labels in test and train for replace thenm by index.
            y_set = self.y_set_form(
                sent_test + sent_train,
                (classification_task,
                self.config.get('Classification_tasks')[self.config.get('Corpora')][classification_task])
                )
            unique_labels = unique_elements(y_set)
            self.label2ind, self.ind2label = elements_encode(unique_labels)
            self.max_label_numbers = max(self.label2ind.values()) + 1
            print('labels: %s; with label for 0: %s' % (len(unique_labels), self.max_label_numbers))
            del (y_set, unique_labels)

            # After we can encode test and train data.
            y_train = self.data_prepare(
                self.y_set_form(
                sent_train,
                (classification_task,
                self.config.get('Classification_tasks')[self.config.get('Corpora')][classification_task])
                )
            )
            save_binary(y_train, 'y_train_%s.pkl' % (classification_task, ))
            del y_train

            y_test = self.data_prepare(
                self.y_set_form(
                sent_test,
                (classification_task,
                self.config.get('Classification_tasks')[self.config.get('Corpora')][classification_task])
                )
            )
            save_binary(y_test,
                        self.data_path + '/%s/' % (self.config.get('Corpora'),) + '/bilstm/' + 'y_test_%s.pkl' % (
                        classification_task,))

            del y_test

            # Save label2ind
            save_binary(self.label2ind,
                        self.data_path + '/%s/' % (
                            self.config.get('Corpora'),) + '/bilstm/' + 'y_label2ind_%s.pkl' % (classification_task,))

    def y_set_form(self, data, task_type):
        """
        Forming y in accordance with classification task.
        :param data:
        :param task_type:
        :return:
        """

        y = None
        if 'Grammem_tag' in task_type[0]:
            y = [[''.join([grammems for grammems in t[4].split('|') if task_type[1][0] in grammems]) for t in sent] for sent in data]
            y = [[t if t != '' else 'Null' for t in s] for s in y]
            print(y[:10])
        else:
            if task_type[0] == 'POS':
                y = [[t[task_type[1][0]] for t in sent] for sent in data]
                print(y[:10])
            if task_type[0] == 'Morpho_tag':
                y = [[t[task_type[1][0]] for t in sent] for sent in data]
                print(y[:10])
            if task_type[0] == 'All':
                y = [['Pos=' + t[task_type[1][0]] + '|' + t[task_type[1][1]] for t in sent] for sent in data]
                print(y[:10])
        return y

    def data_prepare(self, y_set):
        """
        Creating one-hot vector for encoding labels.
        :param y:
        :return:
        """

        y_enc = [[0] * (self.sent_max_len - len(ey)) + [self.label2ind[c] for c in ey] for ey in y_set]
        y_enc = [[one_hot_encode(c, self.max_label_numbers) for c in ey] for ey in y_enc]
        y_train = pad_sequences(y_enc, maxlen=self.sent_max_len)
        print('Testing tensor shapes:')
        print('y_shape:', y_train.shape)
        return y_train


if __name__ == '__main__':
    LabelPreparation()
