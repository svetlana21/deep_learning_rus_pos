import os

from config.Config_worker import Config
from data_preparation.utils import *


class Stat:
    def __init__(self):
        self.config = Config(model_type='bilstm')
        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/')

        data = [
            self.data_path + '/' + self.config.get('Corpora') + '/test',
            self.data_path + '/' + self.config.get('Corpora') + '/train'
        ]

        for set in data:
            self.stat_pipline(set)

    def stat_pipline(self, file):
        sent = load_data(file)
        sent_len = length_count(sent)
        sent_count = len(sent_len)
        sent_freq_length_stat(sorted(sent_len))

        x_set = seq_form(sent)
        unique_tokens = unique_elements(x_set)
        unique_symbols = unique_chars(x_set)
        max_token_length = max([len(token) for token in unique_tokens])

        y_pos_set = seq_form(sent, data_type='y', task_type='POS')
        unique_pos_labels = unique_elements(y_pos_set)

        y_all_set = seq_form(sent, data_type='y', task_type='All')
        unique_all_labels = unique_elements(y_all_set)

        print('unique_tokens:', len(unique_tokens))
        print('unique_symbols:', len(unique_symbols))
        print('max_token_length:', max_token_length)
        print('tokens count', len([tokens for sent in x_set for tokens in sent]))
        print('unique_pos_labels:', len(unique_pos_labels))
        print('unique_all_labels:', len(unique_all_labels))
        print('sent_count:', sent_count)
        print(get_unique_grammatical_category(sent))


if __name__ == '__main__':
    stat = Stat()
