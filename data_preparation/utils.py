from collections import Counter
import pickle

import numpy as np


def get_unique_grammatical_category(data):
    """
    Names of grammatical categories.
    :param data:
    :return
    """

    grammatical_category = set()
    x = [[c[4] for c in x] for x in data]
    x = [[[grammatical_category.add(g.split('=')[0]) for g in k.split('|')] for k in el] for el in x]
    return sorted(grammatical_category)


def elements_encode(unique_elements):
    """
    Encoding labels by numbers. index + 1, because we use padding.
    :param unique_labels:
    :return:
    """

    return {label: (index + 1) for index, label in enumerate(unique_elements)}, \
           {(index + 1): label for index, label in enumerate(unique_elements)}


def one_hot_encode(x, n):
    """
    One-hot-encoding.
    :param x:
    :param n:
    :return:
    """

    result = np.zeros(n)
    result[x] = 1
    return result


def save_binary(data, file_name):
    """
    Save data in binary format.
    :param data:
    :param file_name:
    :return:
    """

    with open(file_name, 'wb') as file:
        pickle.dump(data, file)


def load_bin_data( path_to_data):
    """
    Load binary data.
    :return:
    """

    with open(path_to_data, 'rb') as f:
        data = pickle.load(f)
    return data


def unique_chars(x_data):
    """
    Massive of unique symbols.
    :param x_dataa:
    :param y_data:
    :return:
    """

    return sorted(set([char for sent in x_data for tokens in sent for char in tokens]))


def unique_elements(data):
    """
    Massive of unique elements.
    :param x_set:
    :return:
    """

    return sorted(set([tokens for sent in data for tokens in sent]))


def seq_form(data, data_type='x', task_type=None):
    """
    Forming data sequences.
    :param data:
    :return:
    """

    if data_type == 'x':
        return [[c[1] for c in x] for x in data]
    else:
        if task_type == 'POS':
            return [[c[3] for c in x] for x in data]
        else:
            return [[c[4] for c in x] for x in data]


def sent_freq_length_stat(sent_lengths):
    """
    Count length of sequences.
    :param self: 
    :param sent_lengths: 
    :return: 
    """
    c = Counter(sent_lengths)
    print('\nSent length freq:', c, '\n')


def length_count(data):
    """
    Length of each sequences.
    :param self: 
    :param data: 
    :return: 
    """

    return [len(x) for x in data]


def load_data(path_to_data):
    """
    Loading sequences.
    :param self: 
    :param path_to_data: 
    :return: 
    """

    print('Loading:', path_to_data, '\n')
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