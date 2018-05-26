from collections import Counter
import pickle
import os
import numpy as np

from keras.utils import to_categorical


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


def symbols_encode(unique_elements, adding_index):
    """
    Encoding chars by numbers. index + 1 (“null” label);
    .
    :param unique_labels:
    :return:
    """

    return {symbol: (index+adding_index) for index, symbol in enumerate(unique_elements)}, \
           {(index+adding_index): symbol for index, symbol in enumerate(unique_elements)}


def labels_encode(unique_elements, adding_index):
    """
    Encoding labels by numbers. index + 1, because we use padding.
    :param unique_labels:
    :return:
    """

    return {label: (index+adding_index) for index, label in enumerate(unique_elements)}, \
           {(index+adding_index): label for index, label in enumerate(unique_elements)}


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
        pickle.dump(data, file, protocol=4)


def load_grammatical_cat_model1(data_path, task_type, verbose=1):
    """
    Loading y data for each grammatical category.
    """

    labels = None
    for files in os.listdir(data_path):
        if task_type == 'All':
            if 'y_test_cnn1level_All' in files:
                labels = load_bin_data(data_path + '/' + files)
        else:
            if 'y_test_cnn1level_%s' % (task_type, ) in files:
                labels = load_bin_data(data_path + '/' + files)
    if verbose == 1:
        print('y_data:', labels.shape)
    return labels


def load_grammatical_cat_model2(data_path, task_type, verbose=1):
    """
    Loading y data for each grammatical category.
    """

    labels = None
    labels_2indexes= None
    for files in os.listdir(data_path):
        if task_type == 'All':
            if 'y_test_cnn2level_All' in files:
                labels = load_bin_data(data_path + '/' + files)
            if 'y_label2ind_cnn2level_All' in files:
                labels_2indexes = len(load_bin_data(data_path + '/' + files))
        else:
            if 'y_test_cnn2level_%s' % (task_type, ) in files:
                labels = load_bin_data(data_path + '/' + files)
            if 'y_label2ind_cnn2level_%s' % (task_type, ) in files:
                labels_2indexes = len(load_bin_data(data_path + '/' + files))
    if verbose == 1:
        print('y_data:', labels.shape)
    return labels, labels_2indexes


def preparation_data_to_score_model1(yh, pr):
    return np.argmax(yh, axis=1), np.argmax(pr, axis=1)


def transform2categorical(data, out_size):
    return np.array([[to_categorical(t, out_size+2)[0] for t in s] for s in data])


def preparation_data_to_score_model2(yh, pr, verbose=1, sent_borders=False):
    yh = yh.argmax(2)
    pr = [list(np.argmax(el, axis=-1)) for el in pr]
    # с какого элемента начинаются не 0; например: [37, 47];
    coords = [np.where(yhh > 1)[0][0] for yhh in yh]
    # оставляем только не 0 элементы;
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    # по границам y оставялем не 0 в предсказании;
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    if verbose == 1:
        print("Example of prediction sent:", ypr[-1])
        print("Example of test sent:", yh[-1])
    if sent_borders:
        return yh, ypr
    else:
        # конкатенация массивов для сравнения;
        fyh = [c for row in yh for c in row]
        fpr = [c for row in ypr for c in row]
        return fyh, fpr


def load_bin_data(path_to_data):
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


def seq_form(data, data_type='x', task_type=None, task_index=3):
    """
    Forming data sequences.
    :param data:
    :return:
    """

    if data_type == 'x':
        return [[c[1] for c in x] for x in data]
    else:
        if 'POS' in task_type:
            return [[c[task_index] for c in x] for x in data]
        else:
            return [[c[5] for c in x] for x in data]


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


def load_data(path_to_data, save_sent=False):
    """
    Loading sequences.
    :param self: 
    :param path_to_data: 
    :return: 
    """

    print('Loading:', path_to_data)
    raw = open(path_to_data, 'r').readlines()
    all_x = []
    point = []
    for line in raw:
        stripped_line = line.strip().split('\t')
        if not save_sent:
            if '#' not in stripped_line[0]:
                point.append(stripped_line)
        else:
            point.append(stripped_line)
        if line == '\n':
            all_x.append(point[:-1])
            point = []
    return all_x
