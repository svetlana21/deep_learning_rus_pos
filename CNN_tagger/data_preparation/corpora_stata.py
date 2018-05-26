import os
from pprint import pprint

def load_data(path_to_data):
    # print('Loading:', path_to_data)
    raw = open(path_to_data, 'r').readlines()
    all_x = []
    point = []
    for line in raw:
        stripped_line = line.strip().split('\t')
        point.append(stripped_line)
        if line == '\n':
            all_x.append(point[:-1])
            point = []
    # all_x = all_x[:-1]
    return all_x


def count_elements(data):
    sents = [s[2:] for s in data]
    tokens = [t for s in data for t in s if '#' not in t[0]]

    return {
        'tokens': [t[1] for t in tokens],
        'unique_tokens': list(set([t[1] for t in tokens])),
        'unique_UPOS': list(set([t[3] for t in tokens])),
        'unique_XPOS': list(set([t[4] for t in tokens])),
        'unique_symbols': set([sym for t in tokens for sym in t[1]]),
        'max_token_length': max([len(t[1]) for t in tokens], default=0),
        'max_sent_length': max([len(s) for s in sents], default=0),
        'avg_token_length': sum([len(t[1]) for t in tokens]) / len(tokens) if len(tokens) != 0 else 0,
        'avg_sent_length': sum([len(s) for s in sents]) / len(sents) if len(tokens) != 0 else 0
    }


data_path = '../data/'

for folders in sorted(os.listdir(data_path)):
    if folders.startswith('UD'):
        train = load_data(data_path + folders + '/train')
        test = load_data(data_path + folders + '/test')
        if os.path.exists(data_path + folders + '/dev'):
            dev = load_data(data_path + folders + '/dev')
        else:
            dev = []

        print('\nCorpus:', folders)
        # print('count sent train:', len(train))
        # print('count sent dev:', len(dev))
        # print('count sent test:', len(test))
        print('count sent all data:', len(train) + len(dev) + len(test))

        data_train = count_elements(train)
        data_dev = count_elements(dev)
        data_test = count_elements(test)

        print('non_unique_tokens_all_data:', len(data_train['tokens'] + data_dev['tokens'] + data_test['tokens']))
        print('unique_tokens_all_data:', len(set(data_train['unique_tokens'] + data_dev['unique_tokens'] + data_test['unique_tokens'])))
        print('unique_UPOS_all_data:', len(set(data_train['unique_UPOS'] + data_dev['unique_UPOS'] + data_test['unique_UPOS'])))
        print('unique_XPOS_all_data:', len(set(data_train['unique_XPOS'] + data_dev['unique_XPOS'] + data_test['unique_XPOS'])))
        print('----------------------------')
        print('unique_symbols_all_data:', len(set(list(data_train['unique_symbols']) + list(data_dev['unique_symbols']) + list(data_test['unique_symbols']))))
        print('out of train symbols:', len(list(data_train['unique_symbols'] - data_dev['unique_symbols'] - data_test['unique_symbols'])))
        print('out of train symbols set:', list(data_train['unique_symbols'] - data_dev['unique_symbols'] - data_test['unique_symbols']))
        print('----------------------------')
        print('max_token_length_all_data:', max([data_train['max_token_length'], data_dev['max_token_length'], data_test['max_token_length']]))
        print('max_sent_length_all_data:', max([data_train['max_sent_length'], data_dev['max_sent_length'], data_test['max_sent_length']]))
        print('avg_token_length_all_data:',
              round(sum([len(t) for t in data_train['tokens'] + data_dev['tokens'] + data_test['tokens']])
              / len(data_train['tokens'] + data_dev['tokens'] + data_test['tokens']), 1))
        print('avg_sent_length_train:',
              round(sum([len(s) for s in train + dev + test])
              / (len(train) + len(dev) + len(test)), 1))
        print('----------------------------')
        print('----------------------------')
        # print('unique_symbols_train:', len(data_train['unique_symbols']))
        # print('unique_symbols_dev:', len(data_dev['unique_symbols']))
        # print('unique_symbols_test:', len(data_test['unique_symbols']))
        # print('----------------------------')
        # print('max_token_length_train:', data_train['max_token_length'])
        # print('max_token_length_dev:', data_dev['max_token_length'])
        # print('max_token_length_test:', data_test['max_token_length'])
        # print('max_sent_length_train:', data_train['max_sent_length'])
        # print('max_sent_length_dev:', data_dev['max_sent_length'])
        # print('max_sent_length_test:', data_test['max_sent_length'])
        # print('----------------------------')
        # print('avg_token_length_train:', data_train['avg_token_length'])
        # print('avg_token_length_dev:', data_dev['avg_token_length'])
        # print('avg_token_length_test:', data_test['avg_token_length'])
        # print('avg_sent_length_train:', data_train['avg_sent_length'])
        # print('avg_sent_length_dev:', data_dev['avg_sent_length'])
        # print('avg_sent_length_test:', data_test['avg_sent_length'])
        # print('----------------------------')
