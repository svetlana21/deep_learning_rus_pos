from keras.models import *
from keras.preprocessing.sequence import *

from config.Config_worker import Config
from data_preparation.utils import *


def data_prepare(x_set, name,  word2ind_with_adding, max_sent_length):
    x_enc = []
    count_new_tokens = 0
    count_tokens = 0
    for s in x_set:
        s_enc = []
        for t in s:
            try:
                s_enc.append(word2ind_with_adding[t])
                count_tokens += 1
            except KeyError:
                # в матрицу идет нулевой - emb_vocab.append(zero_vector)
                # потом идет 1 - emb_vocab.append(tokens_tune_vectors['_null_'])
                s_enc.append(1)
                count_new_tokens += 1
        x_enc.append(s_enc)
    x = pad_sequences(x_enc, maxlen=max_sent_length, value=1)
    print('x_shape: %s;' % (name,), x.shape)
    print('count_new_tokens:', count_new_tokens)
    print('count_tokens:', count_tokens)
    return x


label_index = {
    'UPOS': 3,
    'XPOS': 4
}


# ----------------------------------------------------------------------------------------------------------------------

config_models = Config(model_type='models')
config_language_id = Config(model_type='tracks')

model_dir = '../tagger_models/'
test_files_udipipe_dir = '../data/conll2017_x/ud-test-v2.0-conll2017/input/conll17-ud-test-2017-05-09/'
data_path = os.path.abspath(os.path.split(os.path.abspath(__file__))[0] + '/../data/')

for corpora_name in config_models.get("models"):

    if corpora_name == 'UD_Russian':

        for tag_types in config_models.get("models")[corpora_name]:

            best_restart = config_models.get("models")[corpora_name][tag_types].get('Best restart #')
            model_level = config_models.get("models")[corpora_name][tag_types].get('Best model level #')

            if model_level != 'None' and model_level != 'INPROGRESS':

                model = None
                x_test_udipipe_tokenize_data = None
                tokens_for_prediction = None
                ind2label = None
                max_object_length = None
                predictions = None

                print('\nLanguage: %s; Tag type: %s; best_restart: %s; model_level: %s' % (corpora_name, tag_types, str(best_restart), str(model_level)))
                pattern = 'cnn_%slevel_model_%s_%s_%s.pkl' % (str(model_level), corpora_name, tag_types, str(best_restart))
                for folders in os.listdir(model_dir):
                    if pattern == folders:
                        model = load_model(model_dir + pattern)
                        print(model)

                for test_files in os.listdir(test_files_udipipe_dir):
                    if test_files == config_language_id.get('languages')[corpora_name]:
                        print(test_files_udipipe_dir + test_files)

                        sent_test_all = load_data(test_files_udipipe_dir + test_files, corpora_type=corpora_name, save_sent=True)

                        label2ind_with_adding = load_bin_data(data_path + '/' + corpora_name + '/cnn/model_level_%s/y_label2ind_cnn%slevel_%s' % (
                                model_level, model_level, tag_types + '_' + str(best_restart)))

                        char_emb_feature = load_bin_data(data_path + '/' + corpora_name + '/cnn/model_level_%s/char_emb_cnn%s_feature_data_%s.pkl' % (
                                model_level, model_level, tag_types + '_' + str(best_restart),))

                        # --------------------------------------------------------------------------------------------------

                        if model_level == 2:

                            ind2label = {label2ind_with_adding[el]: el for el in label2ind_with_adding}

                            ind2object = char_emb_feature['word2ind']
                            max_object_length = char_emb_feature['max_sent_length']

                            tokens_for_prediction = [[t[1] for t in s if "#" not in t[0]] for s in sent_test_all]

                            print('count_ind2object:', len(ind2object))
                            x_test_udipipe_tokenize_data = data_prepare(
                                tokens_for_prediction,
                                corpora_name,
                                ind2object,
                                max_object_length)

                        else:
                            ind2object = char_emb_feature['ind2symbol']
                            max_object_length = char_emb_feature['max_token_length']

                # ---------------------------------------------------------------------------------------------------------

                prediction = model.predict(x_test_udipipe_tokenize_data)

                if model_level == 2:

                    fake_prediction = pad_sequences(
                        [[2 for i in range(len(s))] for s in tokens_for_prediction],
                        maxlen=max_object_length,
                        value=1)

                    _, labels_2indexes = load_grammatical_cat_model2(
                        data_path + '/' + corpora_name + '/cnn/model_level_%s/' % (model_level,),
                        tag_types + '_' + str(best_restart),
                        verbose=1)

                    fake_prediction = transform2categorical(fake_prediction, labels_2indexes)
                    _, predictions = preparation_data_to_score_model2(fake_prediction, prediction, sent_borders=True)

                    predictions = [[ind2label.get(t) if t in ind2label else 'UNKNOWN' for t in s] for s in predictions]

                else:
                    pass

                del (model, x_test_udipipe_tokenize_data)

                # ---------------------------------------------------------------------------------------------------------

                new_sents = []
                for i in range(len(sent_test_all)):
                    new_sent = [] + [t for t in sent_test_all[i] if '#' in t[0]]
                    words = [t for t in sent_test_all[i] if '#' not in t[0]]
                    for ti in range(len(words)):
                        new_sent.append(words[ti][:label_index[tag_types]] + [predictions[i][ti]] + words[ti][label_index[tag_types]+1:])
                    new_sents.append(new_sent)

                f = open('../marking/predictions/result_%s_%s_%s_%s.conllu' % (
                    model_level, best_restart, tag_types, corpora_name), "w", encoding='utf-8')
                for s in new_sents:
                    for t in s:
                        f.write('\t'.join(t) + '\n')
                    f.write('\n')
                f.close()
