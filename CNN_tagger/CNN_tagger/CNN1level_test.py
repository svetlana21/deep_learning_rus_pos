import numpy as np
import logging
import random as rn
np.random.seed(1024)
rn.seed(1024)

import tensorflow as tf
tf.set_random_seed(1024)

from keras.models import *
from keras.callbacks import *
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from config.Config_worker import Config
from data_preparation.utils import *

import gc
gc.collect()


class ModelCNN1Test:
    """
    Mean of option "Task_type". Grammem: "Grammem_tag_Animacy", POS, All (all morphology properties.).
    """

    def __init__(self,
                 use_config=True,
                 corpora='UD_Russian-SynTagRus',
                 task_type='POS',
                 verbose=1,
                 class_index=3
                 ):

        if use_config:
            self.config = Config(model_type='cnn')
            self.task_type = self.config.get('Task_type')
            self.corpora = self.config.get('Corpora')
        else:
            self.task_type = task_type
            self.corpora = corpora

        logging.info('\nModel test for 1 level model.')
        logging.info("Task: {}".format(self.task_type))
        logging.info("Corpora: {}".format(self.corpora))
        logging.info("Label index: {}".format(str(class_index)))

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/%s/cnn/model_level_1/' % (self.corpora,))
        self.model_path = os.path.abspath(file_path + '/../tagger_models/')

        self.x_test = load_bin_data(self.data_path + '/x_test_cnn1level.pkl')
        if verbose == 1:
            print('x_test shape:', self.x_test.shape)
        self.y_test = load_grammatical_cat_model1(self.data_path, self.task_type, verbose=1)

        self.estimation = 0

    def testing(self, verbose=1):
        """
        Model testing.
        :return:
        """

        model = load_model(self.model_path + '/cnn_1level_model_%s_%s.pkl' % (self.corpora, self.task_type,))
        pr = model.predict(self.x_test, verbose=verbose)
        fyh, fpr = preparation_data_to_score_model1(self.y_test, pr)
        logging.info("Testing sklearn acc: {}".format(str(accuracy_score(fyh, fpr))))
        logging.info("Testing sklearn f1_score: {}".format(str(f1_score(fyh, fpr, average='weighted'))))
        logging.info("Testing sklearn cohen_kappa_score: {}\n".format(str(cohen_kappa_score(fyh, fpr))))
        self.save_classes(fpr)
        self.estimation = accuracy_score(fyh, fpr)
        del (fyh, fpr, model, self.x_test, self.y_test, pr)

    def save_classes(self, data):
        save_binary(data, self.data_path + '/results/cnn1_level_marks_%s.pkl' % (self.task_type,))
