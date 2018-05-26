import os
import logging
import time
import uuid

from data_preparation.DataPrepareCNN1level import DataCNN1Level
from CNN_tagger.CNN1level_train import CNN1levelTagger
from CNN_tagger.CNN1level_test import ModelCNN1Test

from data_preparation.DataPrepareCNN2level import DataCNN2Level
from CNN_tagger.CNNProbEmbeddings import CNNProbEmbeddings
from CNN_tagger.CNN2level_train import CNN2levelTagger
from CNN_tagger.CNN2level_test import ModelCNN2Test


def init_logging():
    fmt = logging.Formatter('%(asctime)-15s %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    log_file_name = time.strftime("%Y_%m_%d-%H_%M_%S-") + str(uuid.uuid4())[:8] + '.txt'
    logging.info('Logging to {}'.format(log_file_name))
    logfile = logging.FileHandler(os.path.join(log_dir_name, log_file_name), 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)


data_path = 'data/'

# Вопроизводимость результатов https://github.com/keras-team/keras/issues/2280

init_logging()
logging.info("Start")

RESTART = 3  # number of restart to find the best random initialization

done_corpora = []

for index, folders in enumerate(os.listdir(data_path)):
    if not folders.endswith('_x') and not folders.endswith('_*') and folders not in done_corpora:

            for task_info in ( ('UPOS', 3), ('XPOS', 4), ):

                    max_accuracy = 0
                    best_start = None
                    best_model_level = None

                    for restarts in range(RESTART):

                        data_preparation_cnn1 = DataCNN1Level(
                            use_config=False,
                            corpora=folders,
                            task_type=task_info[0] + '_' + str(restarts),
                            class_index=task_info[1],
                            dev=True,
                            verbose=1)

                        cnn1_tagger = CNN1levelTagger(
                            use_config=False,
                            corpora=folders,
                            task_type=task_info[0] + '_' + str(restarts),
                            class_index=task_info[1],
                            verbose=1,
                            batch_size=512,
                            epoch=300,
                            dev=True
                        )
                        cnn1_tagger.network_initialization(verbose=0)
                        cnn1_tagger.training(verbose=1, dev=False)

                        test_cnn1 = ModelCNN1Test(
                            use_config=False,
                            corpora=folders,
                            task_type=task_info[0] + '_' + str(restarts),
                            verbose=1,
                            class_index=task_info[1]
                        )
                        test_cnn1.testing(verbose=0)

                        if test_cnn1.estimation > max_accuracy:
                            max_accuracy = test_cnn1.estimation
                            best_start = restarts
                            best_model_level = 1

                        # ------------------------------------------------------------------------------------

                        CNNProbEmbeddings(
                            use_config=False,
                            corpora=folders,
                            task_type=task_info[0] + '_' + str(restarts),
                            class_index=task_info[1],
                            dev=True,
                            verbose=1,
                            prob_cnn_emb_layer_name="dense_3"
                        )

                        data_preparation_cnn2 = DataCNN2Level(
                            use_config=False,
                            corpora=folders,
                            task_type=task_info[0] + '_' + str(restarts),
                            class_index=task_info[1],
                            dev=True,
                            verbose=1,
                            prob_cnn_emb_layer_name="dense_3")

                        cnn2_tagger = CNN2levelTagger(
                            use_config=False,
                            corpora=folders,
                            task_type=task_info[0] + '_' + str(restarts),
                            class_index=task_info[1],
                            verbose=1,
                            batch_size=60,
                            epoch=300,
                            dev=True
                        )
                        cnn2_tagger.network_initialization(verbose=0)
                        cnn2_tagger.training(verbose=1, dev=False)

                        test_cnn2 = ModelCNN2Test(
                            use_config=False,
                            corpora=folders,
                            task_type=task_info[0] + '_' + str(restarts),
                            verbose=1,
                            class_index=task_info[1]
                        )
                        test_cnn2.testing(verbose=0)

                        if test_cnn2.estimation > max_accuracy:
                            max_accuracy = test_cnn2.estimation
                            best_start = restarts
                            best_model_level = 2

                        logging.info("*" * 100)

                    logging.info("-" * 100)
                    logging.info("Name corpora: {}".format(folders))
                    logging.info("Task: {}".format(task_info[0]))
                    logging.info("Best restart acc: {}".format(str(max_accuracy)))
                    logging.info("Best restart #: {}".format(str(best_start)))
                    logging.info("Best model level #: {}".format(str(best_model_level)))
                    logging.info("-" * 100)

logging.info("End")
