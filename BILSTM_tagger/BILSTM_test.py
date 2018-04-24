# Keras==2.0.3 tensorflow=1.1.0rc1
import pickle

from config.Config_worker import Config


from keras.optimizers import *
from keras.models import *
from keras.callbacks import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import seaborn
import gc; gc.collect()
from keras import backend as K


class ModelTest:
    """
    Mean of option "Task_type". Grammem: "Grammem_tag_Animacy", POS, All (all morphology properties.).
    """
    def __init__(self):
        self.config = Config(model_type='bilstm')
        self.task_type = self.config.get('Task_type')

        print('#' * 100)
        print('Task:', self.task_type)
        print('Corpora:', self.config.get('Corpora'))
        print('#' * 100)

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(file_path + '/../data/%s/' % (self.config.get('Corpora'),))
        self.model_path = os.path.abspath(file_path + '/../tagger_models/')

        self.x_test = self.load_binary_data(self.data_path + '/x_test.pkl')
        print('X test shape:', self.x_test.shape)

        self.y_test = self.load_grammatical_cat()
        self.model = None

    def load_grammatical_cat(self):
        """
        Loading y data for each grammatical category.
        """

        labels = []
        for files in os.listdir(self.data_path):
            if 'y_test_Grammem_tag' in files:
                if self.task_type != 'All': 
                    if self.task_type in files:
                        labels.append(
                            [files.split('.')[0].split('y_test_Grammem_tag_')[1], 
                            self.load_binary_data(self.data_path + '/' + files)])
                else:
                    labels.append(
                        [files.split('.')[0].split('y_test_Grammem_tag_')[1], 
                        self.load_binary_data(self.data_path + '/' + files)])
        return sorted(labels)

    def load_binary_data(self, path_to_data):
        """
        Data load.
        :return:
        """

        with open(path_to_data, 'rb') as f:
            data = pickle.load(f)
        return data

    def testing(self):
        """
        Model testing.
        :return:
        """

        test_data = [self.x_test, self.x_test]
        model = load_model(self.model_path + '/model_%s_%s.pkl' % (self.config.get('Corpora'),
                                                                   self.task_type,))

        print('Model load.')
        pr = model.predict(test_data, verbose=1)
        for i in range(len(self.y_test)):
            print('*' * 100)
            print('Testing category:', self.y_test[i][0])
            fyh, fpr = self.preparetion_data_to_score(self.y_test[i][1], pr[i])
            print('Testing sklearn: acc:', accuracy_score(fyh, fpr))
            print('Testing sklearn: f1_score:', f1_score(fyh, fpr, average='weighted'))
            # The Kappa or Cohen’s kappa is the classification accuracy normalized by the imbalance of the classes in the data.
            print('Testing sklearn: cohen_kappa_score:', cohen_kappa_score(fyh, fpr))
            del (fyh, fpr)

    def preparetion_data_to_score(self, yh, pr):
        """    
        yh = [array([ 57, 156, 300, 120, 306,  31, 148,  38,  70,  36, 196, 306, 200,
                31, 116, 275]), array([ 36,  35,  35, 294, 109, 275])]
        ypr = [array([  0,   0,   0,   0,   0, 120, 120, 120, 120, 120, 120, 120, 120,
               120, 120, 120]), array([0, 0, 0, 0, 0, 0])]
    
        fyh = [57, 156, 300, 120, 306, 31, 148, 38, 70, 36, 196, 306, 200, 31, 116, 275, 36, 35, 35, 294, 109, 275]
        fpr = [0, 0, 0, 0, 0, 0, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 0, 0, 0, 0, 0, 275]
    
        :param yh:
        :param pr:
        :return:
        """

        yh = yh.argmax(2)
        pr = [list(np.argmax(el, axis=1)) for el in pr]
        # с какого элемента начинаются не 0; напрмиер: [37, 47];
        coords = [np.where(yhh > 0)[0][0] for yhh in yh]
        # оставляем только не 0 элементы;
        yh = [yhh[co:] for yhh, co in zip(yh, coords)]
        # по границам y оставялем не 0 в предсказании;
        ypr = [prr[co:] for prr, co in zip(pr, coords)]
        # конкатенация массивов для сравнения;
        fyh = [c for row in yh for c in row]
        fpr = [c for row in ypr for c in row]
        return fyh, fpr

if __name__ == '__main__':
    lstm_tagger = ModelTest()
    lstm_tagger.testing()
