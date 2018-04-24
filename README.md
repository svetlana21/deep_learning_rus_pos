## Research of Neural Network Effectiveness for a Morphological tagging for russian.
2017 Mikhail Domrachev


-------------------------------------------
### Table of Contents 

  * [Data](#data)
  * [Features](#features)
  * [Models](#models)
  * [Metrics](#metrics)
  * [Results](#results)
  * [Contact](#contact)


------------------------------------
### Data
<a name="data" ></a>

|Corpus       | Unique symbols | Tokens | Max_token_length | Unique words | Sentences | Min_sent_length | Max_sent_length |
|-------------|----------------|--------|------------------|--------------|-----------|-----------------|-----------------|
|GICR (train) |     99         | 815877 |        31        |   107646     |  62360    |        7        |       110       | 
|GICR (test)  |     97         | 270251 |        35        |    54262     |  20786    |        7        |        80       |       


------------------------------------
### Tasks
<a name="tasks" ></a>

1. Multi-target classification, POS task: identify part of the speech. 13 labels + 1 for 0.
2. Multi-target classification, Integr_morpho_tag task: identify set of morphological features. 264 labels + 1 for 0.
3. Multioutput-multiclass classification. 12 morphlogical categories.

------------------------------------
### Features
<a name="features" ></a>

We used Embedding layer and encoded sentences using the unique tokens dictionary.

 + Features based on char data. 
 
 We find the maximum token length and count unique symbols. We encode the token characters using one-hot encoding with the number of unique symbols as a parameter.
 1. We concatenate all of symbols vectors into 1.
 2. We using this data as is.

 + Features based on word embeddings.
 
We train word2vec model on combination of plain text data: 

- LiveJournal (https://github.com/dialogue-evaluation/morphoRuEval-2017/tree/LiveJournal).
- Fiction (https://github.com/dialogue-evaluation/morphoRuEval-2017/tree/librusec).
- Social nets (https://github.com/dialogue-evaluation/morphoRuEval-2017/tree/social_media).

With parameters: min_count=5, window=10, size=300, negative=10, skip-gram, negative sampling.

------------------------------------
### Models
<a name="Models" ></a>

------------------------------------
### BiLSTM
<a name="BiLSTM" ></a>

Siamese Bidirectional Long-Short Term Memory

![Иллюстрация к проекту](https://github.com/Ulitochka/LSTM_Tagger/blob/master/tagger_models/rnn_model_schema_gycrya_All.png)


------------------------------------
### Metrics
<a name="metrics" ></a>

Accuracy, f1_score, cohen_kappa.

*cohen_kappa*: The Kappa or Cohen’s kappa is the classification accuracy normalized by the imbalance of the classes in the data.

*Integr_morpho_tag*: In frame of the features tagging task, output classes consist of all the unique combinations of lexical and grammatical properties (except PoS) that exist in the train set, one-hot encoded. Such an approach allows to decrease the computational complexity of the model. However, there might emerge combinations not pre sented in the training set, and such examples could be classified incorrectly. 


------------------------------------
### Results
<a name="results" ></a>

Task               | siamese_bilstm     | crf (linear chain) | 
-------------------|--------------------|--------------------|
POS                | 97.3 / - / -       |   98.2 / 98.2 / -  | 
Integr_morpho_tag  | 94.8 / - / -       |   91.4 / -    / -  |
Case               | 95.5 / 95.5 / 92.7 |   95   / 90.2 / -  | 
Gender             | 96.6 / 96.6 / 94.2 |   97.8 / 96.2 / -  |
Number             | 97.8 / 97.8 / 96.3 |   98   / 97.3 / -  |
Tense              | 99.6 / 99.5 / 97.7 |   99.8 / 99.2 / -  |
Person             | 99.6 / 99.6 / 98.2 |   99.6 / 98   / -  |
VerbForm           | 99.6 / 99.6 / 98.7 |   99.8 / 99.3 / -  |
Mood               | 99.7 / 99.7 / 98.4 |   99.8 / 99.1 / -  |
Variant            | 99.7 / 99.7 / 91.4 |   99.8 / 94.7 / -  |
Degree             | 98.9 / 98.9 / 95.7 |   99.1 / 97.1 / -  | 
NumForm            | 99.9 / 99.9 / 99.7 |   99   / 99.8 / -  |
Voice              | 99.6 / 99.6 / 98.5 |   99   / 99.2 / -  |
Animacy            | 97.3 / 97.1 / 92.5 |   97.4 / 97.1 / -  |


------------------------------------
### Contact
<a name="contact"></a>

mikhail.al.domrachev@gmail.com




