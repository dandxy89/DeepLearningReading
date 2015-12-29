# -*- coding: utf-8 -*-
''' This example demonstrates the use of Word2vec vectors for text
    classification.

    A simple implementation of using the embedding layer in Keras to embed
    Word2vec (ndim=300) vectors into a LSTM Network.

    Dataset:
        test-neg.txt: 12500 negative movie reviews from the test data
        test-pos.txt: 12500 positive movie reviews from the test data
        train-neg.txt: 12500 negative movie reviews from the training data
        train-pos.txt: 12500 positive movie reviews from the training data

    Will require Gensim (Text Processing) within this script.

    Caution:
        Masking is Theano-only for the time being. Not yet implemented in TF
        
    Results after 2 epochs:
        Validation Accuracy: 0.8485
                       Loss: 0.3442
        

'''


import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
np.random.seed(1337)  # For Reproducibility


# set parameters:
vocab_dim = 300
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 30
window_size = 5
batch_size = 32
n_epoch = 2
cpu_count = multiprocessing.cpu_count()
data_locations = {'./Data/test-neg.txt': 'TEST_NEG',
                  './Data/test-pos.txt': 'TEST_POS',
                  './Data/train-neg.txt': 'TRAIN_NEG',
                  './Data/train-pos.txt': 'TRAIN_POS'}


def import_tag(datasets=None):
    ''' Imports the datasets into one of two dictionaries

        Dicts:
            train & test

        Keys:
            values > 12500 are "Negative" in both Dictionaries

        '''
    if datasets is not None:
        train = {}
        test = {}
        for k, v in datasets.items():
            with open(k) as fpath:
                data = fpath.readlines()
            for val, each_line in enumerate(data):
                if v.endswith("NEG") and v.startswith("TRAIN"):
                    train[val] = each_line
                elif v.endswith("POS") and v.startswith("TRAIN"):
                    train[val + 12500] = each_line
                elif v.endswith("NEG") and v.startswith("TEST"):
                    test[val] = each_line
                else:
                    test[val + 12500] = each_line
        return train, test
    else:
        print('Data not found...')


def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [document.lower().replace('\n', '').split() for document in text]
    return text


def create_dictionaries(train=None,
                        test=None,
                        model=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (train is not None) and (model is not None) and (test is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(data):
            ''' Words become integers
            '''
            for key in data.keys():
                txt = data[key].lower().replace('\n', '').split()
                new_txt = []
                for word in txt:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data[key] = new_txt
            return data
        train = parse_dataset(train)
        test = parse_dataset(test)
        return w2indx, w2vec, train, test
    else:
        print('No data provided...')


print('Loading Data...')
train, test = import_tag(datasets=data_locations)
combined = train.values() + test.values()

print('Tokenising...')
combined = tokenizer(combined)

print('Training a Word2vec model...')
model = Word2Vec(size=vocab_dim,
                 min_count=n_exposures,
                 window=window_size,
                 workers=cpu_count,
                 iter=n_iterations)
model.build_vocab(combined)
model.train(combined)

print('Transform the Data...')
index_dict, word_vectors, train, test = create_dictionaries(train=train,
                                                            test=test,
                                                            model=model)

print('Setting up Arrays for Keras Embedding Layer...')
n_symbols = len(index_dict) + 1  # adding 1 to account for 0th index
embedding_weights = np.zeros((n_symbols + 1, vocab_dim))
for word, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[word]

print('Creating Datesets...')
X_train = train.values()
y_train = [1 if value > 12500 else 0 for value in train.keys()]
X_test = test.values()
y_test = [1 if value > 12500 else 0 for value in test.keys()]

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Convert labels to Numpy Sets...')
y_train = np.array(y_train)
y_test = np.array(y_test)

print('Defining a Simple Keras Model...')
model = Sequential()  # or Graph or whatever
model.add(Embedding(output_dim=vocab_dim,
                    input_dim=n_symbols + 1,
                    mask_zero=True,
                    weights=[embedding_weights]))
model.add(LSTM(vocab_dim))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

print('Compiling the Model...')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode='binary')

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,
          validation_data=(X_test, y_test), show_accuracy=True)

print("Evaluate...")
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)

