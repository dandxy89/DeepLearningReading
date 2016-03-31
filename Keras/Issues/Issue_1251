''' Issue: https://github.com/fchollet/keras/issues/1251
    Modified the examples/imdb_bidirectional_lstm.py

    Multi-layer Bi-LSTM

    Model 1 : As proposed on Dec 14, 2015

              Test accuracy:

                        0.83179999999999998

    Model 2 : mossaab0: @dandxy89 I can't seem to be able to compile your code.
                        Can you compile the following?

              Other:

                        Restarted Python Console between comparisons

              Test accuracy:

                        0.81440000000000001

              Modifications!:

                        Added Parameters from script to Embedding and Input Layer

'''
import numpy as np
np.random.seed(607)

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.models import Graph
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 100
batch_size = 32
embedding_size = 128

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      test_split=0.2)

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Model 1 :
model = Graph()
# Input Layer
model.add_input(name='input', input_shape=(maxlen,),
                dtype=int)
# Embedding Layer
model.add_node(Embedding(max_features, embedding_size, input_length=maxlen),
               name='embedding',
               input='input')
# First Hidden Lyaer with Bi-LSTM
model.add_node(LSTM(128, return_sequences=True),
               name='f1', input='embedding')
model.add_node(LSTM(128, go_backwards=True, return_sequences=True),
               name='b1',
               input='embedding')
# ADD DROPOUT?
# Second Hidden Layer with Bi-LSTM
model.add_node(LSTM(128),
               name='f2',
               inputs=['f1', 'b1'],
               merge_mode='sum')

model.add_node(LSTM(128, go_backwards=True),
               name='b2',
               inputs=['f1', 'b1'],
               merge_mode='sum')
# ADD DROPOUT / Merge?
model.add_node(Dense(1, activation='sigmoid'),
               name='sigmoid',
               inputs=['f2', 'b2'])

model.add_output(name='output',
                 input='sigmoid')

model.compile('adam',
              {'output': 'binary_crossentropy'})

model.fit({'input': X_train, 'output': y_train},
          batch_size=batch_size,
          nb_epoch=1,
          show_accuracy=True)
acc = accuracy(y_test,
               np.round(np.array(model.predict({'input': X_test},
                                               batch_size=batch_size)['output'])))
print('Test accuracy:', acc)

# Model 2 :
model = Graph()
# Input Layer
# model.add_input(name='input', input_shape=(128,), dtype=int)
model.add_input(name='input', input_shape=(maxlen,),
                dtype=int)
# Embedding Layer
# model.add_node(Embedding(128, 128, input_length=128), name='embedding', input='input')
model.add_node(Embedding(max_features, embedding_size, input_length=maxlen),
               name='embedding',
               input='input')
# First Hidden Layer with Bi-LSTM
model.add_node(LSTM(128, return_sequences=True),
               name='f1',
               input='embedding')
model.add_node(LSTM(128, go_backwards=True, return_sequences=True),
               name='b1',
               input='embedding')
# ADD DROPOUT / Merge?
# Second Hidden Layer with Bi-LSTM
model.add_node(LSTM(128),
               name='f2',
               inputs=['f1', 'b1'],
               merge_mode='sum')
model.add_node(LSTM(128, go_backwards=True),
               name='b2',
               inputs=['f1', 'b1'],
               merge_mode='sum')

model.add_node(Dense(1, activation='sigmoid'),
               name='sigmoid',
               inputs=['f2', 'b2'])

model.add_output(name='output',
                 input='sigmoid')
model.compile('adagrad',
              {'output': 'binary_crossentropy'})

model.fit({'input': X_train, 'output': y_train},
          batch_size=batch_size,
          nb_epoch=1,
          show_accuracy=True)
acc = accuracy(y_test,
               np.round(np.array(model.predict({'input': X_test},
                                               batch_size=batch_size)['output'])))
print('Test accuracy:', acc)
