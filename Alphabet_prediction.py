# -*- coding: utf-8 -*-
""" Attempt at making a Stateful RNN for Predicting the
    Alphabet

    FAQ: http://keras.io/faq/#how-can-i-use-stateful-rnns
"""

import string
import numpy as np
from random import random, randrange
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense


class Alphabet(object):

    def __init__(self):
        self.letters = string.ascii_lowercase
        self.mapping = {let: i for i, let in enumerate(self.letters)}
        self.combined = []

    def get_value(self):
        return self.mapping.values()[randrange(0, 26)]

    def choice(self, prev=None):
        if random() <= 0.3:
            return 0
        else:
            if prev is None:
                return self.get_value()
            else:
                new_value = prev[0] + 1
                if new_value > 27:
                    return 0
                else:
                    return new_value

    def create_list(self, n_examples):
        self.n_examples = n_examples
        for ex in range(self.n_examples):
            new_example = []
            prev = None
            for i in range(26):
                new_example.append(self.choice(prev))
                prev = new_example[-1:]
                if new_example[-2:] == [0, 0]:
                    break
            self.combined.append(new_example)

    def split(self, ratio, n_examples):
        self.create_list(n_examples)
        split_ratio = int(self.n_examples * split_ratio)
        return np.array(self.combined[:split_ratio]), np.array(self.combined[split_ratio:])


# set parameters:
examples = 100000
ratio = 0.7

# Create datasets
Data = Alphabet()
X_train, X_test = Data.split(ratio, examples)
y_train = np.array([[Data.choice(example[-1:])] for example in X_train])
y_test = np.array([[Data.choice(example[-1:])] for example in X_test])
assert(X_train.shape[0] == y_train.shape[0])

# Prepare for Keras Model
X_train = sequence.pad_sequences(X_train, 26)
X_test = sequence.pad_sequences(X_test, 26)

# Define the Model
model = Sequential()
model.add(LSTM(32, batch_input_shape=(32, 10, 16), 
               stateful=True))
model.add(Dense(16, activation='softmax'))

# Compile Model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')

