from __future__ import print_function

import sys
import numpy as np
import random
import time

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

allChars= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
allChars *= 50

text = ''.join(allChars)

def chunks(string, length):
    return [string[0+i:length+i] for i in range(0, len(string), length)]


totalTimeSteps = 8 # The total number of characters in text must be larger than this
text = text[:len(text) // totalTimeSteps  * totalTimeSteps]  # Make text divisible by batch size
print('corpus length:', len(text))
chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
batches = chunks(text, totalTimeSteps)
batchSize = len(batches)
print('batch size:' , batchSize)
featurelen = len(chars)
numOfPrevSteps = 1 # We are only looking at the most recent character each time. 

print('Formatting Data')

X = np.zeros([batchSize, totalTimeSteps , featurelen]) 
for b in range(len(batches)):
    for r in range(totalTimeSteps):
        currentChar = text[r + b*totalTimeSteps]
        X[b][r][char_indices[currentChar]] = 1
print('Formatted Data ',X)

i = 0
for matrix in X:
    cl = ''
    for row in matrix:
        mi = list(row).index(max(row))
        c = indices_char[mi]
        cl = cl + c
    i += 1
    print('batch ',i,cl)

print('Building model...')
model = Sequential()
model.add(LSTM(512 ,
               return_sequences=True,
               batch_input_shape=(batchSize, numOfPrevSteps , featurelen),
               stateful=True))
model.add(Dropout(0.2))
model.add(LSTM(512,
               return_sequences=False,
               stateful=True))
model.add(Dropout(0.2))
model.add(Dense( featurelen ))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop')
model.reset_states()

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

print('starting training')
num_epochs = 1000
X = X[:batchSize]
for e in range(num_epochs):
    print('epoch - ',e+1)
    startTime = time.time()
    for i in range(0,totalTimeSteps-1):
        model.train_on_batch(X[:, numOfPrevSteps*i:(i+1)*numOfPrevSteps, :],
                             np.reshape(X[:, (i+1)*numOfPrevSteps, :], (batchSize, featurelen))
                             ) # Train on guessing a single element based on the previous element
    model.reset_states()
    numberToGenerate = 100
    if (e+1)%100 == 0 or (e+1) == num_epochs:
        numberToGenerate = 200
        diversities = [.2,.5,1.0,1.2]
        startChar = random.choice(text)
        for diversity in diversities:
            next_char = startChar
            print('Generating with diversity - ',diversity,' and seed - ',next_char)
            for _ in range(numberToGenerate):
                C = np.zeros([batchSize,1,featurelen])
                C[0][0][char_indices[next_char]] = 1
                pred = model.predict(C)
                pred = pred[0] # We have to pass in an entire batch at a time, but we only care about the first since we are only generation one letter
                next_index = sample(pred, diversity)
                next_char = indices_char[next_index]
                sys.stdout.write(next_char)
                sys.stdout.flush()
            model.reset_states()
            print()
    totalTime = time.time() - startTime
    print('Completed epoch in ',totalTime,' seconds')
    print()
print('training complete')