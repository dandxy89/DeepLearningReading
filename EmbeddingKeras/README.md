### Using Word2vec vectors into a RNN Model

A successful attempt to incorporate Word Vectors rather than using an Embedding class for an NLP (Natural Language Processing) task. 

This method passes a weight matrix to the Embedding layer. By example:

Create an `index_dict` that maps all the words in your dictionary to indices from 1 to n_symbols, where 0 is reserved for the masking. So, an example `index_dict` is the following:

`{
 'yellow': 1,
 'four': 2,
 'woods': 3,
...
}`

And you also have a dictionary called `word_vectors` that maps words to vectors like so:

`{
 'yellow': array([0.1,0.5,...,0.7]),
 'four': array([0.2,1.2,...,0.9]),
...
}`

#### Example Code:

*Embedding*

`vocab_dim = 300 # dimensionality of your word vectors`
`n_symbols = len(index_dict) + 1 # adding 1 to account for 0th index (for masking)`
`embedding_weights = np.zeros((n_symbols+1,vocab_dim))`
`for word,index in index_dict.items():`
`    embedding_weights[index,:] = word_vectors[word]`

*Assemble the Model*

`model = Sequential() # or Graph or whatever`
`model.add(Embedding(output_dim=rnn_dim, input_dim=n_symbols + 1, mask_zero=True, weights=[embedding_weights])) # note you have to put embedding weights in a list by convention`
`model.add(LSTM(dense_dim, return_sequences=False))`
`model.add(Dropout(0.5))`
`model.add(Dense(n_symbols, activation='softmax')) # for this is the architecture for predicting the next word, but insert your own here`
