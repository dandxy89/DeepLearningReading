from keras.models import Graph, Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.regularizers import l2
from keras.layers.recurrent import LSTM
from Attention import LstmAttentionLayer
from common import make_fixed_embeddings


def basic_model(
        embed_size=300,
        hidden_size=100,
        lr=0.001,
        dropout=0.0,
        reg=0.0):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, embed_size)))
    #model.add(Dropout(dropout, input_shape = (None, embed_size)))
    model.add(Dropout(dropout))
    model.add(LSTM(hidden_size))
    model.add(Dense(3, W_regularizer=l2(reg)))
    model.add(Dropout(dropout))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr))

    return model


def attention_model(hidden_size, glove):
    premise_layer = LSTM(
        output_dim=hidden_size,
        return_sequences=True,
        inner_activation='sigmoid')
    hypo_layer = LSTM(
        output_dim=hidden_size,
        return_sequences=True,
        inner_activation='sigmoid')
    attention = LstmAttentionLayer(hidden_size, inner_activation='sigmoid')

    graph = Graph()
    graph.add_input(name='premise_input', input_shape=(None,), dtype='int')
    graph.add_node(
        make_fixed_embeddings(
            glove,
            None),
        name='prem_word_vec',
        input='premise_input')
    graph.add_node(premise_layer, name='premise', input='prem_word_vec')
    graph.add_input(name='hypo_input', input_shape=(None,), dtype='int')
    graph.add_node(
        make_fixed_embeddings(
            glove,
            None),
        name='hypo_word_vec',
        input='hypo_input')
    graph.add_node(hypo_layer, name='hypo', input='hypo_word_vec')
    graph.add_node(
        attention,
        name='attention',
        inputs=[
            'premise',
            'hypo'],
        merge_mode='join')
    graph.add_node(Dense(3), name='dense', input='attention')
    graph.add_node(Activation('softmax'), name='softmax', input='dense')
    graph.add_output(name='output', input='softmax')
    graph.compile(
        loss={
            'output': 'categorical_crossentropy'},
        optimizer='adam')

    return graph
