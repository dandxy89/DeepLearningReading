import numpy as np
from keras.layers.embeddings import Embedding


def make_fixed_embeddings(glove, seq_len):
    glove_mat = np.array(glove.values())
    return Embedding(
        input_dim=glove_mat.shape[0],
        output_dim=glove_mat.shape[1],
        weights=[glove_mat],
        trainable=False,
        input_length=seq_len)
