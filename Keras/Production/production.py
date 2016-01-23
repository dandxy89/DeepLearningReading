import keras.backend as K
from keras.models import model_from_json
import numpy as np
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data


def _get_test_data():
    np.random.seed(1234)

    train_samples = 2000
    test_samples = 500

    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples,
                                                         nb_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         nb_class=4)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    X_test = np.abs(X_test.astype(int))
    X_train = np.abs(X_train.astype(int))
    y_test = np.abs(y_test.astype(int))
    y_train = np.abs(y_train.astype(int))

    return (X_train, y_train), (X_test, y_test)


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]


def slice_X(X, start=None, stop=None):
    '''
    '''
    if type(X) == list:
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return X[start]
        else:
            return X[start:stop]


def predict(model, data, batch_n):
    outs = []
    size = data.shape[0]
    batches = make_batches(size=size, batch_size=batch_n)
    index_array = np.arange(size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        ins_batch = slice_X(data, batch_ids)

        batch_outs = model([ins_batch])
        if type(batch_outs) != list:
            batch_outs = [batch_outs]
        if batch_index == 0:
            for batch_out in batch_outs:
                shape = (size,) + batch_out.shape[1:]
                outs.append(np.zeros(shape))

        for i, batch_out in enumerate(batch_outs):
            outs[i][batch_start:batch_end] = batch_out
    return outs[0]


def import_model():
    model = model_from_json(open('model_architecture.json').read())
    model.load_weights('model_weights.h5')

    m_input = model.get_input(train=False)
    m_output = model.get_output(train=False)

    f = K.function([m_input], [m_output])

    return f


# set parameters:
input_dim = 100
model = import_model()

# get data:
(X_train, y_train), (X_test, y_test) = _get_test_data()

# TEST THIS X_new.shape == (100,)
X_new = np.array([i for i in range(100)])
X_new2 = np.vstack((X_new, X_new))

#x = predict(model=model,
#             data=X_new,
#             batch_n=100)

#assert x.shape[0] == X_new.shape[0]

x = predict(model=model,
             data=X_new2,
             batch_n=100)

assert x.shape[0] == X_new2.shape[0]

x = predict(model=model,
             data=X_train,
             batch_n=100)

assert x.shape[0] == X_train.shape[0]