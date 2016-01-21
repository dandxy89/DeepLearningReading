import keras.backend as K
from keras.models import model_from_json
import numpy as np
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data

# set parameters:
input_dim = 100

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

    X_test.astype(int)
    X_train.astype(int)
    y_test.astype(int)
    y_train.astype(int)

    return (X_train, y_train), (X_test, y_test)

# get data:
(X_train, y_train), (X_test, y_test) = _get_test_data()
model = model_from_json(open('model_architecture.json').read())
model.load_weights('model_weights.h5')

m_input = model.get_input(train=False)
m_output = model.get_output(train=False)
f = K.function([m_input], [m_output])

res = f([X_train])[0]