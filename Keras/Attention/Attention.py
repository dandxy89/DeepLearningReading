from keras.layers.recurrent import Recurrent
from keras import activations, initializations
from keras import backend as K
import numpy as np


class LSTMAttentionLayer(Recurrent):

    def __init__(
            self,
            output_dim,
            init='glorot_uniform',
            inner_init='orthogonal',
            forget_bias_init='one',
            activation='tanh',
            inner_activation='hard_sigmoid',
            batch_size=64,
            feed_state=False,
            **kwargs):

        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.batch_size = batch_size
        self.feed_state = feed_state
        super(LSTMAttentionLayer, self).__init__(**kwargs)

    def set_previous(self, layer):
        self.previous = layer
        self.build()

    @property
    def output_shape(self):
        return (None, self.output_dim)

    def build(self):

        if self.stateful:
            self.reset_states()
        else:
            self.states = [None, None]

        self.W_s = self.init((self.output_dim, self.output_dim))
        self.W_t = self.init((self.output_dim, self.output_dim))
        self.W_a = self.init((self.output_dim, self.output_dim))
        self.w_e = K.zeros((self.output_dim,))

        self.W_i = self.init((2 * self.output_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = K.zeros((self.output_dim,))

        self.W_f = self.init((2 * self.output_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim,))

        self.W_c = self.init((2 * self.output_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = K.zeros((self.output_dim,))

        self.W_o = self.init((2 * self.output_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = K.zeros((self.output_dim,))

        self.trainable_weights = [self.W_s, self.W_t, self.W_a, self.w_e,
                                  self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o]

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'

        if hasattr(self, 'states'):
            K.set_value(
                self.states[0], np.zeros(
                    (self.batch_size, self.output_dim)))
            K.set_value(
                self.states[1], np.zeros(
                    (self.batch_size, self.output_dim)))
        else:
            self.states = [K.zeros((self.batch_size, self.output_dim)),
                           K.zeros((self.batch_size, self.output_dim))]

    def set_state(self, noise):
        K.set_value(self.states[0], noise)

    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        self.h_t = X[1]

        self.h_s = X[0]
        if self.feed_state:
            self.h_init = X[2]

        self.P_j = K.dot(self.h_s, self.W_s)

        if self.stateful and not train:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(self.h_s)
            if self.feed_state:
                initial_states[0] = self.h_init

        last_output, outputs, states = K.rnn(
            self.step, self.h_t, initial_states)

        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(X)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.output_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def step(self, x, states):

        P_t = K.dot(x, self.W_t)
        P_a = K.dot(states[0], self.W_a)
        sum3 = self.P_j + \
            P_t.dimshuffle((0, 'x', 1)) + P_a.dimshuffle((0, 'x', 1))
        E_kj = K.tanh(sum3).dot(self.w_e)
        Alpha_kj = K.softmax(E_kj)
        weighted = self.h_s * Alpha_kj.dimshuffle((0, 1, 'x'))
        a_k = weighted.sum(axis=1)
        m_k = K.T.concatenate([a_k, x], axis=1)

        x_i = K.dot(m_k, self.W_i) + self.b_i
        x_f = K.dot(m_k, self.W_f) + self.b_f
        x_c = K.dot(m_k, self.W_c) + self.b_c
        x_o = K.dot(m_k, self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(states[0], self.U_i))
        f = self.inner_activation(x_f + K.dot(states[0], self.U_f))
        c = f * states[1] + i * \
            self.activation(x_c + K.dot(states[0], self.U_c))
        o = self.inner_activation(x_o + K.dot(states[0], self.U_o))
        h = o * self.activation(c)

        return h, [h, c]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(LSTMAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
