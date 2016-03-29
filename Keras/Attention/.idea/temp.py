import logging
import os
import random
from random import randint

from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Graph
from keras.optimizers import RMSprop
from keras.regularizers import l2


