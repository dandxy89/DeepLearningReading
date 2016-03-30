import logging
import os
import random
from collections import OrderedDict

import numpy as np


class QADataset(object):

    def __init__(self, data_path, vocab_file,
                 n_entities, need_sep_token, **kwargs):
        self.provides_sources = ('context', 'question', 'answer', 'candidates')
        self.path = data_path
        self.vocab = ['@entity%d' % i for i in range(n_entities)] + \
                     [w.rstrip('\n') for w in open(vocab_file)] + \
                     ['<UNK>', '@placeholder'] + \
            (['<SEP>'] if need_sep_token else [])
        self.n_entities = n_entities
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {w: i + 1 for i, w in enumerate(self.vocab)}

    def to_word_id(self, w, cand_mapping):
        if w in cand_mapping:
            return cand_mapping[w]
        elif w[:7] == '@entity':
            raise ValueError("Unmapped entity token: %s" % w)
        elif w in self.reverse_vocab:
            return self.reverse_vocab[w] + 1
        else:
            return self.reverse_vocab['<UNK>'] + 1

    def to_word_ids(self, s, cand_mapping):
        return np.array([self.to_word_id(x, cand_mapping)
                         for x in s.split(' ')], dtype=np.int32)

    def get_data(self, state=None, request=None):
        if request is None or state is not None:
            raise ValueError(
                "Expected a request (name of a question file) and no state.")

        lines = [l.rstrip('\n')
                 for l in open(os.path.join(self.path, request))]

        ctx = lines[2]
        q = lines[4]
        a = lines[6]
        cand = [s.split(':')[0] for s in lines[8:]]

        entities = range(self.n_entities)
        while len(cand) > len(entities):
            logging.warning(
                "Too many entities (%d) for question: %s, using duplicate entity identifiers" %
                (len(cand), request))
            entities = entities + entities
        random.shuffle(entities)
        cand_mapping = {t: k for t, k in zip(cand, entities)}

        ctx = self.to_word_ids(ctx, cand_mapping)
        q = self.to_word_ids(q, cand_mapping)
        cand = np.array([self.to_word_id(x, cand_mapping)
                         for x in cand], dtype=np.int32)
        a = np.int32(self.to_word_id(a, cand_mapping))

        if not a < self.n_entities:
            raise ValueError("Invalid answer token %d" % a)
        if not np.all(cand < self.n_entities):
            raise ValueError("Invalid candidate in list %s" % repr(cand))
        if not np.all(ctx < self.vocab_size):
            raise ValueError(
                "Context word id out of bounds: %d" % int(
                    ctx.max()))
        if not np.all(ctx >= 0):
            raise ValueError("Context word id negative: %d" % int(ctx.min()))
        if not np.all(q < self.vocab_size):
            raise ValueError(
                "Question word id out of bounds: %d" % int(
                    q.max()))
        if not np.all(q >= 0):
            raise ValueError("Question word id negative: %d" % int(q.min()))

        return (ctx, q, a, cand)


def import_glove_line(line):
    partition = line.partition(' ')
    return partition[0], np.fromstring(partition[2], sep=' ')


def import_glove(filename, premade=None, filter=None):
    word_map = premade
    with open(filename, "r") as f:
        for val, line in enumerate(f):
            print(val)
            head, vec = import_glove_line(line)
            if head in filter:
                word_map[head] = vec
    return word_map

# set parameters:
glove_file = 'glove.6B.100d.txt'
n_entities = 550
dataset = '/home/dan1/Desktop/Subversion/trunk/NewsAna' \
          'lytics/Q&A/deepmind-qa'
dataset_name = 'cnn'
data_path = os.path.join(dataset,
                         dataset_name,
                         "questions",
                         "mini_test")
vocab_file = '/home/dan1/Desktop/Subversion/trunk/NewsAnalytics' \
             '/Q&A/deepmind-qa/cnn/stats/training/vocab.txt'


# Dataset Importer
QA_dataset = QADataset(data_path=data_path,
                       vocab_file=vocab_file,
                       n_entities=n_entities,
                       need_sep_token=False)
ordered1 = OrderedDict()
for key, value in sorted(
    QA_dataset.reverse_vocab.iteritems(), key=lambda k_v: (
        k_v[1], k_v[0])):
    ordered1[key] = value
ordered_vec = ordered1.copy()

# Open the Glove File
ordered_vec = import_glove(filename=glove_file,
                           premade=ordered_vec,
                           filter=ordered_vec.keys())

for key, value in ordered_vec.items():
    if not isinstance(value, np.ndarray):
        ordered_vec[key] = np.zeros(shape=(100,))

import json
with open('word_mapping.json', 'w') as fp:
    json.dump(ordered1, fp)

array = np.array(ordered_vec.values())

import h5py

# h5f = h5py.File('embedding_data.h5', 'w')
# h5f.create_dataset('dataset_1', data=array)
# h5f.close()

h5f = h5py.File('embedding_data.h5', 'r')
b = h5f['dataset_1'][:]
h5f.close()
