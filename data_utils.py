import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)


class Vocab(object):

    def __init__(self, vocab_file, save_dir=None, min_freq=1, lower_case=True,
                 max_size=None):
        self.vocab_file = vocab_file
        self.save_dir = save_dir
        self.min_freq = min_freq
        self.lower_case = lower_case
        self.max_size = max_size
        self.sym2idx = {}
        self.idx2sym = []

        self.pad = '<pad>'
        self.unk = '<unk>'
        self.eos = '<eos>'

        self.build_vocab()

    def build_vocab(self):
        if not os.path.exists(self.vocab_file):
            raise ValueError('vocabulary file not exist!')

        if self.vocab_file.strip().endswith('pkl'):
            print('vocabulary loaded from `{}`'.format(self.vocab_file))
            with open(self.vocab_file, 'rb') as f:
                self.idx2sym = pickle.load(f)
            self.sym2idx = dict(zip(self.idx2sym, range(len(self.idx2sym))))
            return

        counter = defaultdict(int)
        with open(self.vocab_file, 'r', encoding='utf8') as f:
            for line in f:
                for symbol in line:
                    if self.lower_case:
                        symbol = symbol.lower()
                    counter[symbol] += 1

        counter[self.pad] = int(1e100)
        counter[self.unk] = int(1e100 - 1)
        counter[self.eos] = int(1e100 - 2)

        items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        items = [i[0] for i in items if i[1] >= self.min_freq][:self.max_size]
        self.idx2sym = items
        self.sym2idx = dict(zip(items, range(len(items))))

        print('final vocab size {} from {} unique tokens.'.format(
            len(self), len(counter)))

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, 'vocab.pkl'), 'bw') as f:
            pickle.dump(self.idx2sym, f)

    def tokenize(self, text, add_eos=False):
        if self.lower_case:
            symbols = [char.lower() for char in text]
        else:
            symbols = list(text)
        if add_eos:
            symbols.append(self.eos)

        return symbols

    def convert_to_index(self, symbols):
        return [self.get_index(symbol) for symbol in symbols]

    def get_index(self, symbol):
        if self.lower_case:
            symbol = symbol.lower()
        return self.sym2idx.get(symbol, self.sym2idx[self.unk])

    def get_indices(self, symbols):
        if isinstance(symbols, str):
            symbols = self.tokenize(symbols)
        return [self.get_index(symbol) for symbol in symbols]

    def get_symbols(self, indices, join=False):
        if np.ndim(indices) > 1:
            return [self.get_symbols(indices_, join=join) for indices_ in indices]
        else:
            symbols = [self.idx2sym[idx] for idx in indices]
            if join:
                symbols = ''.join(symbols)
            return symbols

    def convert_to_ndarray(self, symbols, dtype=np.int32):
        return np.asarray(self.convert_to_index(symbols), dtype=dtype)

    def encode_file(self, file, ordered=True, verbose=True, verbose_size=10000):
        if verbose:
            print('encoding file {} ...'.format(file))

        encoded = []
        with open(file, 'r', encoding='utf8') as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                if verbose and idx > 0 and idx % verbose_size == 0:
                    print(' line ' + str(idx))
                symbols = self.tokenize(line, add_eos=True)
                indices = self.convert_to_ndarray(symbols)
                encoded.append(indices)

        if ordered:
            encoded = np.concatenate(encoded)

        return encoded

    def __len__(self):
        return len(self.idx2sym)

    @property
    def size(self):
        return len(self.idx2sym)


def _int64_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


class ChineseCorpus(object):

    def __init__(self, path, vocab: Vocab, save_dir=None):
        self.corpus_path = path
        self.vocab = vocab
        if save_dir is None:
            self.save_dir = path
        else:
            self.save_dir = save_dir

    def path_of_tfrecord(self, name, batch_size, seq_len):
        return os.path.join(self.save_dir, '{}.bsz-{}.qlen-{}.tfrecord'.format(
            name, batch_size, seq_len))

    def create_ordered_tfrecords(self, names, batch_size, seq_len):
        if names is None:
            names = ('train', 'test', 'valid')
        if isinstance(names, str):
            names = (names,)
        for name in names:
            file = os.path.join(self.corpus_path, name + '.txt')
            if not os.path.exists(file):
                logging.warning('file `{}` not found!'.format(file))
                continue

            tfrecord_name = self.path_of_tfrecord(name, batch_size, seq_len)
            if os.path.exists(tfrecord_name):
                logging.info('{} already exists.'.format(tfrecord_name))
                continue

            encoded = self.vocab.encode_file(file, ordered=True)
            # ordered sequence to batch
            num_steps = len(encoded) // batch_size
            encoded = encoded[:batch_size * num_steps]
            batch_data = np.reshape(encoded, (batch_size, num_steps))

            self.convert_to_tfrecord(batch_data, name=name, batch_size=batch_size,
                                     seq_len=seq_len)

    def convert_to_tfrecord(self, batched_data, name, batch_size, seq_len):
        if np.ndim(batched_data) < 2:
            raise ValueError('batched_data cannot be 1D array!')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        filename = self.path_of_tfrecord(name, batch_size, seq_len)
        record_writer = tf.io.TFRecordWriter(filename)

        # total steps
        num_steps = (np.shape(batched_data)[1] - 1) // seq_len
        for i in range(0, num_steps * seq_len, seq_len):
            for j in range(batch_size):
                inputs = batched_data[j, i:i + seq_len]
                labels = batched_data[j, i + 1:i + seq_len + 1]
                feature = {
                    'inputs': _int64_feature(inputs),
                    'labels': _int64_feature(labels)
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                serialized_example = example.SerializeToString()
                # tf.train.Example.FromString(serialized_example)
                record_writer.write(serialized_example)

        record_writer.close()

        filename = os.path.split(filename)[1]
        print('Done writing {}. batches: {}'.format(filename, num_steps))

    def get_dataset(self, name, batch_size, seq_len):

        def parser(record):
            record_spec = {
                'inputs': tf.io.FixedLenFeature(shape=(seq_len,), dtype=tf.int64),
                'labels': tf.io.FixedLenFeature(shape=(seq_len,), dtype=tf.int64)
            }
            example = tf.io.parse_single_example(record, record_spec)
            # cast int64 dtype to int32
            for key in list(example.keys()):
                val = example[key]
                if val.dtype == tf.int64:
                    example[key] = tf.cast(val, dtype=tf.int32)
            return example

        tfrecord = self.path_of_tfrecord(name, batch_size, seq_len)
        if not os.path.exists(tfrecord):
            self.create_ordered_tfrecords(name, batch_size, seq_len)
        dataset = tf.data.TFRecordDataset(tfrecord)
        dataset = dataset.map(parser).batch(batch_size)
        if name == 'train':
            dataset = dataset.repeat()

        return dataset


if __name__ == '__main__':
    vocab_file = 'data/poetry/train.txt'
    vocab_save_dir = 'data/poetry/'
    vocab = Vocab(vocab_file, vocab_save_dir, min_freq=2, max_size=10000)

    # corpus_path = 'data/poetry/'
    # batch_size = 16
    # seq_len = 32
    # mem_len = 128
    # save_dir = 'data/poetry/'
    # corpus = ChineseCorpus(path=corpus_path, vocab=vocab, save_dir=save_dir)
    # corpus.get_dataset('train', batch_size, seq_len)
