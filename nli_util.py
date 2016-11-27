from collections import Counter
from functools import wraps
from glob import glob
import itertools
import shelve
import string

import numpy as np

LING_DIR = '/home/sophia//python_programming/LING_550/final_project/'
LING_CACHE = '/home/sophia//python_programming/LING_550/final_project/cache.db'
C_BIGRAMS = [x for x in
             itertools.product(string.ascii_lowercase + ' ', repeat=2)]
INDICES = {bigram: ix for ix, bigram in enumerate(C_BIGRAMS)}


def shelved(key):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, clear_cache=False, **kwargs):
            with shelve.open(LING_CACHE) as db:
                if key in db and not clear_cache:
                    out = db[key]
                else:
                    # database is overwritten
                    out = f(*args, **kwargs)
                    db[key] = out
            return out
        return wrapped
    return decorator


def get_x_files(mode='training'):
    if mode == 'training':
        d = LING_DIR + '/nli/training/*'
    elif mode == 'testing':
        d = LING_DIR + '/nli/testing/*'
    else:
        raise ValueError('not valid training mode')

    return sorted(glob(d), key=lambda x: int(x.split('/')[-1][:-4]))


@shelved('training_labels')
def get_training_labels():
    lab_list = []
    with open(LING_DIR + 'nli/training_langid.txt', 'r') as f:
        for row in f:
            lab_list.append(row.split(',')[-1])
    possible_labels = sorted(set(lab_list))
    out = np.zeros((len(lab_list), len(possible_labels)))

    for lx, lab in enumerate(lab_list):
        out[lx, possible_labels.index(lab)] = 1

    return out


@shelved('char_bigrams')
def get_char_bigrams(mode='training'):
    files = get_x_files(mode=mode)
    # initialze array of correct size with zeroes
    # features are columns; different files (essays) are rows
    out = np.zeros((len(files), len(C_BIGRAMS)))
    for fx, fn in enumerate(files):
        # fx = file/row
        with open(fn, 'r') as f:
            text = f.read().lower()
            # for each bigram found in text, add 1 to corresponding spot in
            # array
            for bigram in zip(text, text[1:]):
                try:
                    out[fx, INDICES[bigram]] += 1
                except KeyError:
                    continue

    return out


def get_unigram_count(min_occurrences=5):
    cnt = Counter()
    # make dict of unigrams in all training files
    # not spell corrected
    for fn in get_x_files(mode='training'):
        with open(fn, 'r') as f:
            for line in f:
                for word in line.split():
                    cnt[word] += 1

    # only accept common unigrams
    final_cnt = {k: v for k, v in cnt.items() if v > min_occurrences}

    return final_cnt


@shelved('word_unigrams')
def get_unigrams(mode='training', **kwargs):
    files = get_x_files(mode=mode)
    ugram_ctr = get_unigram_count(**kwargs)

    indices = {word: ix for ix, word in enumerate(sorted(ugram_ctr.keys()))}

    out = np.zeros((len(files), len(ugram_ctr)))
    for fx, fn in enumerate(files):
        with open(fn, 'r') as f:
            for line in f:
                for word in line.split():
                    try:
                        out[fx, indices[word]] += 1
                    except KeyError:
                        continue

    return out
