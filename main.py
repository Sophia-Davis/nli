from collections import Counter
from glob import glob
from pprint import pprint as pp
import numpy as np
import operator

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC, SVC

from nli_util import LING_DIR, get_char_bigrams, get_unigrams, shelved  # noqa
from nli_util import get_training_labels


def list_all_chars():
    cnt = Counter()
    for fn in glob(LING_DIR + '/nli/training/*txt'):
        with open(fn, 'r') as f:
            for c in f.read():
                cnt[c.lower()] += 1

    for elem in sorted(cnt.items(), key=lambda x: x[1]):
        print(elem)


def evaluate_svc(features, labels, bag=10):
    model = LinearSVC(dual=False)
    bagged_model = BaggingClassifier(base_estimator=model, n_estimators=bag,
                            max_samples=0.5, max_features=0.5, n_jobs=6,
                            verbose=1)
    # model = SVC(verbose=1)
    labels = np.argmax(labels, axis=1)
    print(cross_val_score(bagged_model, X=features, y=labels, scoring='accuracy'))


if __name__ == '__main__':
    # unigrams = get_unigram_count()
    word_ug = get_unigrams('training')
    ch_bigrams = get_char_bigrams('training')

    comprehensive_array = np.concatenate((word_ug, ch_bigrams), axis=1)
    labels = get_training_labels()
    print(evaluate_svc(comprehensive_array, labels))

# uncomment three lines below to test bigrams
# bigrams = get_char_bigrams('/home/sophia/python_*/*550/f*_p*/nli/trai*/*')
# labels = get_training_labels(clear_cache=True)

# print(evaluate_svc(bigrams, labels))
