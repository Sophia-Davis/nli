from collections import Counter
from glob import glob
from pprint import pprint as pp
import numpy as np
import operator

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix

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
    # model = LinearSVC(dual=False)
    # bagged_model = BaggingClassifier(base_estimator=model, n_estimators=bag,
    #                         max_samples=0.5, max_features=0.5, n_jobs=6,
    #                         verbose=1)
    # model = SVC(verbose=1)
    model = RandomForestClassifier(n_estimators=2000, max_depth=20, verbose=1,
                                   n_jobs=6)
    labels = np.argmax(labels, axis=1)
    # print(cross_val_score(model, X=features, y=labels, scoring='accuracy'))
    # out = cross_val_score(model, X=features, y=labels,
    # scoring=confusion_matrix))


def train_svc(features, labels, bag=10):
    model = LinearSVC(dual=False, verbose=1)
    bagged_model = BaggingClassifier(base_estimator=model, n_estimators=bag,
                                     max_samples=0.5, max_features=0.5, n_jobs=6,
                                     verbose=1)
    bagged_model.fit(features, labels)

    return bagged_model


if __name__ == '__main__':
    # unigrams = get_unigram_count()
    word_ug = get_unigrams('training')
    ch_bigrams = get_char_bigrams('training')

    labels, names = get_training_labels()
    labels = np.argmax(labels, axis=1)

    n_train = int(0.8*len(labels))

    comprehensive_array = np.concatenate((word_ug, ch_bigrams), axis=1)
    # print(evaluate_svc(comprehensive_array, labels))

    import pickle
    m = train_svc(comprehensive_array[:n_train], labels[:n_train], bag=10)

    with open('ug_weights.p', 'wb') as f:
        pickle.dump(m, f)

    with open('ug_weights.p', 'rb') as f:
        m = pickle.load(f)

    y_pred = m.predict(comprehensive_array[n_train:])
    cm = confusion_matrix(labels[n_train:], y_pred)
    cm = np.float64(cm)
    cm /= np.sum(cm, axis=1)


    import matplotlib.pyplot as plt
    print(names)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, interpolation='nearest', cmap='viridis')
    fig.colorbar(cax)

    plt.xticks(list(range(11)))
    plt.yticks(list(range(11)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

    plt.show()

# uncomment three lines below to test bigrams
# bigrams = get_char_bigrams('/home/sophia/python_*/*550/f*_p*/nli/trai*/*')
# labels = get_training_labels(clear_cache=True)

# print(evaluate_svc(bigrams, labels))
