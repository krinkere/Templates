import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import itertools


def stats(stats_input):
    all_words = [word for tokens in stats_input for word in tokens]
    sentence_lengths = [len(tokens) for tokens in stats_input]
    VOCAB = sorted(list(set(all_words)))
    print("\t%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
    print("\tMax sentence length is %s" % max(sentence_lengths))

    return all_words, VOCAB, sentence_lengths


def display_histogram(histo_input, xlabel="", ylabel=""):
    fig = plt.figure(figsize=(10, 10))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist(histo_input)
    plt.show()


def display_plot_lsa(test_data, test_labels):
    fig = plt.figure(figsize=(16, 16))

    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    colors = ['orange', 'blue', 'blue']

    plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                cmap=matplotlib.colors.ListedColormap(colors))
    red_patch = mpatches.Patch(color='orange', label='Irrelevant')
    green_patch = mpatches.Patch(color='blue', label='Disaster')
    plt.legend(handles=[red_patch, green_patch], prop={'size': 30})

    plt.show()


def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None, average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None, average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


def display_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.winter):
    fig = plt.figure(figsize=(10, 10))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    plt.show()
    print(cm)

