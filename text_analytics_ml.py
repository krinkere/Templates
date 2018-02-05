from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, confusion_matrix)


def cv(data):
    count_vectorizer = CountVectorizer()
    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer


def one_hot_encode(df, text_field):
    """ Machine Learning models take numerical values as input. Our dataset is a list of sentences, so in order for our
    algorithm to extract patterns from the data, we first need to find a way to represent it in a way that our algorithm
    can understand, i.e. as a list of numbers.

    One-hot encoding (Bag of Words)
         we build a vocabulary of all the unique words in our dataset, and associate a unique index to each word in the
         vocabulary. Each sentence is then represented as a list that is as long as the number of distinct words in our
         vocabulary. At each index in this list, we mark how many times the given word appears in our sentence. This is
         called a Bag of Words model, since it is a representation that completely ignores the order of words in our
         sentence.
    """
    list_corpus = df[text_field].apply(lambda x: " ".join(x)).tolist()
    list_labels = df["class_label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,
                                                        random_state=40)
    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_counts, X_test_counts


def run_logistic_reg(y_train, X_train_counts, X_test_counts):
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    clf.fit(X_train_counts, y_train)
    y_predicted_counts = clf.predict(X_test_counts)

    return y_predicted_counts


def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


def get_confusion_matrix(y_test, y_predicted_counts):
    cm = confusion_matrix(y_test, y_predicted_counts)

    return cm

