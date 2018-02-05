import pandas as pd
from text_analytics_utils import (perform_stemming, remove_stopwords_text, tokenize_text, sanitize_text,
                                  lowercase_text, load_nltk_data, escape_html_char, perform_spellchecking)
from text_analytics_stats import (display_plot_lsa, display_histogram, stats, display_confusion_matrix)
from text_analytics_ml import (one_hot_encode, run_logistic_reg, get_metrics, get_confusion_matrix)

pd.set_option('expand_frame_repr', False)


def get_data():
    data = pd.read_csv("data/socialmedia_relevant_cols.csv", encoding="ISO-8859-1")

    return data


def clean_data(input_df, column_name):
    # input_df, column_name = escape_html_char(input_df, 'text')
    input_df, column_name = lowercase_text(input_df, column_name)
    input_df, column_name = sanitize_text(input_df, column_name)

    return input_df, column_name


def tokenize_vis_data(input_df, column_name, visualize=True):
    input_df, column_name = tokenize_text(input_df, column_name)
    if visualize:
        print("Tokenized")
        _, _, sentence_lengths = stats(input_df[column_name])
        display_histogram(sentence_lengths, xlabel="Sentence length", ylabel="Number of sentences")

    return input_df, column_name


def remove_sw_vis_data(input_df, column_name, visualize=True):
    input_df, column_name = remove_stopwords_text(input_df, column_name)
    if visualize:
        print("Stopwords removed")
        _, _, sentence_lengths = stats(input_df[column_name])
        display_histogram(sentence_lengths, xlabel="Sentence length", ylabel="Number of sentences")

    return input_df, column_name


def stem_vs_data(input_df, column_name, visualize=True):
    input_df, column_name = perform_stemming(input_df, column_name)
    if visualize:
        print("Stemmed")
        _, _, sentence_lengths = stats(input_df[column_name])
        display_histogram(sentence_lengths, xlabel="Sentence length", ylabel="Number of sentences")

    return input_df, column_name


def spellcheck_vis_data(input_df, column_name, visualize=True):
    input_df, column_name = perform_spellchecking(input_df, column_name)
    if visualize:
        print("Spell Checked")
        _, _, sentence_lengths = stats(input_df[column_name])
        display_histogram(sentence_lengths, xlabel="Sentence length", ylabel="Number of sentences")

    return input_df, column_name


if __name__ == "__main__":
    load_nltk_data()
    input_data = get_data()
    display_histogram(input_data['choose_one'].values, ylabel="count")
    cleaned_data, col_nm = clean_data(input_data, "text")
    tokenized_data, col_nm = tokenize_vis_data(cleaned_data, col_nm)
    sw_removed_data, col_nm = remove_sw_vis_data(tokenized_data, col_nm)
    stemmed_data, col_nm = stem_vs_data(sw_removed_data, col_nm)
    X_train, X_test, y_train, y_test, X_train_counts, X_test_counts = one_hot_encode(stemmed_data, col_nm)
    display_plot_lsa(X_train_counts, y_train)

    y_predicted_counts = run_logistic_reg(y_train, X_train_counts, X_test_counts)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    cm = get_confusion_matrix(y_test, y_predicted_counts)
    display_confusion_matrix(cm, classes=['Irrelevant', 'Disaster', 'Unsure'], normalize=False,
                             title='Confusion matrix')
