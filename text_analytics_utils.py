from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from textblob import TextBlob, Word
from nltk import download
from html.parser import HTMLParser
import pandas as pd

# Initialize Stemmer
stemmer = LancasterStemmer()


def load_nltk_data():
    # Download NLTK corpus:
    #   Download stopwords list.
    download('stopwords')
    #   Download data for tokenizer.
    download('punkt')


def escape_html_char(df, text_field):
    html_parser = HTMLParser()
    generated_text_field = 'htmlparsed_text'
    df[generated_text_field] = df[text_field].apply(html_parser.feed)

    return df, generated_text_field


def sanitize_text(df, text_field):
    """ Remove all irrelevant characters such as any non alphanumeric characters, urls, and mentions """
    generated_text_field = 'sanitized_text'
    df[generated_text_field] = df[text_field].str.replace(r'http\S+', ' ')
    df[generated_text_field] = df[generated_text_field].str.replace(r'http', ' ')
    df[generated_text_field] = df[generated_text_field].str.replace(r'@\S+', ' ')
    df[generated_text_field] = df[generated_text_field].str.replace(r'@', 'at')
    # you can go a little bit more relaxed: [^A-Za-z0-9(),!?@\'\`\"\_\n]
    df[generated_text_field] = df[generated_text_field].str.replace(r'[^0-9a-zA-Z]+', ' ')

    return df, generated_text_field


def lowercase_text(df, text_field):
    """ Convert all characters to lowercase, in order to treat words such as “hello”, “Hello”, and “HELLO” the same """
    generated_text_field = 'lowercased_text'
    df[generated_text_field] = df[text_field].str.lower()

    return df, generated_text_field


def remove_punctuation(df, text_field):
    """ Remove punctuation """
    generated_text_field = 'punctuation_removed_text'
    df[generated_text_field] = df[text_field].str.replace('[^\w\s]', '')

    return df, generated_text_field


def tokenize_text(df, text_field):
    """ Tokenize your text by separating it into individual words """
    generated_text_field = 'tokenized_text'
    df[generated_text_field] = df[text_field].apply(word_tokenize)

    return df, generated_text_field


def remove_stopwords_text(df, text_field):
    """ Remove stop words """
    generated_text_field = 'removed_stopwords_text'
    stop_words = stopwords.words('english')
    df[generated_text_field] = df[text_field].apply(lambda x: [w for w in x if w not in stop_words])

    return df, generated_text_field


def remove_common_words(df, text_field):
    """ Remove top 10 common words """
    generated_text_field = 'removed_common_words'
    freq = pd.Series(' '.join(df[text_field]).split()).value_counts()[:10]
    freq = list(freq.index)
    df[generated_text_field] = df[text_field].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    return df, generated_text_field


def remove_rare_words(df, text_field):
    """ Remove top 10 rare words """
    generated_text_field = 'removed_rare_words'
    freq = pd.Series(' '.join(df[text_field]).split()).value_counts()[-10:]
    freq = list(freq.index)
    df[generated_text_field] = df[text_field].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    return df, generated_text_field


def perform_stemming(df, text_field):
    """ Perform word stemming (reduce words such as “am”, “are”, and “is” to a common form such as “be”) """
    generated_text_field = 'stemmed_text'
    df[generated_text_field] = df[text_field].apply(lambda x: [stemmer.stem(w) for w in x])

    return df, generated_text_field


def perform_lemmatization(df, text_field):
    """
    Lemmatization is a more effective option than stemming because it converts the word into its
    root word, rather than just stripping the suffices. It makes use of the vocabulary and does a
    morphological analysis to obtain the root word. Therefore, we usually prefer using lemmatization
    over stemming.
    """
    generated_text_field = 'lemmatized_text'
    df[generated_text_field] = df[text_field].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    return df, generated_text_field


def perform_spellchecking(df, text_field):
    """
    WARNING: Takes a long time to run!
    Perform spell checking using TextBlob library based on P. Norvig article: norvig.com/spell-correct.html
    """
    generated_text_field = 'spell_corrected_text'
    df[generated_text_field] = df[text_field].apply(lambda x: [Word(w).spellcheck() for w in x])

    return df, generated_text_field


# b = TextBlob("I havv goood speling!")
# print(b.correct())
#
# w = Word('nutral')
# print(w.spellcheck())