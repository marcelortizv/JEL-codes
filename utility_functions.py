import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cufflinks as cf

cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px

import scipy.stats as stats
from sklearn.utils.fixes import loguniform
import scipy as sp
from scipy.sparse import hstack
from collections import Counter

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.multiclass import OneVsRestClassifier

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

def split(word):
    return [char for char in word]


def get_wcloud(df):
    stopwords = set(STOPWORDS)
    words = ''
    for document in df.abstract:
        val = str(document).lower()
        tokens = val.split()
        words += " ".join(tokens) + " "
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(words)
    return wordcloud


def plot_4_wclouds(df, l1, l2, l3, l4):
    l1 = str(l1).upper()
    l2 = str(l2).upper()
    l3 = str(l3).upper()
    l4 = str(l4).upper()

    wc_1 = get_wcloud(df[df['jel_dummy_' + l1] == 1])
    wc_2 = get_wcloud(df[df['jel_dummy_' + l2] == 1])
    wc_3 = get_wcloud(df[df['jel_dummy_' + l3] == 1])
    wc_4 = get_wcloud(df[df['jel_dummy_' + l4] == 1])
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
    ax1.imshow(wc_1)
    ax1.axis('off')
    ax1.set_title('JEL Code:' + l1)
    ax2.imshow(wc_2)
    ax2.axis('off')
    ax2.set_title('JEL Code:' + l2)
    ax3.imshow(wc_3)
    ax3.axis('off')
    ax3.set_title('JEL Code:' + l3)
    ax4.imshow(wc_4)
    ax4.axis('off')
    ax4.set_title('JEL Code:' + l4)


def get_top_n_words(corpus, n=20):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def plot_top_words(df, column_name, letter, n=20):
    letter = str(letter).upper()
    common_words = get_top_n_words(df[df['jel_dummy_' + letter] == 1][column_name], n)
    df2 = pd.DataFrame(common_words, columns=['abstract', 'count'])
    df2.groupby('abstract').sum()['count'].sort_values(ascending=False).iplot(
        kind='bar', yTitle='Count', linecolor='black',
        title=f'Top {n} words in review after removing stop words in JEL Code: {letter}')


def Accuracy(y_true, y_pred):
    """
    Accuracy based on Jaccard Similarity Score
    :param y_true: ground truth
    :param y_pred: prediction
    :return: Jaccard Similarity Score
    """
    jaccard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true, y_pred).sum(axis=1)
    return jaccard.mean()


def print_ml_score(y_test, y_pred, clf):
    print('Classifier: ', clf.__class__.__name__)
    print('Accuracy Score: {}'.format(Accuracy(y_test, y_pred)))
    print("-----------------------------------")


def train_model(classifier, feature_vector_train, label_train, feature_vector_test, label_test):
    # fit the training set on the classifier
    clf = OneVsRestClassifier(classifier)
    clf.fit(feature_vector_train, label_train)

    # predict the labels on test set
    predictions = clf.predict(feature_vector_test)

    return print_ml_score(label_test, predictions, classifier)


wnl = WordNetLemmatizer()
def clean_text(text_series):
    econ_stopwords = ['model', 'using', 'paper']
    text_tokens = word_tokenize(text_series)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    tokens_without_sw = [word for word in tokens_without_sw if not word in econ_stopwords]
    tokens_without_sw_lemma = [wnl.lemmatize(word, pos="v") for word in tokens_without_sw if not word in econ_stopwords]

    # removing stopwords and econ stopwords
    text_series = " ".join(tokens_without_sw_lemma)
    # removing double quotes from text
    text_series = text_series.replace('"', '')
    # removing single quotes from text
    text_series = text_series.replace("'", '')
    # removing comma from text
    text_series = text_series.replace(',', '')
    # removing dot from text
    text_series = text_series.replace('.', '')
    # removing double dot from text
    text_series = text_series.replace(':', '')
    # removing percentage from text
    text_series = text_series.replace('%', '')
    # remove numbers from text
    text_series = re.sub(r'[0-9]+', '', text_series)

    return text_series