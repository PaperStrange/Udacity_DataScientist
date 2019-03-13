import json
import gc, re, sys
import logging

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download(["punkt", "wordnet", "stopwords", "averaged_perceptron_tagger", "brown"])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import brown

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from flask import Flask
from flask import render_template, request, jsonify
import plotly
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

# from ..models.train_classifier import StatisticalAnalysis, BalanceWeight, number_normalizer  # Note 2019/3/13: SystemError occurs when using the below two import:"Parent module '' not loaded, cannot perform relative importl"
# from ..models.train_classifier import tokenize_word  # Note 2019/3/13: SystemError occurs when using the below two import:"Parent module '' not loaded, cannot perform relative importl"
# from DS_term2_project2.models.train_classifier import StatisticalAnalysis, BalanceWeight, number_normalizer  # Note 2019/3/13: ImportError occur when importing : No module named 'DS_term2_project2'
# from DS_term2_project2.models.train_classifier import tokenize_word  # Note 2019/3/13: ImportError occur when importing : No module named 'DS_term2_project2'

WEIGHTS_DF = pd.read_csv("../data/weights.csv")

class StatisticalAnalysis(BaseEstimator, TransformerMixin):
    """Extracts statics features of the array of token counts.

    Evaluates several statistical standards from the matrix of token counts
    provided by an instance such like "CountVectorizer".

    Tries to balance data by expanding more numberical features.

    Attributes:
        statistics_count: A function calculating the length of the array
        statistics_std: A function calculating the standard deviation of
        the array
        statistics_mean: A function calculating the mean of the array
        fit: A function inherited from "TransformerMixin"
        transform: A self-designed function to finish statistical feature
        extraction using attribute functions
    """

    def statistics_count(self, x_arr):
        """Calculating the length of the array"""
        return x_arr.shape[0]

    def statistics_std(self, x_arr):
        """Calculating the standard deviation of the array"""
        return x_arr.std()

    def statistics_mean(self, x_arr):
        """Calculating the mean of the array"""
        return x_arr.mean()

    def fit(self, X, y=None):
        """Fit the data"""
        return self

    def transform(self, X):
        """Apply statistical feature extraction to the data"""
        X = X.toarray()
        X_df = pd.DataFrame(X)

        X_count = X_df.apply(self.statistics_count, axis=1)
        X_std = X_df.apply(self.statistics_std, axis=1)
        X_mean = X_df.apply(self.statistics_mean, axis=1)
        X_sum = X_df.apply(sum, axis=1)
        del X

        return pd.concat([X_count, X_std, X_mean, X_sum], axis=1)


class BalanceWeight(BaseEstimator, TransformerMixin):
    """Tries to balance by apply a numeric weight to each data.

    Uses global parameter WEIGHTS_DF which containing each weight for each
    data to allocate weights.

    Attributes:
        fit: A function inherited from "TransformerMixin"
        transform: A self-designed function to allocate weight to statistical
        features oriented from the above "StatisticalAnalysis" class

    Raises:
        KeyError: occurs when using only one row data to classify

    """

    def fit(self, X, y=None):
        """Fit the data"""
        return self

    def transform(self, X):
        """Allocate weight to statistical features"""
        try:
            X_balanced = np.array(
                [each_row*WEIGHTS_DF[ind] for ind, each_row in enumerate(X.values)]
            )
        except KeyError:
            return X
        else:
            return X_balanced


def number_normalizer(tokens):
    """ Maps all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.

    By applying this form of dimensionality reduction, some methods may perform
    better.

    Args:
        tokens: The tokens extracted by a "TweetTokenizer" instance

    Returns:
        A generator object

    Raises:
        None
    """

    return ("numberplaceholder" if token[0].isdigit() else token for token in tokens)


def tokenize_word(text):
    """Tokenizes text row by row from pandas DataFrame.

    Tokenizes text by combining several instances including "TweetTokenizer",
    "WordNetLemmatizer", "PorterStemmer" and self-designed "number_normalizer"
    and several regular expressions for text normalization.

    It may fail when strings out of range of the defined reg rules.

    Args:
        text: A row text data

    Returns:
        A numpy array containing clean tokens extracted from text

        "Enjoy! 4 beautiful seasons and 3 day night~ @everyone" -->
            ['enjoy',
            'numberplacehold',
            'beauti',
            'season',
            'and',
            'numberplacehold',
            'day',
            'night']]

    Raises:
        None
    """

    url_reg = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    punct_reg = "[^a-zA-Z0-9@]+"
    detected_urls = re.findall(url_reg, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    text = re.sub(punct_reg, " ", text)
    text = text.lower()

    stop_words = stopwords.words("english")
    word_tokenizer_tweet = TweetTokenizer()
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    tokens = word_tokenizer_tweet.tokenize(text)
    tokens = number_normalizer(tokens)
    clean_tokens_list = \
        [lemmatizer.lemmatize(stemmer.stem(tok)).strip()
         for tok in tokens
         if tok not in stop_words]

    for ind, tok in enumerate(clean_tokens_list):
        if "@" in tok:
            del clean_tokens_list[ind]

    return clean_tokens_list


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse2.db')
df = pd.read_sql_table('DisasterResponse2', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()