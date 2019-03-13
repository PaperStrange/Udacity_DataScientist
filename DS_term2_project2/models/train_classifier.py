import gc, re, sys
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine

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


def load_data(database_filepath):
    """ Loads data from a databse file.

    Loads data using the pre-stored database file and calculate weights
    per each data.

    The weight of each data is calculated by two steps: first calculate the
    proportion of all classes which stored in "genre" column of raw data, then
    referring to the class that each data is belonged to, feed the weight
    using the reciprocal of the class proportion.

    Args:
        database_filepath: The file path of pre-stored database file

    Returns:
        X: A pandas dataframe containing raw twitter message data
        Y: A pandas dataframe containing 36 targets
        category_names: A list containing names of 36 categorical targets
        respectively

    Raises:
        None
    """

    global WEIGHTS_DF

    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql("SELECT * FROM " + database_filepath, engine)
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = list(Y.columns)

    df_genre = df.loc[:, "genre"]
    genre_summary = df_genre.value_counts()
    weights = 1 / (genre_summary / genre_summary.sum())
    WEIGHTS_DF = pd.Series(np.zeros(df_genre.shape))
    for ind, each_class in enumerate(genre_summary.index):
        WEIGHTS_DF[df_genre[df_genre==each_class].index] = weights[ind]

    return X, Y, category_names


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
    """

    def fit(self, X, y=None):
        """Fit the data"""
        return self

    def transform(self, X):
        """Allocate weight to statistical features"""
        X_balanced = np.array(
            [each_row*WEIGHTS_DF[ind] for ind, each_row in enumerate(X.values)]
        )

        return X_balanced


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


def build_model():
    """Builds model using sklearn "Pipeline" and "GridSearchCV".

    Bulids model combining sklearn instances and self-designed instances.

    It may fail when inputting data with improper feature data formation and
    target dimension.

    Args:
        None

    Returns:
        A Pipeline or GridSearchCV instance

    Raises:
        None
    """

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect1', CountVectorizer(tokenizer=tokenize_word, max_df=0.5, ngram_range=(1, 2), max_features=10000)),
                ('tfidf1', TfidfTransformer(use_idf=True))
            ])),

            ("statistics_balanced_pipeline", Pipeline([
                ("vect3", CountVectorizer(tokenizer=tokenize_word, max_df=0.5, ngram_range=(1, 2), max_features=10000)),
                ("statistics", StatisticalAnalysis()),
                ("balanceweight", BalanceWeight())
            ])),

        ], transformer_weights={'text_pipeline': 0.8, 'statistics_balanced_pipeline': 0.8})

         ),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, min_samples_split=10, max_features="sqrt", min_weight_fraction_leaf=0, oob_score=True)))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'features__transformer_weights': (
            {'text_pipeline': 0.8, 'statistics_balanced_pipeline': 1.0},
            {'text_pipeline': 0.8, 'statistics_balanced_pipeline': 0.6},
            {'text_pipeline': 1.0, 'statistics_balanced_pipeline': 0.8},
            {'text_pipeline': 0.6, 'statistics_balanced_pipeline': 0.8},
        )
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    # return pipeline
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates model prediction results.

    Evaluates model prediction results using sklearn "classification_report"
    to dispaly precision, recall and f1-score for each class of each category.

    Args:
        model: The classifier model after fitting training data
        X_test: A pandas dataframe containing feature data for testing
        Y_test: A pandas dataframe containing target/category data for testing
        category_names: A list containing names of 36 categorical targets
        respectively

    Returns:
        None

    Raises:
        None
    """

    Y_pred = model.best_estimator_.predict(X_test)
    for ind, each_col in enumerate(category_names):
        print(
            "----------Check metrics for category named as {}----------"
                .format(each_col))
        print(classification_report(Y_test.loc[:, each_col], Y_pred[:, ind]))
        print("")


def save_model(model, model_filepath):
    """Saves model.

    Saves model to the specific file path as a ".pkl" file.

    It may need more time to save a full GridSearchCV model
    Args:
        model: The classifier model after fitting and predicting
        model_filepath: The file path model is expected to save in

    Returns:
        None

    Raises:
        None
    """
    joblib.dump(model, model_filepath)


def main():
    """Packages all functions in this script.

    Packages the above functions into one function for a convenient call to
    finish the whole modeling.

    If running input do not satisfy the required formation, a guide will be
    printed to display the preferred input formation.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print("Building model...")
        model = build_model()
        
        print("Training model...")
        model.fit(X_train, Y_train)
        
        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        # save_model(model, model_filepath)
        save_model(model.best_estimator_, model_filepath)

        print("Trained model saved!")

    else:
        print("Please provide the filepath of the disaster messages database "\
              "as the first argument and the filepath of the pickle file to "\
              "save the model to as the second argument. \n\nExample: python "\
              "train_classifier.py ../data/DisasterResponse.db classifier.pkl")


if __name__ == "__main__":
    main()