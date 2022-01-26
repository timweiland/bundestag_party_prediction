#!/usr/bin/env python3
import logging
import string
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk


def remove_punctuation(s):
    return s.translate(str.maketrans("", "", string.punctuation))


def count_char(text, char):
    return text.count(char)


stop_words = stopwords.words("german")


def is_stop_word(word):
    return word in stop_words


def is_word(s):
    return any(c.isalnum() for c in s)


class NumExclamationQuestionExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that adds two features containing the relative number of exclamation marks and question marks.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def count_exclamations(self, text):
        return count_char(text, "!") / len(text)

    def count_questions(self, text):
        return count_char(text, "?") / len(text)

    def transform(self, X):
        X["relative_num_exclamations"] = X["text"].apply(self.count_exclamations)
        X["relative_num_questions"] = X["text"].apply(self.count_questions)
        return X


class Tokenizer(BaseEstimator, TransformerMixin):
    """
    Transformer that tokenizes the speeches.
    IMPORTANT: Note that this step changes the text in-place to save memory.
    Make sure you do not need the original text anymore after you apply this transformer!
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def tokenize(self, text):
        words = nltk.word_tokenize(text)
        words = [word for word in words if is_word(word)]
        return " ".join(words).lower()

    def transform(self, X):
        X["text"] = X["text"].apply(self.tokenize)
        return X


class AvgWordLengthExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that adds a feature containing the average word length of each text.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def get_avg_word_length(self, text):
        words = text.split()
        return sum([len(word) for word in words]) / len(words)

    def transform(self, X):
        X["avg_word_length"] = X["text"].apply(self.get_avg_word_length)
        return X


class StopWordFractionExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that adds a feature containing the fraction of stop words.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def get_stop_word_fraction(self, text):
        words = text.split()
        num_stop_words = len([word for word in words if is_stop_word(word)])
        return num_stop_words / len(words)

    def transform(self, X):
        X["stop_word_fraction"] = X["text"].apply(self.get_stop_word_fraction)
        return X


class StopWordRemover(BaseEstimator, TransformerMixin):
    """
    Transformer that removes stop words.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def remove_stop_words(self, text):
        words = text.split()
        words = [word for word in words if not is_stop_word(word)]
        return " ".join(words)

    def transform(self, X):
        X["text"] = X["text"].apply(self.remove_stop_words)
        return X


class TfidfScoreExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that adds a feature containing the average tfidf score of each speech.
    Should be used with tokenized text.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vectorizer = TfidfVectorizer()
        score_matrix = vectorizer.fit_transform(X["text"])
        X["avg_tfidf"] = score_matrix.mean(axis=1).ravel().A1
        return X


class RemoveUnwantedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that removes unwanted features/columns.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(
            columns=[
                "index",
                "date",
                "agenda",
                "speechnumber",
                "speaker",
                "party.facts.id",
                "chair",
                "terms",
                "text",
                "parliament",
                "iso3country",
            ]
        )
        return X
