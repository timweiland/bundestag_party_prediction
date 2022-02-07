#!/usr/bin/env python3
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as ssp


class BoWExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, num_words, max_df=0.5):
        self.num_words = num_words
        self.max_df = max_df

    def fit(self, X, y=None):
        self.vectorizer = CountVectorizer(
            analyzer="word",
            stop_words=None,
            tokenizer=None,
            preprocessor=None,
            max_features=self.num_words,
            max_df=self.max_df,
        )
        self.vectorizer.fit(X["text"])
        return self

    def transform(self, X):
        bow = self.vectorizer.transform(X["text"])
        X = X.drop(columns=["text"])
        X = X.to_numpy()

        return ssp.hstack((X, bow))


def get_fair_bow_regression():
    return Pipeline(
        [
            ("bow_extractor", BoWExtractor(num_words=15000)),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "regression",
                LogisticRegression(
                    solver="saga",
                    random_state=42,
                    n_jobs=-1,
                    max_iter=250,
                    class_weight="balanced",
                ),
            ),
        ]
    )
