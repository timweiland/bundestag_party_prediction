#!/usr/bin/env python3
import logging
import re
from sklearn.base import BaseEstimator, TransformerMixin


class RemoveChairTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that removes speech blocks from the chair of the Bundestag, i.e. the person who moderates the discussion.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[X["chair"] == False]


class RemoveCommentaryTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that removes commentary.
    We define commentary as anything that wasn't said by the main speaker, e.g. interjections from other members of parliament
    or meta-commentary such as "[INTERVENTION BEGINS]".
    """

    def __init__(self, remove_leftovers=True, verbose=True):
        self.remove_leftovers = remove_leftovers
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def remove_parens(self, text):
        text_clean = re.sub(r"\([^)]*\)", "", text)
        text_clean = re.sub(
            r"\([^)]*", "", text_clean
        )  # Sometimes there will be a comment right before the end of the speech with no closing paren
        return text_clean

    def remove_interventions(self, text):
        text_clean = re.sub("\[INTERVENTION BEGINS\]", "", text)
        text_clean = re.sub("\[INTERVENTION ENDS\]", "", text_clean)
        return text_clean

    def process(self, text):
        return self.remove_interventions(self.remove_parens(text))

    def find_leftovers(self, X):
        return X[X["text"].str.find("[") > -1]

    def transform(self, X):
        X = X.reset_index()
        X["text"] = X["text"].apply(self.process)
        if self.remove_leftovers:
            leftovers = self.find_leftovers(X)
            indices_to_drop = leftovers.index
            num_before = len(X)
            X = X.drop(indices_to_drop)
            num_after = len(X)
            num_dropped = num_before - num_after
            self.logger.info(
                f"RemoveCommentaryTransformer: Dropped {num_dropped} ({100 * num_dropped / num_before:.2f}%) speeches. {num_after} speeches remain."
            )
        return X


class RemoveSpecialCharacterTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that removes special characters not required for further analysis, e.g. underscores used for formatting.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def remove_underscores(self, text):
        return re.sub("_", "", text)

    def transform(self, X):
        X["text"] = X["text"].apply(self.remove_underscores)
        return X


class GermanSpellingReformTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that enforces the changes of the German spelling reform, e.g. 'ß' => 'ss'.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def replace_esszet(self, text):
        return re.sub("ß", "ss", text)

    def transform(self, X):
        X["text"] = X["text"].apply(self.replace_esszet)
        return X
