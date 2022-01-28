#!/usr/bin/env python3
import logging
import string
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import textstat


def remove_punctuation(s):
    return s.translate(str.maketrans("", "", string.punctuation))


def count_char(text, char):
    return text.count(char)


stop_words = stopwords.words("german")


def is_stop_word(word):
    return word in stop_words


def is_word(s):
    return any(c.isalnum() for c in s)


d = {} # create dictionary 'd' with words as keys and their sentiment-scores as values

with open("./SentiWS_v2.0/SentiWS_v2.0_Negative.txt") as f:
    for line in f:
        split = re.split('\||\s|,',line)
        keys = [split[0]] + split[3:-1]
        value = float(split[2])
        for key in keys:
            d[key] = value

with open("./SentiWS_v2.0/SentiWS_v2.0_Positive.txt") as g:
    for line in g:
        split = re.split('\||\s|,',line)
        keys = [split[0]] + split[3:-1]
        value = float(split[2])
        for key in keys:
            d[key] = value


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


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that adds a feature containing text length.
    """
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def text_length(self, text):
        return len(text)

    def transform(self, X):
        X["text_length"] = X["text"].apply(self.text_length)
        return X


class AvgSentenceLengthExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that adds a feature containing the average sentence length.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def avg_sentence_length(text):
        text_length = len(text)
        sentences = nltk.sent_tokenize(text,language='german')
        n_sentences = len(sentences)
        return text_length/n_sentences

    def transform(self, X):
        X["avg_sentence_length"] = X["text"].apply(self.avg_sentence_length)
        return X


class NumOfProfanitiesExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that adds a feature containing the number of profanities.
    The profanities stem from a predefined list of unique strings, which also includes declinations.
    """
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    profanities = ['Idiotin', 'Idioten', 'Möchtegern', 'Stümper', 'Hinterwäldlerin', 'Drecksack', 'Arschgeige', 
               'Arschloch', 'Volltrottel', 'unterbelichtet', 'Nörglerin', 'Schwachkopf', 'Faulpelz', 
               'Dreckskerl', 'Rowdy', 'hochnäsig', 'großmäulig', 'Scheiße', 'Deppen', 'verrückter', 
               'Backpfeifengesicht', 'abgefahrener', 'Lümmel', 'Ochse', 'Nörgeln', 'Depp', 'bescheuertster', 
               'Schnapsidee', 'Trottel', 'Nervensägen', 'blöde', 'Schlitzohr', 'Hanswürste', 'Zuhälter', 
               'Bauerntölpel', 'Hetzer', 'Schnauze', 'Dummköpfin', 'spinnen', 'Hetzerin', 'hochnäsige', 
               'spießig', 'Kacke', 'Ratte', 'Lackschuhpanter', 'Heuchlerin', 'dämlicher', 'beschissene', 
               'Arsch', 'Nervensäge', 'beschissen', 'Blödmann', 'Klugscheißer', 'Bastard', 'aufgeblasener', 
               'dummer', 'lahm', 'kotzen', 'altbacken', 'dümmstes', 'idiotisch', 'Schwachköpfe', 'scheiß', 
               'abgefahren', 'Dummkopf', 'dumme', 'Dummköpfe', 'Kleingeist', 'Hornochse', 'bescheuertstes', 
               'Schlange', 'Hackfresse', 'Armleuchter', 'Dreckschwein', 'hirnrissig', 'Verrückte', 'schlampig', 
               'kacke', 'Harzer', 'Pisser', 'blödes', 'Hund', 'spinnt', 'Hornochsen', 'Abschaum', 'Stinktier', 
               'Esel', 'Amateur', 'Großmaul', 'bescheuerte', 'verrückt', 'Alleswisser', 'blöd', 'Luder', 
               'schäbiger', 'Berufsrandalierer', 'Fresse', 'Stümperin', 'Zicke', 'aufgeblasene', 'Hanswurst', 
               'Sack', 'Teufel', 'Vollpfosten', 'Ziege', 'Galgenkandidat', 'Hurensohn', 'kindisch', 'Idiot', 
               'Dreckschweine', 'schmierig', 'Verrückter', 'Angeberin', 'Schwein', 'Hinterwäldler', 'verdammte', 
               'kleingeistig', 'aufgeblasen', 'Affe', 'bescheuertste', 'versifft', 'Bastarde', 'bieder', 
               'schäbig', 'Blöde', 'Schweine', 'Gangster', 'blödsinnig', 'dumm', 'stümperhaft', 'Arschlöcher', 
               'Spießer', 'verdammt', 'bescheuert', 'affig', 'Faulpelze', 'Angeber', 'Arschgeigen', 'Aasgeier', 
               'Besserwisser', 'Blöder', 'verrückte', 'Blödmänner', 'Heuchler', 'Nörgler', 'dämlich', 'dümmste', 
               'Hurensöhne', 'heuchlerisch', 'Pisse', 'bescheuerter', 'kleinkariert', 'Karnickel', 'kleinkarierte',
               'Blähhals']
    
    def count_profanities(text):
        n_profanities = 0
        tokens = nltk.word_tokenize(text, language='german')
        for profanity in profanities:
            n_profanities += tokens.count(profanity)
        return n_profanities

    def transform(self, X):
        X["num_profanities"] = X["text"].apply(self.count_profanities)
        return X
    
class TTRExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that adds a feature containing the type-to-token ratio (#unique words / #total words).
    """
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def TTR(text):
        tokens = nltk.word_tokenize(text, language='german')
        tokens = [token.lower() for token in tokens if token.isalpha()]
        n_total = len(tokens)
        n_unique = len(set(tokens))
        return n_unique/n_total

    def transform(self, X):
        X["TTR"] = X["text"].apply(self.TTR)
        return X


class ReadabilityExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that adds a feature containing a readability score. 
    Readability is calculated as Flesch-Reading-Ease for the German language.
    Interpretation: score of 0-30: very difficult, 30-50: difficult,
    50-60: medium/difficult, 60-70: medium, 70-80: medium/easy, 80-90: easy,
    90-100: very easy. (https://de.wikipedia.org/wiki/Lesbarkeitsindex#Flesch-Reading-Ease)
    """
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def readability(text):
        textstat.set_lang("de")
        return textstat.flesch_reading_ease(text)

    def transform(self, X):
        X["readability"] = X["text"].apply(self.readability)
        return X
    

class SentimentExtractor(BaseEstimator, TransformerMixin):    
    """
    Transformer that adds a feature containing a sentiment score (range: -1 to +1) for the text,
    calculated as average of sentiment scores for all words in the text which have an entry in the 'SentiWS' data set.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def sentiment(text):
        tokens = nltk.word_tokenize(text, language='german')
        tokens = [token.lower() for token in tokens if token.isalpha()]
        sentiment_sum = 0
        sentiment_n = 0
        for token in tokens:
            sentiment_score = d.get(token)
            if sentiment_score != None:
                sentiment_sum += sentiment_score
                sentiment_n += 1
        if sentiment_n > 0:
            return sentiment_sum/sentiment_n
        else:
            return 0

    def transform(self, X):
        X["sentiment"] = X["text"].apply(self.sentiment)
        return X
