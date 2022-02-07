import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import scipy.sparse as ssp

from models import get_fair_bow_regression


def get_train_test_split(clean_df, feats_df):
    X = feats_df
    X["text"] = clean_df["text"]
    X = X.dropna(axis=0)
    y = X["party"]
    X = X.drop(columns=["party"])
    X.reset_index(drop=True, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


clean_filename = "parlspeech_bundestag_clean.csv"
feats_filename = "parlspeech_bundestag_feats.csv"


@click.command()
@click.argument("data_filepath", type=click.Path(exists=True))
@click.argument("model_filepath", type=click.Path())
def main(data_filepath, model_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    model_filepath = Path(model_filepath)
    clean_filepath = Path(data_filepath) / clean_filename
    clean_df = pd.read_csv(clean_filepath, parse_dates=["date"], low_memory=False)
    feats_filepath = Path(data_filepath) / feats_filename
    feats_df = pd.read_csv(feats_filepath)

    logger.info("Generating train test split...")
    X_train, X_test, y_train, y_test = get_train_test_split(clean_df, feats_df)
    model = get_fair_bow_regression()
    logger.info("Training model...")
    model.fit(X_train, y_train)
    y_test_preds = model.predict(X_test)
    acc = accuracy_score(y_test, y_test_preds)
    print(f"Done training model. Final accuracy: {acc:.2f}")
    print("Writing model to file...")
    dump(model, model_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
