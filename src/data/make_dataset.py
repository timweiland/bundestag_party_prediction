# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from transformers.clean_text import *
from transformers.extract_features import *


def get_cleaning_pipeline():
    return Pipeline(
        [
            ("chair_remover", RemoveChairTransformer()),
            (
                "commentary_remover",
                RemoveCommentaryTransformer(remove_leftovers=True, verbose=True),
            ),
            ("special_chars_remover", RemoveSpecialCharacterTransformer()),
            ("spelling_reformer", GermanSpellingReformTransformer()),
            ("double_fullstops_remover", ReplaceDoubleFullstopsTransformer()),
        ],
        verbose=True,
    )


def get_feature_pipeline():
    return Pipeline(
        [
            ("text_length_extractor", TextLengthExtractor()),
            ("avg_sentence_length_extractor", AvgSentenceLengthExtractor()),
            ("num_exclamation_question_extractor", NumExclamationQuestionExtractor()),
            ("readability_extractor", ReadabilityExtractor()),
            ("tokenizer", Tokenizer()),
            ("num_profanities_extractor", NumOfProfanitiesExtractor()),
            ("ttr_extractor", TTRExtractor()),
            ("sentiment_extractor", SentimentExtractor()),
            ("avg_word_length_extractor", AvgWordLengthExtractor()),
            ("stop_word_fraction_extractor", StopWordFractionExtractor()),
            ("stop_word_remover", StopWordRemover()),
            ("tfidf_score_extractor", TfidfScoreExtractor()),
            ("unwanted_features_remover", RemoveUnwantedFeaturesTransformer()),
        ],
        verbose=True,
    )


input_csv_name = "parlspeech_bundestag.csv"
output_csv_name_clean = "parlspeech_bundestag_clean.csv"
output_csv_name_feats = "parlspeech_bundestag_feats.csv"


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")

    input_filepath = Path(input_filepath) / input_csv_name
    output_filepath_clean = Path(output_filepath) / output_csv_name_clean
    output_filepath_feats = Path(output_filepath) / output_csv_name_feats

    cleaning_pipeline = get_cleaning_pipeline()
    logger.info("Reading input data...")
    data = pd.read_csv(input_filepath, parse_dates=["date"], low_memory=False)
    logger.info("Running cleaning pipeline...")
    data = cleaning_pipeline.fit_transform(data)
    logger.info("Writing cleaned dataset...")
    data.to_csv(output_filepath_clean)

    feature_pipeline = get_feature_pipeline()
    logger.info("Running feature pipeline...")
    data = feature_pipeline.fit_transform(data)
    logger.info("Writing dataset with features...")
    data.to_csv(output_filepath_feats, index=False)
    logger.info("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
