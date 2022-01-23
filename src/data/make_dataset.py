# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from transformers.clean_text import *


def get_pipeline():
    return Pipeline(
        [
            ("chair_remover", RemoveChairTransformer()),
            (
                "commentary_remover",
                RemoveCommentaryTransformer(remove_leftovers=True, verbose=True),
            ),
            ("special_chars_remover", RemoveSpecialCharacterTransformer()),
            ("spelling_reformer", GermanSpellingReformTransformer()),
        ]
    )


input_csv_name = "parlspeech_bundestag.csv"
output_csv_name = "parlspeech_bundestag.csv"


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
    output_filepath = Path(output_filepath) / output_csv_name

    pipeline = get_pipeline()
    logger.info("Reading input data...")
    data = pd.read_csv(input_filepath, parse_dates=["date"], low_memory=False)
    logger.info("Running pipeline...")
    data = pipeline.fit_transform(data)
    logger.info("Writing transformed dataset...")
    data.to_csv(output_filepath)
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
