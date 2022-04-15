# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import zipfile
from pipeline import TransformationPipeline
import pandas as pd

def unzip_all_file(file_path):
    file_name = file_path.replace('.zip', '')

    # unzip the folder
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(file_name)

    # list all zip file inside the folder
    for file in os.listdir(file_name):

        if file.endswith('.zip'):
            unzip_all_file(os.path.join(file_name, file))


def download_dataset(p_project_dir, p_competition_name):
    """ Download any dataset directly from kaggle. The authentication token kaggle.jon must be setup. Download
        the dataset in data/raw and unzip all files.
    """

    raw_data_folder = os.path.join(p_project_dir, "data", "raw")

    # Build the kaggle API command
    cmd = "kaggle competitions download -c " + p_competition_name + " -p " + str(raw_data_folder)
    os.system(cmd)

    #
    zip_dataset_path = os.path.join(raw_data_folder, p_competition_name + '.zip')
    unzip_all_file(zip_dataset_path)


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    logger.info('input: %s --- output: %s', input_filepath, output_filepath)

    config_file_name = './data_cfg.yaml'

    pipeline = TransformationPipeline(config_file_name)

    # Get the training file
    dataset_path = os.path.join(input_path, 'train.csv')
    dataset = pd.read_csv(dataset_path)

    # Drop the ID column
    dataset = dataset.drop(['id'], axis=1)

    dataset.species = pipeline.encode_label(dataset.species)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    input_path = os.path.join(project_dir, 'data', 'raw')
    output_path = os.path.join(project_dir, 'data', 'processed')
    main(input_path, output_path)
