import numpy as np
import pandas as pd
from loguru import logger
import datetime
import sys
sys.path.append('../../../../infrastructure/tools')
from utilities import utilities
import json
from data_config import *
import pickle
sys.path.append('../../../../infrastructure/preprocessing')
from data_transformer import dummyTransformer
from feature_engineering import feature_engineering
from feature_engine.datetime import DatetimeFeatures
import duckdb
from sklearn.model_selection import train_test_split


def main():

    logger.add("feature_engineering.log", level="INFO")

    logger.info(f"Execution time stamp: {datetime.datetime.now()}")

    logger.info(f"processing training data")

    create_data(f'{intermediate}/intermediate_training_user.csv', f'{processed_data}/training_user.csv', testing=False)

    logger.info(f"processing testing data")

    create_data(f'{intermediate}/intermediate_testing_user.csv', f'{processed_data}/testing_user.csv', testing=True)

    logger.info(f"All done.")


def create_data(input_data_path:str, output_data_path:str, testing=False):

    transactions = pd.read_csv(input_data_path)

    logger.info('Generating dummies and clustering them.')

    auto_dummy_vars = ['merchant_state','mcc','errors','merchant_type']

    for var in auto_dummy_vars:

        logger.info(f'Creating dummies and clustering schema for {var}')

        transactions[var].fillna('missing')

        if not testing:

            feature_engineering.auto_group_dummies(transactions, varlist=[var], y_var = 'target', mtype='classification',out_filename=f'auto_group_dummy_{var}_user.yaml', min_obs=20)

        logger.info(f'Clustering dummies for {var}')

        dt = dummyTransformer(utilities.read_yaml_file(f'auto_group_dummy_{var}_user.yaml'))

        transactions = dt.fit_transform(transactions)

    logger.info(f'saving data to {output_data_path}: shape: ({transactions.shape[0]},{transactions.shape[1]})')

    transactions.to_csv(output_data_path, index=False)


if __name__ == "__main__":
    main()





