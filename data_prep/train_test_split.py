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

    transactions = pd.read_csv(f"{intermediate}/transactions.csv").sample(frac=0.2,random_state=246)

    transactions['amount'] = transactions['amount'].apply(lambda x: float(x.replace("$", "")))

    X_train, X_test, y_train, y_test = train_test_split(transactions, transactions['target'], test_size=0.4, random_state=42)

    logger.info(f'saving training data to {intermediate}/training.csv')

    X_train.to_csv(f'{intermediate}/training.csv', index=False)

    logger.info(f'saving testing data to {intermediate}/testing.csv')

    X_test.to_csv(f'{intermediate}/testing.csv', index=False)

    logger.info('All done.')


if __name__ == "__main__":
    main()