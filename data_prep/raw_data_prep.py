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

    logger.add("feature_engineering_intermediate.log", level="INFO")

    logger.info(f"Execution time stamp: {datetime.datetime.now()}")

    logger.info(f"saving intermediate training data to {intermediate}/intermediate_training_user.csv")

    create_data(f'{intermediate}/training.csv', f'{intermediate}/intermediate_training_user.csv',window_agg_yaml_user)

    logger.info(f"saving intermediate testing data to {intermediate}/intermediate_testing_user.csv ")

    create_data(f'{intermediate}/testing.csv', f'{intermediate}/intermediate_testing_user.csv',window_agg_yaml_user)

    logger.info(f"All done.")


def create_data(input_data_path:str, output_data_path:str, window_agg_yaml:str):

    cards = pd.read_csv(f'{intermediate}/cards.csv')

    transactions = pd.read_csv(input_data_path).sort_values(by=['user_id','card_id']).reset_index(drop=True)

    logger.info(f'preprocess cards data. shape is: ({cards.shape[0]},{cards.shape[1]})')

    logger.info('converting dollar amount to float numbers')

    for var in ['per_capita_income', 'yearly_income', 'total_debt', 'credit_limit']:
        cards[var] = cards[var].apply(lambda x: float(x.replace("$", "")))

    cards['card_on_dark_web'] = cards['card_on_dark_web'].apply(lambda x: 1 if x.lower() == 'yes' else (0 if x.lower() == 'no' else x))

    user_attr = [
        'user_id',
        'current_age',
        'retirement_age',
        'birth_year',
        'birth_month',
        'gender',
        'address',
        'latitude',
        'longitude',
        'per_capita_income',
        'yearly_income',
        'total_debt',
        'credit_score',
        'num_credit_cards']

    user_info = cards[user_attr].drop_duplicates().reset_index(drop=True)

    logger.info('Creating user level aggregates')

    user_info_agg = utilities.pivot(cards, varlist={'credit_limit': 'sum', 'card_on_dark_web': 'sum'},
                                    by_vars=['user_id'])

    user_info=user_info.merge(user_info_agg, on='user_id')

    user_info['dti'] = user_info.total_debt / user_info.yearly_income

    user_info['income_to_median'] = np.minimum(user_info.yearly_income / user_info.per_capita_income, 10)

    with open(f'{model_objects}/geo_encoding.pkl', 'rb') as f:
        geo_code_model = pickle.load(f)

    user_info['geo_encoding'] = geo_code_model.predict(user_info[['latitude', 'longitude']])

    logger.info(f'after aggregate cards data by user, the user info data shape is ({user_info.shape[0]},{user_info.shape[1]})')

    logger.info(f'preprocess transaction data, shape is ({transactions.shape[0]},{transactions.shape[1]})')

    transactions = transactions.loc[transactions.target.isin(['No', 'Yes'])]

    transactions.target = transactions['target'].apply(lambda x: 1 if x == 'Yes' else 0)

    logger.info(f'after removing missing targets, shape is ({transactions.shape[0]},{transactions.shape[1]})')

    user_info_keep = ['user_id', 'sum_credit_limit','sum_card_on_dark_web','dti','income_to_median','geo_encoding']

    cards = cards.merge(user_info[user_info_keep], on='user_id', how='left')

    logger.info('Combining cards and transactions data.')

    combine_data = transactions.merge(cards, on=['user_id', 'card_id'], how='left').sort_values(by=['user_id', 'card_id']).reset_index(drop=True)

    logger.info('Creating datetime features')

    dtf = DatetimeFeatures(features_to_extract=["year", "month", 'weekend', 'year_end', 'year_start', 'hour'])

    date_features = dtf.fit_transform(combine_data[['date']])

    combine_data = pd.concat([combine_data, date_features], axis=1)

    yml_file = utilities.read_yaml_file(window_agg_yaml)

    tw_vars = yml_file['time_window_aggregates']

    logger.info('Creating window features')

    con = utilities.register_df_view(combine_data[['user_id','card_type','card_id','date','transaction_id','amount']],'df')

    for var in tw_vars:

        newfeatures = yml_file[var]

        for f in newfeatures:

            params = f.split(';')

            logger.info(f'Creating window features for {var}, params: {params}')

            newcolumns = feature_engineering.create_time_window_agg(con, 'df', [var], 'date', params[0], params[1], params[2])

            mergeby = params[2].split(',')
            mergeby.append('date')

            combine_data = combine_data.merge(newcolumns, on=mergeby, how='left')

    utilities.drop_df_view(con, 'df')

    logger.info(f'saving data to {output_data_path}: shape: ({combine_data.shape[0]},{combine_data.shape[1]})')

    combine_data.to_csv(output_data_path, index=False)


if __name__ == "__main__":
    main()





