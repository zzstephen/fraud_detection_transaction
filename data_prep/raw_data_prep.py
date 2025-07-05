import numpy as np
import pandas as pd
from loguru import logger
import datetime
import sys
sys.path.append('../../../../infrastructure/tools')
from feature_engineering import feature_engineering
from utilities import utilities
import json
from data_config import *
import pickle

def main():
    cards = pd.read_csv(f'{intermediate}/cards.csv')
    transactions = pd.read_csv(f"{intermediate}/transactions.csv")

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

    user_info_agg = utilities.pivot(cards, varlist={'credit_limit': 'sum', 'card_on_dark_web': 'sum'},
                                    by_vars=['user_id'])

    user_info = user_info.merge(user_info_agg, on='user_id')

    user_info['dti'] = user_info.total_debt / user_info.yearly_income

    user_info['income_to_median'] = np.minimum(user_info.yearly_income / user_info.per_capita_income, 10)

    with open(f'{model_objects}/geo_encoding.pkl', 'rb') as f:
        geo_code_model = pickle.load(f)

    user_info['geo_encoding'] = geo_code_model.predict(user_info[['latitude', 'longitude']])

    logger.info(f'after aggregate cards data by user, the user info data shape is ({user_info.shape[0]},{user_info.shape[1]})')

    logger.info(f'preprocess transaction data, shape is ({transactions.shape[0]},{transactions[1]})')

    transactions = transactions.loc[transactions.target.notna()]

    logger.info(f'after removing missing targets, shape is ({transactions.shape[0]},{transactions[1]})')

    transactions.target = transactions['target'].apply(lambda x: 1 if x == 'Yes' else 0)




if __name__ == "__main__":
    main()





