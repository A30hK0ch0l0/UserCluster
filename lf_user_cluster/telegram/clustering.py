import pandas as pd
import numpy as np
import pickle
import os

data_path = f'{os.path.dirname(os.path.abspath(__file__))}/data'


def representatives_matrix():
    with open(f'{data_path}/160-factorized representatives normalized.pickle', "rb") as myfile:
        representatives_df = pickle.load(myfile)
    return np.asmatrix(representatives_df)


def category_codes():
    category_codes = pd.read_pickle(f'{data_path}/category_codes.pickle')
    category_codes.drop('Unnamed: 0', axis=1, inplace=True)
    return category_codes


def userid_code():
    userid_code = pd.read_pickle(f'{data_path}/document_codes.pickle')

    return userid_code


def user_matrix():
    users_df = pd.read_pickle(f'{data_path}/users_factors_normalized.pickle')
    return users_df, np.asmatrix(users_df)


representatives_matrix = representatives_matrix()
users_df, user_matrix = user_matrix()
userid_code = userid_code()
category_codes = category_codes()


def similar_item_finder(user_vector):
    try:
        sim_df = pd.DataFrame(np.matmul(representatives_matrix, user_vector))
        sim_df = sim_df.reset_index()
        sim_df.columns = ['group_code', 'sim']
        sim_df = sim_df.sort_values('sim', ascending=False)
        sim_df = sim_df.reset_index(drop=True)
        sim_df = sim_df.merge(category_codes, on='group_code')
    except:
        pass

    return sim_df
