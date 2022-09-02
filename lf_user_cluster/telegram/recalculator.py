import pandas as pd
import numpy as np
import os
import pickle

data_path = f'{os.path.dirname(os.path.abspath(__file__))}/data'
n_factors = 100


def build_Y():
    with open(f'{data_path}/groups_factorized_normalized.pickle', "rb") as myfile:
        Y = pickle.load(myfile)
    Y = np.asmatrix(Y)
    return Y


def read_item_codes():
    return pd.read_pickle(f'{data_path}/item_codes.pickle')


def the_text_linear_equation(Y, YtY, final_text_df, regularization, n_factors):
    # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
    # YtCuY + regularization * I = YtY + regularization * I + Yt(Cu-I)
    # accumulate YtCuY + regularization*I in A
    A = YtY + regularization * np.eye(n_factors)

    # accumulate YtCuPu in b
    b = np.zeros(n_factors)
    b = np.asmatrix(b)
    b = b.reshape(n_factors, 1)
    for t in range(len(final_text_df.columns)):
        i = int(final_text_df[t][0])
        confidence = final_text_df[t][1]
        factor = Y[i]
        A += (confidence - 1) * np.outer(factor, factor)
        b += confidence * factor.T
    return A, b


def the_text_factor(Y, YtY, final_text_df, regularization, n_factors):
    # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
    A, b = the_text_linear_equation(Y, YtY, final_text_df, regularization, n_factors)
    return np.linalg.solve(A, b)


def the_text_factor_recalculator(final_text_df):
    return the_text_factor(Y, YtY, final_text_df, .01, n_factors)


def recalculator2(x):
    x = pd.DataFrame(x)
    x.columns = ['item']
    x = x.groupby('item').agg({'item': 'count'})
    x.columns = ['freq']
    x = x.reset_index()
    x = x.merge(item_codes, on='item')
    if len(x) > 0:
        x = x[['item_code', 'freq']]
        x = x.reset_index(drop=True)
        x = x.T
        x = x.reset_index(drop=True)
        return the_text_factor_recalculator(x)
    else:
        zz = np.zeros(n_factors)
        zz = np.asmatrix(zz)
        zz = zz.reshape(n_factors, 1)
        return zz


Y = build_Y()
YtY = Y.T.dot(Y)
item_codes = read_item_codes()
