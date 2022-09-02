import pandas as pd
import numpy as np
import ast
from .data_utils import normalizer


def inference(user_id, following_list, platform_data_frame, n_factors) -> tuple:
    user_data_frame = prepare_data(user_id, following_list)
    user_data_frame["factors"] = user_data_frame['following_user_id'].apply(recalculator2, platform_data_frame=platform_data_frame, n_factors=n_factors)
    user_data_frame["factors"] = user_data_frame["factors"].apply(normalizer)
    user_data_frame["factors"] = user_data_frame["factors"].apply(lambda x: np.array(x).reshape(-1, ).tolist())
    user_data_frame["group"] = user_data_frame["factors"].apply(subject_finder, platform_data_frame=platform_data_frame)
    

    if not user_data_frame["group"][0]:
        user_data_frame["group"][0] = [1000]

    return user_data_frame



def prepare_data(user_id, following_list):
    user_data_frame = pd.DataFrame([user_id, following_list]).T
    user_data_frame.columns = ["user_id", "following_user_id"]
    user_data_frame["following_user_id"] = user_data_frame["following_user_id"].apply(lambda x: ast.literal_eval(x))
    

    return user_data_frame
    


def subject_finder(my_factor, platform_data_frame):
    my_factor_sim = platform_data_frame['normalized_centroids'].apply(
        lambda s: float(np.matmul(np.matrix(s), my_factor)), axis=1)
    my_factor_sim = pd.DataFrame(my_factor_sim).sort_values(0, ascending=False)
    my_factor_sim[1] = platform_data_frame['thresholds']['treshhold']
    my_factor_sim = my_factor_sim.loc[my_factor_sim[0] > my_factor_sim[1]]
    return list(my_factor_sim.index[0:3])


def recalculator2(x, platform_data_frame, n_factors):
    if type(x[0]) != str:
        x = list(map(str, x))

    x = pd.DataFrame(x)
    x.columns = ['item']
    x = x.groupby('item').agg({'item': 'count'})
    x.columns = ['freq']
    x = x.reset_index()
    x = x.merge(platform_data_frame['item_codes'], on='item')
    if len(x) > 0:
        x = x[['item_code', 'freq']]
        x = x.reset_index(drop=True)
        x = x.T
        x = x.reset_index(drop=True)
        return the_text_factor_recalculator(x, platform_data_frame, n_factors)
    else:
        zz = np.zeros(n_factors)
        zz = np.asmatrix(zz)
        zz = zz.reshape(n_factors, 1)
        return zz


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


def the_text_factor_recalculator(final_text_df, platform_data_frame, n_factors):
    return the_text_factor(platform_data_frame['Y'], platform_data_frame['YtY'], final_text_df, 0.01, n_factors)

