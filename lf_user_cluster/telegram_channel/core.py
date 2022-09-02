import os
import pandas as pd
import numpy as np

n_factors = 100


def normalizer(x):
    return x if np.linalg.norm(x) == 0 else x / np.linalg.norm(x)


base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
Y = pd.read_pickle(f"{base_path}/050_word_model_item_factors.pickle")
Y = np.asmatrix(Y)
YtY = Y.T.dot(Y)

item_codes = pd.read_pickle(f"{base_path}/050_word_model_item_codes.pickle")
normalized_centroids = pd.read_pickle(f"{base_path}/100_clusters_centroids.pickle")

normalized_centroids = pd.DataFrame(normalized_centroids)
normalized_centroids = normalized_centroids.apply(lambda x: normalizer(x), axis=1)

thresholds = pd.read_pickle(f"{base_path}/120_agg_analysis_table.pickle")


def inference(df: pd.DataFrame) -> pd.DataFrame:
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
            b += (confidence) * factor.T
        return A, b

    def the_text_factor(Y, YtY, final_text_df, regularization, n_factors):
        # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
        A, b = the_text_linear_equation(Y, YtY, final_text_df, regularization, n_factors)
        return np.linalg.solve(A, b)

    def the_text_factor_recalculator(final_text_df):
        return the_text_factor(Y, YtY, final_text_df, 0.01, n_factors)

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

    df.posts = df.posts.astype(str).str.split()
    df["factors"] = df['posts'].apply(recalculator2)
    df["factors"] = df["factors"].apply(normalizer)
    df["factors"] = df["factors"].apply(lambda x: np.array(x).reshape(-1, ).tolist())

    def subject_finder(my_factor):
        my_factor_sim = normalized_centroids.apply(lambda s: float(np.matmul(np.matrix(s), my_factor)), axis=1)
        my_factor_sim = pd.DataFrame(my_factor_sim).sort_values(0, ascending=False)
        my_factor_sim[1] = thresholds['treshhold']
        my_factor_sim = my_factor_sim.loc[my_factor_sim[0] > my_factor_sim[1]]
        return list(my_factor_sim.index[0:3])

    df["group"] = df["factors"].apply(subject_finder)
    df['group'] = df['group'].apply(lambda x: x if len(x) else [1000])

    return df.drop(columns=['factors', 'posts'])
