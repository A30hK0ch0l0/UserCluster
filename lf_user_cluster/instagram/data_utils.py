import pandas as pd
import numpy as np
import os


def normalizer(x):
    if np.linalg.norm(x) == 0:
        return x
    else:
        return x / np.linalg.norm(x)

def get_data_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


def load_instagram_data():
    base_path = get_data_path()
    Y = pd.read_pickle(f"{base_path}/040_word_model_item_factors.pickle")
    instagram_user = dict()
    Y = np.asmatrix(Y)

    instagram_user['YtY'] = Y.T.dot(Y)

    instagram_user['Y'] = Y

    instagram_user['item_codes'] = pd.DataFrame(pd.read_pickle(f"{base_path}/040_word_model_item_codes.pickle"))
    normalized_centroids = pd.read_pickle(f"{base_path}/060_document_70clusters_centroids.pickle")

    normalized_centroids = pd.DataFrame(normalized_centroids)
    instagram_user['normalized_centroids'] = normalized_centroids.apply(lambda x: normalizer(x), axis=1)

    instagram_user['thresholds'] = pd.read_pickle(f"{base_path}/120_agg_analysis_table.pickle")
    return instagram_user
