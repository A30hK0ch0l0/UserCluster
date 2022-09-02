import numpy as np
import pandas as pd
from .utilities.DataModel import DataModel
from .utilities.UserClusterExceptions import UpdateError, PlatformError


class Twitter:

    def __init__(self, n_factors: int = 100):
        """
        :param n_factors: Vector length
        """
        try:
            self._n_factors = n_factors

            model = DataModel()
            self._item_codes = model.item_codes()
            self._item_factors = model.item_factors()
            self._thresholds = model.thresholds()
            model.centroids()
            self._item_factors_t = model.item_factors_t()
            self._item_codes2 = model.item_codes2()
            self._item_factors2 = model.item_factors2()
            self._item_factors_t2 = model.item_factors_t2()
            model.alternative_words()
            model.categories_one()
            self._categories_two = model.categories_two()
            self._categories_two['vector_of_cluster'] = self._categories_two['valid_keywords'].apply(self.cluster_vector_creator)
            self._categories_two = self._categories_two.reset_index(drop=True)
            self._categories_three = self._categories_two[['faction', 'vector_of_cluster']]
        except UpdateError:
            print("Exception: Update is failed!", 'error')
        except PlatformError:
            print("Exception: Platform doesn't selected correctly!", 'error')

    def dict_to_dataframe(self, _dict):
        """
        :param _dict: A dict of key values
        :return: Convert dictionary to dataframe
        """
        data_frame = pd.DataFrame([_dict.keys(), _dict.values()]).T
        data_frame = data_frame.merge(self._item_codes, left_on=0, right_on='item')[['item_code', 1]]
        data_frame['item_code'] = data_frame['item_code'].apply(lambda q: str(q))
        data_frame[1] = data_frame[1].apply(lambda q: (q / np.log(q + np.e)))
        data_frame = data_frame.T
        data_frame = data_frame.reset_index(drop=True)
        return data_frame

    def list_to_dataframe(self, _list):
        """
        :param _list: A list of values
        :return: Convert list to dataframe
        """
        data_frame = pd.DataFrame(_list)
        data_frame = data_frame.merge(self._item_codes, left_on=0, right_on='item')['item_code']
        data_frame = pd.DataFrame(data_frame)
        data_frame['freq'] = 1
        data_frame = data_frame.T
        data_frame = data_frame.reset_index(drop=True)
        return data_frame

    def text_linear_equation(self, separated_text_with_frequency: pd.DataFrame, regularization):
        """
        :param separated_text_with_frequency: A separated text with frequency
        :param regularization: Regularization parameter for equation
        :return: Calculating text linear equation
        """
        # Xu = (YtCuY + regularization * I) ^ -1 (YtCuPu)
        # YtCuY + regularization * I = Yt + regularization * I + Yt (Cu-I)
        # accumulate YtCuY + regularization * I in A
        A = self._item_factors_t + regularization * np.eye(self._n_factors)
        # accumulate YtCuPu in b
        b = np.zeros(self._n_factors)
        b = np.asmatrix(b)
        b = b.reshape(self._n_factors, 1)
        for t in range(len(separated_text_with_frequency.columns)):
            i = int(separated_text_with_frequency[t][0])
            confidence = separated_text_with_frequency[t][1]
            factor = self._item_factors[i]
            A += (confidence - 1) * np.outer(factor, factor)
            b += confidence * factor.T
        return A, b

    def text_factor_recalculate(self, separated_text_with_frequency: pd.DataFrame):
        """
        :param separated_text_with_frequency: Get separated text with frequency
        :return: Calculate text factor based on frequency
        """
        regularization = .01
        A, b = self.text_linear_equation(separated_text_with_frequency, regularization)
        return np.linalg.solve(A, b)

    def user_vector_creator(self, user_dictionary) -> dict:
        """
        :param user_dictionary: Get user hashtag dict
        :return: Convert dict to dataframe
        """
        return self.text_factor_recalculate(self.dict_to_dataframe(user_dictionary))

    def cluster_vector_creator(self, cluster_list):
        """
        :param cluster_list: Get a user hashtag vector
        :return: Get user cluster from vector
        """
        return self.text_factor_recalculate(self.list_to_dataframe(cluster_list))

    @staticmethod
    def cosine_similarity(x, y):
        """
        :param x: X point
        :param y: Y point
        :return: Calculate cosine similarity between X and Y
        """
        return (x.T @ y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def cluster_finder(self, vector):
        """
        :param vector: Get a vector
        :return: Find user cluster
        """
        temp_categories = self._categories_three.copy()
        temp_categories['sim'] = temp_categories['vector_of_cluster'].apply(lambda q: float(self.cosine_similarity(q, vector)))
        temp_categories = temp_categories.sort_values('sim', ascending=False).head(2)
        temp_categories = temp_categories.loc[temp_categories['sim'] > .25]
        return ['others'] if len(temp_categories) == 0 else list(temp_categories['faction'])

    @staticmethod
    def _convert_list_to_dict(keys: list, values: dict) -> dict:
        data = {}
        for index, key in enumerate(keys):
            data[key] = {}
            for value in values:
                if type(values[value][index]) is np.matrix:
                    data[key][value] = np.concatenate(values[value][index]).ravel().tolist()[0]
                else:
                    data[key][value] = values[value][index]
        return data

    def calculate_cluster(self, user_df):
        """
        retweeted_hashtags: hashtags of each user in his retweet posts
        hashtags: hashtags of each user in his tweet posts
        """
        # user is inactive
        if user_df is None or len(user_df) == 0:
            return [1001]

        weights = {'hashtags': 4, 'retweeted_hashtags': 1, 'origin_users': 1}
        user_df['weights'] = user_df['item_type'].apply(lambda x: weights[x])
        user_df['confidency'] = user_df['freq'] * user_df['weights']
        user_df['confidency'] = user_df['confidency'].apply(lambda x: int(x))
        user_df['item'] = user_df['item'].apply(lambda x: [x])
        user_df['item'] = user_df.apply(lambda x: x['item'] * x['confidency'], axis=1)
        user_df = user_df[['item']]
        user_df = list(user_df['item'])
        user_df = [a for sublist in user_df for a in sublist]
        user_df = self.recalculator2(user_df)
        return self.clusters_finder(user_df)

    def the_text_linear_equation(self, Y, YtY, final_text_df, regularization, n_factors):
        # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
        # YtCuY + regularization * I = YtY + regularization * I + Yt(Cu-I)
        # accumulate YtCuY + regularization*I in A
        A = YtY + regularization * np.eye(n_factors)

        # accumulate YtCuPu in b
        b = np.zeros(self._n_factors)
        b = np.asmatrix(b)
        b = b.reshape(self._n_factors, 1)
        for t in range(len(final_text_df.columns)):
            i = int(final_text_df[t][0])
            confidence = final_text_df[t][1]
            factor = Y[i]
            A += (confidence - 1) * np.outer(factor, factor)
            b += confidence * factor.T
        return A, b

    def the_text_factor(self, Y, YtY, final_text_df, regularization, n_factors):
        # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
        A, b = self.the_text_linear_equation(Y, YtY, final_text_df, regularization, n_factors)
        return np.linalg.solve(A, b)

    def the_text_factor_recalculator(self, final_text_df):
        return self.the_text_factor(self._item_factors2, self._item_factors_t2, final_text_df, .01, self._n_factors)

    def recalculator2(self, x):
        if len(x) > 0:
            x = pd.DataFrame(x)
            x.columns = ['item']
            x = x.groupby('item').agg({'item': 'count'})
            x.columns = ['freq']
            x = x.reset_index()
            x = x.merge(self._item_codes2, on='item')
            if len(x) > 0:
                x = x[['item_code', 'freq']]
                x = x.reset_index(drop=True)
                x = x.T
                x = x.reset_index(drop=True)
                return self.the_text_factor_recalculator(x)
            else:
                zz = np.zeros(self._n_factors)
                zz = np.asmatrix(zz)
                zz = zz.reshape(self._n_factors, 1)
                return zz
        else:
            zz = np.zeros(self._n_factors)
            zz = np.asmatrix(zz)
            zz = zz.reshape(self._n_factors, 1)
            return zz

    def clusters_finder(self, my_factor):
        my_factor_sim = self._thresholds['centroids'].apply(lambda s: float(np.matmul(s, my_factor)))
        my_factor_sim = pd.DataFrame(my_factor_sim).sort_values('centroids', ascending=False)
        my_factor_sim['treshold'] = self._thresholds['treshold']
        my_factor_sim = my_factor_sim.loc[my_factor_sim['centroids'] > my_factor_sim['treshold']]
        clusters = list(my_factor_sim.index[0:2])
        return [1000] if not clusters else clusters
