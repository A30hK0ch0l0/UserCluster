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
            self._item_factors_t = model.item_factors_t()
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

    def set_user_cluster(self, items: list):
        # user is inactive
        if items is None or len(items) == 0:
            return [1001]

        hashtags = {item['item']: item['freq'] for item in items}
        vector = self.user_vector_creator(hashtags)
        user_cluster = self.cluster_finder(vector=vector)
        return user_cluster
