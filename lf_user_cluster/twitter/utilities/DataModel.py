import os
import numpy as np
import pandas as pd


class DataModel:

    def __init__(self):
        self.__samples = None
        self.__categories = None
        self.__item_codes = None
        self.__item_codes2 = None
        self.__thresholds = None
        self.__centroids = None
        self.__item_factors = None
        self.__item_factors2 = None
        self.__alternative_words = None
        self.path = os.path.join(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}', 'data')
        self.common_path = f"{self.path}/common"

    @staticmethod
    def __half_space_to_equal_sign(text):
        """
        :param text: Get a text
        :return: Change half space in text to equal sign
        """
        return text.replace('=', '‌')

    @staticmethod
    def __convert_text_to_list(text):
        """
        :param text: Get a text
        :return: Convert a text to list
        """
        text = text.split('"')
        text = list(set(text))
        if '' in text:
            text.remove('')
        if ' ' in text:
            text.remove(' ')
        if '  ' in text:
            text.remove('  ')
        return text

    def __keyword_validator(self, keyword_list):
        """
        :param keyword_list: List of keywords
        :return: Intersection keywords and item codes list and remove not exist keywords
        """
        if self.__item_codes is None:
            self.item_codes()
        return list(set(keyword_list).intersection(set(self.__item_codes['item'])))

    def item_codes(self) -> pd.DataFrame:
        if self.__item_codes is None:
            csv_path = f'{self.path}/item_codes.csv'
            self.__item_codes = pd.read_csv(csv_path)
        return self.__item_codes

    def item_factors(self) -> np.matrix:
        if self.__item_factors is None:
            csv_path = f'{self.path}/item_factors.csv'
            self.__item_factors = np.asmatrix(pd.read_csv(csv_path))
        return self.__item_factors

    def item_factors_t(self) -> np.matrix:
        if self.__item_factors is None:
            self.item_factors()
        return self.__item_factors.T.dot(self.__item_factors)

    def thresholds(self) -> pd.DataFrame:
        if self.__thresholds is None:
            pickle_path = f'{self.path}/thresholds.pickle'
            self.__thresholds = pd.read_pickle(pickle_path)
        return self.__thresholds

    # maybe is extra
    def centroids(self) -> pd.DataFrame:
        if self.__centroids is None:
            pickle_path = f'{self.path}/thresholds.pickle'
            df = pd.read_pickle(pickle_path)
            self.__centroids = df[['centroids']]
        return self.__centroids

    def item_codes2(self) -> pd.DataFrame:
        if self.__item_codes2 is None:
            pickle_path = f'{self.path}/item_codes.pickle'
            self.__item_codes2 = pd.read_pickle(pickle_path)
        return self.__item_codes2

    def item_factors2(self) -> np.matrix:
        if self.__item_factors2 is None:
            pickle_path = f'{self.path}/item_factors.pickle'
            file = pd.read_pickle(pickle_path)
            self.__item_factors2 = np.asmatrix(file)
        return self.__item_factors2

    def item_factors_t2(self) -> np.matrix:
        if self.__item_factors2 is None:
            self.item_factors2()
        return self.__item_factors2.T.dot(self.__item_factors2)

    def alternative_words(self) -> dict:
        if self.__alternative_words is None:
            csv_path = f'{self.common_path}/alternative_words.csv'
            df = pd.read_csv(csv_path)
            self.__alternative_words = df.to_dict('records')
        return self.__alternative_words

    def categories_one(self):
        csv_path = f'{self.path}/categories.csv'
        categories = pd.read_csv(csv_path)
        categories = categories.T
        categories = categories.drop(0, axis=1)
        categories = categories.fillna('empty')
        self.__categories = categories
        return categories

    def categories_two(self):
        categories = pd.DataFrame(self.__categories.index)
        categories = categories.reset_index()
        categories.columns = ['id', 'faction']
        categories['hashtag'] = categories['id'].apply(self.list_creator)
        categories['valid_keywords'] = categories['hashtag'].apply(self.__keyword_validator)
        categories['len_valid'] = categories['valid_keywords'].apply(lambda x: len(x))
        categories = categories.loc[categories['len_valid'] > 0]
        return categories

    def list_creator(self, index):
        category = self.__categories.iloc[index]
        category = list(set(list(category)) - {'empty'})
        return category

    def samples(self, count) -> pd.DataFrame:
        if self.__samples is None:
            csv_path = f'{self.path}/samples.csv'
            self.__samples = pd.read_csv(csv_path).sample(count)
            self.__samples.reset_index(drop=True)
        return self.__samples
