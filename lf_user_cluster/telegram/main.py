import pandas as pd
import numpy as np
import os
import ast


from .recalculator import recalculator2
from .clustering import similar_item_finder


data_path = f'{os.path.dirname(os.path.abspath(__file__))}/data'

def get_users():
    df = pd.read_excel(f'{data_path}/test users set.xlsx')
    df.rename(columns={'(id)': 'user_id', '(group id)': 'group_ids'}, inplace=True)
    df['group_ids'] = df['group_ids'].apply(lambda s: list(ast.literal_eval(s)))
    return df




def normalizer(x):
    return x / np.linalg.norm(x)



def categorize(group_ids):
    try:
        user_vector = recalculator2(group_ids)
        user_vector = normalizer(user_vector)
        similar_items = similar_item_finder(user_vector)
        similar_items = similar_items[similar_items.sim > 0.2]
        similar_items = similar_items[:3]
        if not list(similar_items.group_code):
            return 1,[1000]
        return user_vector, list(similar_items.group_code)

    except:
        return [1,'others']


def categorize_df(user_df):
    #next line must be uncommented if the users vectors are needed as output
    #user_df['user_vector'] = user_df.group_ids.apply(lambda x: categorize(x)[0])
    user_df['user_categories'] = user_df.group_ids.apply(lambda x: categorize(x)[1])
    return user_df
