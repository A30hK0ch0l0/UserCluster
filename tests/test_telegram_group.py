from lf_user_cluster.telegram.main import get_users, categorize_df
import os
import ast
import pandas as pd

data_path = f'{os.path.dirname(os.path.abspath(__file__))}'

def get_users():
    df = pd.read_excel(f'{data_path}/test users set.xlsx')
    df.rename(columns={'(id)': 'user_id', '(group id)': 'group_ids'}, inplace=True)
    df['group_ids'] = df['group_ids'].apply(lambda s: list(ast.literal_eval(s)))
    return df


def test_pickle():
    input_df = get_users()
    result = {'user_id': 64678117, 'user_categories': [71, 101, 98]}
    output_df = categorize_df(input_df)

    assert output_df[['user_id', 'user_categories']].iloc[0].to_dict() == result



