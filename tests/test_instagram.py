#!/usr/bin/env python
import os
import inspect
import sys
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from lf_user_cluster.instagram.data_utils import load_instagram_data
from lf_user_cluster.instagram.core import inference as instagram_telegram_inference

def test_instagram():
        user_id = '1398775624'

        following_list = str(['1973047319', '1728391520', '1228060097','2031021351','1381895679','1709359656','346625770','5704466376','1243677235','5840322063','234026189','1531611069',
                              '840698077','1949382920','496990363','1111634433','3691146947','2052961839','30916040','2897199222','494310035','1635686295','2128295586','1728391520'])

        instagram_data_frame = load_instagram_data()
        output = instagram_telegram_inference(user_id, following_list, instagram_data_frame, n_factors=50)
           
        data_path = f'{os.path.dirname(os.path.abspath(__file__))}/'

        # output.to_pickle(f'{data_path}test.pickle')

        test = pd.read_pickle(f'{data_path}test.pickle')
        
        assert ((output['group'] == test['group']).all())
        
test_instagram()