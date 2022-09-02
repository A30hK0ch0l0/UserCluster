# Install

`PROJECT_DIR` is project root

```bash
mkdir ${PROJECT_DIR}
cd ${PROJECT_DIR}

sudo apt update -y
sudo apt install -y python3 python3-venv git

git clone http://185.208.77.246/inference/user_cluster.git .
# git checkout develop

python3 -m venv venv
source venv/bin/activate
pip install -U pip wheel setuptools pytest
pip install -r requirements.txt
```

# Test

```bash
bin/test.sh

# or

source venv/bin/activate
pytest
```

## Install package for other projects

```bash
pip install git+http://185.208.77.246/inference/user_cluster.git
```

# User manual Telegram


```python
from lf_user_cluster.telegram import categorize_df


output=categorize_df(df)
print (output)
```



# Input

Input is a dataframe with two columns. One column for user id and one column
for users telegram groups ids as a list.

# Input sample


|    | user_id  | group_ids                                        |
|---:|---------:|:-------------------------------------------------|
|  0 | 64678117 | [1102097946, 1224977047, 1469876648]             |
|  1 |  1206825 | [1168800545, 1492989094, 1172972624, 1430352171] |
|  2 | 33731544 | [1560496829, 1179328959]                         |


# Output sample

|    | user_id  | group_ids                                             |user_categories |
|---:|---------:|:------------------------------------------------------|:---------------|
|  0 | 64678117 | [1102097946, 1224977047, 1469876648]                  | [101, 71, 43]  |
|  1 |  1206825 | [1168800545, 1492989094, 1172972624, 1430352171]      | [22, 98, 9]    |
|  2 | 33731544 | [1560496829, 1179328959]                              | [57, 49, 83]   |


# User manual Instagram

```python

from lf_user_cluster.instagram.data_utils import load_instagram_data
from lf_user_cluster.instagram.core import inference as instagram_telegram_inference

user_id = '1398775624' #input user_id
following_list = str(['1973047319', '1728391520', '1228060097','2031021351','1381895679']) #input following_list
instagram_data_frame = load_instagram_data()
output = instagram_telegram_inference(user_id, following_list, instagram_data_frame, n_factors=50)

print(output)




```

# input sample

|    | user id    | following_list                                                                       |
|---:|-----------:|:-------------------------------------------------------------------------------------|
|  0 | 1398775624 |['1973047319', '1728391520', '1228060097','2031021351','1381895679']                  | 
|  1 | 2031021351 |['1709359656','1580626466','2234518885','747759342','1712989288','5704466376']        |
|  2 | 2031021351 |['11554198778 ,5808152611,2228941700 ,1546029163 ,2292561937 ,1102143604,5345655469 ] |
|  3 | 2985364515 |[2985364515 ,1604573292 ,1611337119, 3268472908 ,1026023571,1503837870, 208339720 ,392707863 ,2302481482 ,704990164 ] |
| 4  |1307754900  |['1307754900, 2073400443, 621070368 ,638542590 ,1568256827 ,2113309130, 2160161303 ,1546029163 ,1921753564 ,3417181397 ,1979951366 ,4567237310, 1973105856 ,2222938775 ,1261629597,2131045696 ,1696836816 ,683252347, 1480642521 ,5790313423 ,2125263122 ] |   

# output sample
|    | user id    | following_list                                                                                                                                                                                                                                             |  categories code|
|---:|-----------:|:-----------------------------------------------------------------------------------                                                                                                                                                                        |:----------------|
|  0 | 1398775624 |['1973047319', '1728391520', '1228060097','2031021351','1381895679']                                                                                                                                                                                        |  [2,15,29]      |
|  1 | 2031021351 |['1709359656','1580626466','2234518885','747759342','1712989288','5704466376']                                                                                                                                                                              |   [2,15]        |
|  2 | 2031021351 |['11554198778 ,5808152611,2228941700 ,1546029163 ,2292561937 ,1102143604,5345655469]                                                                                                                                                                        |     [5]         |
|  3 | 2985364515 |[2985364515 ,1604573292 ,1611337119, 3268472908 ,1026023571,1503837870, 208339720 ,392707863 ,2302481482 ,704990164 ]                                                                                                                                       |     [38]        |
|  4 | 1307754900 |['1307754900, 2073400443, 621070368 ,638542590 ,1568256827 ,2113309130, 2160161303 ,1546029163 ,1921753564 ,3417181397 ,1979951366 ,4567237310, 1973105856 ,2222938775 ,1261629597,2131045696 ,1696836816 ,683252347, 1480642521 ,5790313423 ,2125263122 ]  |     [39]        |


