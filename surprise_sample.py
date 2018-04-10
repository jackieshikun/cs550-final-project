# encoding = utf-8
import json
from pprint import pprint
import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import KNNBasic
from surprise import KNNBaseline
from surprise import SVDpp
from collections import defaultdict
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split


# In[60]:

### preprecess

filename = "test.json"

def preprocess(filename):
  jsonFile = file(filename, 'r')
  data = jsonFile.read()
  lines = data.split('\n')
  itemIDList = []
  userIDList = []
  ratingList = []
  for line in lines:
    if len(line) == 0:
      continue;
    review = json.loads(line)
    itemIDList.append(review['asin'])
    userIDList.append(review['reviewerID'])
    ratingList.append(float(review['overall']))
  ratings_dict = {'userID': userIDList,
                    'itemID': itemIDList,
                    'rating': ratingList}
  df = pd.DataFrame(ratings_dict)
  return df
#filename = "reviews_Electronics_5.json"
df = preprocess(filename)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

# split data into trainset and testset
trainset, testset = train_test_split(data, test_size =.25)


# In[61]:

### set Paramter(baseline and similarity)

### bsl: sgd   als
bsl_options = {'method': 'sgd',
               'n_epochs': 50,
               'learning_rate': 0.05,
               }
### sim name: cosine    msd       pearson     pearson_baseline
### user_based : True ---- similarity will be computed based on users
###            : False ---- similarity will be computed based on items.
sim_options = {'name': 'cosine',
              'user_based':True}


# In[62]:

### train

### model selection
### prediction module:  KNNBasic     KNNWithMeans      SVD

#algo = KNNBaseline(bsl_options=bsl_options, sim_options=sim_options)

# combine neighboors' value to calculate similarity
#algo = KNNBasic(bsl_options=bsl_options, sim_options=sim_options) 

algo = SVD()

algo.fit(trainset)


# In[63]:

### test
predictions = algo.test(testset)

### accuracy:   rmse(root mean square error)     mae(mean absolute error)      fcp
# Compute and print Root Mean Squared Error
accuracy.rmse(predictions, verbose=True)


# In[80]:

### recommendation
def get_top_n(predictions, n=10, threshold = 4.0):
  # First map the predictions to each user.
  top_n = defaultdict(list)
  for uid, iid, true_r, est, _ in predictions:
    top_n[uid].append((iid, est))
  # Then sort the predictions for each user and retrieve the k highest ones.
  for uid, user_ratings in top_n.items():
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    #top_n[uid] = user_ratings[:n]
    top_n[uid] = [rate for rate in user_ratings if rate[1] >= threshold]
  return top_n
top_n = get_top_n(predictions, n=10, threshold = 4.1)
# Print the recommended items for each user
for uid, user_ratings in top_n.items():
  print(uid, [(iid, rate ) for (iid, rate) in user_ratings])


# In[ ]:



