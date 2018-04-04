import json
from pprint import pprint
import pandas as pd

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import KNNBasic
from collections import defaultdict

def get_top_n(predictions, n=10):
  '''Return the top-N recommendation for each user from a set of predictions.

  Args:
    predictions(list of Prediction objects): The list of predictions, as
      returned by the test method of an algorithm.
    n(int): The number of recommendation to output for each user. Default
      is 10.

  Returns:
  A dict where keys are user (raw) ids and values are lists of tuples:
    [(raw item id, rating estimation), ...] of size n.
  '''

  # First map the predictions to each user.
  top_n = defaultdict(list)
  for uid, iid, true_r, est, _ in predictions:
    top_n[uid].append((iid, est))

  # Then sort the predictions for each user and retrieve the k highest ones.
  for uid, user_ratings in top_n.items():
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    top_n[uid] = user_ratings[:n]
  return top_n


def preprocessing(filename):
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
    helpful = review['helpful']
    weight = float(helpful[0]) / (1 if helpful[1] == 0 else helpful[1])
    ratingList.append(float(review['overall']) * weight)
  ratings_dict = {'itemID': itemIDList,
        'userID': userIDList,
        'rating': ratingList}
  df = pd.DataFrame(ratings_dict)
  return df

#filename = "reviews_Electronics_5.json"
filename = "test.json"
df = preprocessing(filename)
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

kf = KFold(n_splits=3)

bsl_options = {'method': 'als',
         'n_epochs': 20,
         }
sim_options = {'name': 'pearson_baseline'}

knn = KNNBasic(bsl_options=bsl_options, sim_options=sim_options)

i = 0
for trainset, testset in kf.split(data):

  # train and test algorithm.
  knn.fit(trainset)
  predictions = knn.test(testset)

  # Compute and print Root Mean Squared Error
  accuracy.rmse(predictions, verbose=True)
  top_n = get_top_n(predictions, n=10)
  # Print the recommended items for each user
  #for uid, user_ratings in top_n.items():
    #print(uid, [iid for (iid, _) in user_ratings])
