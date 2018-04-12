# In[3]:

# encoding = utf-8
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
from surprise import KNNBaseline
from surprise import SVDpp
from collections import defaultdict
from surprise.model_selection import train_test_split
import sparse_data as sp
from surprise.model_selection import PredefinedKFold


# In[7]:

### preprecess

filename = "test.json"
trainFile = "train.txt"
testFile = "test.txt"
#filename = "reviews_Electronics_5.json"

# extract data from filename and generate trainset file and testset file.
def preprocess(filename, trainFile, testFile):
    raw_data = sp.sparse_data(filename)
    train_output = trainFile
    train_f = open(train_output,"w")
    train_row_ind = raw_data.get_train_row_list()
    train_col_ind = raw_data.get_train_col_list()
    for i in range(len(train_row_ind)):
        row = train_row_ind[i]
        col = train_col_ind[i]
        user = raw_data.get_userID(row)
        item = raw_data.get_itemID(col)
        value = raw_data.get_val(row, col, 'rating')
        train_f.write(user+"\t"+item+"\t"+str(value)+"\n")
    train_f.close()
    
    test_output = testFile
    test_f = open(test_output, "w")
    test_row_ind = raw_data.get_test_row_list()
    test_col_ind = raw_data.get_test_col_list()
    for i in range(len(test_row_ind)):
        row = test_row_ind[i]
        col = test_col_ind[i]
        user = raw_data.get_userID(row)
        item = raw_data.get_itemID(col)
        value = raw_data.get_val(row, col, 'rating')
        test_f.write(user+"\t"+item+"\t"+str(value)+"\n")
    test_f.close()
   
preprocess(filename, trainFile, testFile)
folds_files = [(trainFile, testFile)]
reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()


# In[5]:

### set Paramter(baseline and similarity)

### bsl: sgd   als
bsl_options = {'method': 'sgd',
               'n_epochs': 20,
               'learning_rate': 0.05,
               }
### sim name: cosine    msd       pearson     pearson_baseline
### user_based : True ---- similarity will be computed based on users
###            : False ---- similarity will be computed based on items.
sim_options = {'name': 'cosine',
              'user_based':True}


# In[6]:

### recommendation
def get_top_n(predictions, n=10, threshold = 3.0):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
        #top_n[uid] = [rate for rate in user_ratings if rate[1] >= threshold]
    return top_n

def calculate_percision(top_n, testSet):
    # get true recommend set
    trueItemSet = {}
    for user,item,score in testSet:
        if user not in trueItemSet:
            trueItemSet[user] = set()
        trueItemSet[user].add(item)
    # get prediction recommend set
    predictionSet = {}
    for user, user_ratings in top_n.items():
        if user not in predictionSet:
            predictionSet[user] = set()
        if len(user_ratings) == 0:
            continue
        predictionSet[user].add(user_ratings[0])
    # calculate match rate
    matchCount = 0.0
    countDic = {}
    temp = 0
    for k, v in predictionSet.iteritems():
        if k not in trueItemSet:
            countDic[k] = 0
            continue
        temp += 1
        pSet = trueItemSet[k]
        count = 0.0
        for ele in v:
            if ele in pSet:
                count+=1
                matchCount+=1
        if count > 0:
            print count
        countDic[k] = count / len(v)
    return countDic, matchCount / len(predictionSet)

### train

### model selection
### prediction module:  KNNBasic     KNNWithMeans      SVD

#algo = KNNBaseline(bsl_options=bsl_options, sim_options=sim_options)

# combine neighboors' value to calculate similarity
#algo = KNNBasic(bsl_options=bsl_options, sim_options=sim_options) 
for trainset, testset in pkf.split(data):
    algo = SVD()
    #algo = KNNBaseline(bsl_options=bsl_options, sim_options=sim_options)
    algo.fit(trainset)
    ### test
    predictions = algo.test(testset)
    ### accuracy:   rmse(root mean square error)     mae(mean absolute error)      fcp
    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
    top_n = get_top_n(predictions, n=10)
    # Print the recommended items for each user
    #for uid, user_ratings in top_n.items():
    #    print(uid, [(iid, rate ) for (iid, rate) in user_ratings])
    precisions, overall = calculate_percision(top_n, testset)
    print overall
    #print precisions
