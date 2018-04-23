import json
from pprint import pprint
import pandas as pd
import numpy as np
import time

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
import heapq
import math

def preprocess(filename, trainFile, testFile):
    raw_data = sp.sparse_data(filename)
    train_output = trainFile
    userPurchasedSet = {}
    train_f = open(train_output,"w")
    
    train_row_ind = raw_data.get_train_row_list()
    train_col_ind = raw_data.get_train_col_list()
    for i in range(len(train_row_ind)):
        row = train_row_ind[i]
        col = train_col_ind[i]
        user = raw_data.get_userID(row)
        item = raw_data.get_itemID(col)
        if user not in userPurchasedSet:
            userPurchasedSet[user] = set()
        userPurchasedSet[user].add(item)
        value = raw_data.get_val(row, col, 'rating')
        train_f.write(user+"\t"+item+"\t"+str(value)+"\n")
    train_f.close()
    
    userTrueTestSet = {}
    test_output = testFile
    test_f = open(test_output, "w")
    test_row_ind = raw_data.get_test_row_list()
    test_col_ind = raw_data.get_test_col_list()
    for i in range(len(test_row_ind)):
        row = test_row_ind[i]
        col = test_col_ind[i]
        user = raw_data.get_userID(row)
        if user not in userTrueTestSet:
            userTrueTestSet[user] = set()
        item = raw_data.get_itemID(col)
        userTrueTestSet[user].add(item)
        value = raw_data.get_val(row, col, 'rating')
        test_f.write(user+"\t"+item+"\t"+str(value)+"\n")
    test_f.close()
    return raw_data, userPurchasedSet, userTrueTestSet

def calculate_precision(recommendSet, trueSet):
    count = 0.0
    for item in trueSet:
        if item in recommendSet:
            count += 1
    return count / len(trueSet)

def calculate_recall(recommendSet, trueSet):
    count = 0.0
    for item in recommendSet:
        if item in trueSet:
            count += 1
    return count / len(recommendSet)

def calculate_f_feature(precision, recall):
    return 2 * precision * recall / (precision + recall)


if __name__ == '__main__':
    filename = "test.json"
    trainFile = "train.txt"
    testFile = "test.txt"

    raw_data, userPurchasedSet, userTrueTestSet = preprocess(filename, trainFile, testFile)
    folds_files = [(trainFile, testFile)]
    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()
    bsl_options = {'method': 'sgd',
               'n_epochs': 20,
               'learning_rate': 0.005,
               }
    ### sim name: cosine    msd       pearson     pearson_baseline
    ### user_based : True ---- similarity will be computed based on users
    ###            : False ---- similarity will be computed based on items.
    sim_options = {'name': 'msd',
              'user_based':False}
    predictions = {}
    top_n = {}
    testsSet = None
    total_precisions = 0.0
    total_recalls = 0.0
    for trainset, testset in pkf.split(data):
        testsSet = testset
        
        algo = SVD(n_factors = 5)
        #algo = KNNBaseline(bsl_options=bsl_options, sim_options=sim_options)
        algo.fit(trainset)
        pre = algo.test(testset)
        accuracy.rmse(pre)
        
        #calculate_rmse(predictions)
        
        ### test
        rowNum = raw_data.get_row_size()
        colNum = raw_data.get_col_size()
        cur_time = time.time()
        time_cost = 0
        
        #print rowNum, colNum
        for i in range(rowNum):
            user = raw_data.get_userID(i)
            predictions[user] = set()
            pq = []
            heapq.heapify(pq)
            for j in range(colNum):
                item = raw_data.get_itemID(j)
                if user not in userPurchasedSet or item in userPurchasedSet[user]:
                    continue
                value = raw_data.get_val(user, item,'rating')
                predict = algo.predict(user, item, r_ui = 0, verbose=False)[3]
                if len(pq) >= 10:
                    heapq.heappop(pq)
                heapq.heappush(pq, (predict, item))
            top_n[user] = set()
            for items in pq:
                top_n[user].add(items[1])
            if user in userTrueTestSet:
                total_precisions += calculate_precision(top_n[user], userTrueTestSet[user])
                total_recalls += calculate_recall(top_n[user], userTrueTestSet[user])
            if i % 1000 == 0:
                duration = (time.time() - cur_time) / 60
                time_cost += duration
                remaining_time = ((rowNum - i)  / 1000) * duration
                cur_time = time.time()
                print 'precisions', total_precisions, ' recalls', total_recalls
                print 'i:', i, "/", rowNum, 'remaining time:', remaining_time, 'min'
    print 'precicions', total_precisions, ' recalls', total_recalls
    precisions, avgPrecision, recalls, avgRecalls = calculate_percision_and_recall(top_n, testsSet)
    print len(precisions)
    print 'avgPrecision', avgPrecision, 'avgRecall', avgRecalls  