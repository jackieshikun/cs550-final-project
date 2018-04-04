import json
from review_enum import review_enum
import random
import numpy as np
class sparse_data:
  __row2user_dict = {}
  __user2row_dict = {}
  __col2item_dict = {}
  __item2col_dict = {}
  #(row,col), [rate, [helpful, unhelpful], review_text]
  __data = {}
  __row_train_ind = []
  __col_train_ind = []
  __row_test_ind = []
  __col_test_ind = []
  def __init__(self, filename):
    with open(filename, "r") as f:
      lines = f.readlines()
    row_count = 0
    col_count = 0
    row_ind = []
    col_ind = []
    for line in lines:
      record = json.loads(line)
      userId = record['reviewerID']
      itemId = record['asin']
      if userId not in self.__user2row_dict:
        self.__user2row_dict[userId] = row_count
        self.__row2user_dict[row_count] = userId
        row_count += 1
      if itemId not in self.__item2col_dict:
        self.__item2col_dict[itemId] = col_count
        self.__col2item_dict[col_count] = itemId
        col_count += 1
      row_idx_tmp = self.__user2row_dict[userId]
      col_idx_tmp = self.__item2col_dict[itemId]
      self.__data[(row_idx_tmp, col_idx_tmp)] = [float(record['overall']), record['helpful'], record['reviewText']]
      row_ind.append(row_idx_tmp)
      col_ind.append(col_idx_tmp)
    sorted_idx = np.argsort(row_ind)
    row_ind = np.array(row_ind)
    col_ind = np.array(col_ind)
    row_ind = row_ind[sorted_idx]
    col_ind = col_ind[sorted_idx]
    self.__seperate_train_test(row_ind, col_ind)

  # default hold 20% of items purchesed for each user as test data.
  def __seperate_train_test(self, row_idx, col_idx, prob=0.2):
    ptr = 0
    i = 0
    size = len(row_idx)
    while i < size:
      while ptr < size and row_idx[i] == row_idx[ptr]:
        ptr += 1
      count = ptr - i
      if not count == 0:
        rand_size = int(count * prob)
        rand_idx = random.sample(range(0, count), rand_size)
        if not len(rand_idx) == 0:
          self.__row_test_ind += list(row_idx[np.add(rand_idx, i)])
          self.__col_test_ind += list(col_idx[np.add(rand_idx, i)])
        train_idx = [item for item in range(count) if item not in rand_idx]
        if not len(train_idx) == 0:
          self.__row_train_ind += list(row_idx[np.add(train_idx, i)])
          self.__col_train_ind += list(col_idx[np.add(train_idx, i)])
      i = ptr

  def get_row_size(self):
    return len(self.__row2user_dict)

  def get_col_size(self):
    return len(self.__col2item_dict)

  def get_entry_size(self):
    return len(self.__data)

  # -1 means no such record
  def get_val(self, row_id, col_id, attr):
    if (row_id, col_id) in self.__data and attr in review_enum:
      return self.__data[(row_id, col_id)][review_enum[attr]]
    return -1

  def set_val(self, row_id, col_id, value, attr):
    if (row_id, col_id) in self.__data and attr in review_enum:
      self.__data[(row_id, col_id)][review_enum[attr]] = value
      return 0
    return -1

  # -1 means out of bound
  def get_itemID(self, col_id):
    if col_id >= self.get_col_size():
      return str(-1)
    return self.__col2item_dict[col_id]

  def get_userID(self, row_id):
    if row_id >= self.get_row_size():
      return str(-1)
    return self.__row2user_dict[row_id]

  # -1 means out of bound
  def get_col_index(self, itemId):
    if itemId not in self.__item2col_dict:
      return -1
    return self.__item2col_dict[itemId]

  def get_row_index(self, userId):
    if userId not in self.__user2row_dict:
      return -1
    return self.__user2row_dict[userId]

  def get_train_row_list(self):
    return self.__row_train_ind

  def get_train_col_list(self):
    return self.__col_train_ind

  def get_test_row_list(self):
    return self.__row_test_ind

  def get_test_col_list(self):
    return self.__col_test_ind
