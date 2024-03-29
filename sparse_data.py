import json
from review_enum import review_enum
import random
import numpy as np
import bisect
from sklearn.feature_extraction import stop_words
import re

class sparse_data:
  __row2user_dict = {}
  __user2row_dict = {}
  __col2item_dict = {}
  __item2col_dict = {}
  #(row,col), [rate, [helpful, unhelpful], review_text(list)]
  __data = {}
  __row_train_ind = []
  __col_train_ind = []
  __row_test_ind = []
  __col_test_ind = []
  __row_validation_ind = []
  __col_validation_ind = []

  # Review Text Extraction
  __word2id = {}
  __id2word = {}
  def __init__(self, filename):
    with open(filename, "r") as f:
      lines = f.readlines()
    row_count = 0
    col_count = 0
    row_ind = []
    col_ind = []
    n_word = 0
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
      # parse the review text to save memory
      rv = record['reviewText']
      review_words = []
      filter_str = '(.+(s|ing|ed)$|.+[0-9].+)'
      re_filter = re.compile(filter_str)
      for word in re.split(r"\W+", rv):
        word = word.lower()
        if word in self.__word2id:
          review_words.append(self.__word2id[word])
          continue
        if len(word) <= 2:
          continue
        if re_filter.search(word):
          continue
        if word in stop_words.ENGLISH_STOP_WORDS:
          continue
        if word not in self.__word2id:
          self.__word2id[word] = n_word
          self.__id2word[n_word] = word
          n_word += 1
        review_words.append(self.__word2id[word])
      self.__data[(row_idx_tmp, col_idx_tmp)] = [float(record['overall']), record['helpful'], review_words]
      row_ind.append(row_idx_tmp)
      col_ind.append(col_idx_tmp)
    print(n_word)
    sorted_idx = np.argsort(row_ind)
    row_ind = np.array(row_ind)
    col_ind = np.array(col_ind)
    row_ind = row_ind[sorted_idx]
    col_ind = col_ind[sorted_idx]
    row_train_ind, col_train_ind = self.__separate_train_test(row_ind, col_ind)
    self.__separate_train_validation(row_train_ind, col_train_ind, prob=0.0)

  # default hold 10% of total train set as validation data.
  def __separate_train_validation(self, row_idx, col_idx, prob=0.1):
    entry_size = len(row_idx)
    rand_idx = sorted(random.sample(xrange(0, entry_size), int(entry_size * prob)))
    rand_ptr = 0
    for i in range(entry_size):
      r = row_idx[i]
      c = col_idx[i]
      if rand_ptr < len(rand_idx) and rand_idx[rand_ptr] == i:
        rand_ptr += 1
        self.__row_validation_ind.append(r)
        self.__col_validation_ind.append(c)
      else:
        self.__row_train_ind.append(r)
        self.__col_train_ind.append(c)
    print("validation size", len(self.__row_validation_ind))
    print("train size", len(self.__row_train_ind))

  # default hold 20% of items purchesed for each user as test data.
  def __separate_train_test(self, row_idx, col_idx, prob=0.2):
    ptr = 0
    i = 0
    size = len(row_idx)
    row_train_ind = []
    col_train_ind = []
    while i < size:
      while ptr < size and row_idx[i] == row_idx[ptr]:
        ptr += 1
      count = ptr - i
      if not count == 0:
        rand_size = int(count * prob)
        rand_idx = random.sample(xrange(0, count), rand_size)
        if not len(rand_idx) == 0:
          self.__row_test_ind += list(row_idx[np.add(rand_idx, i)])
          self.__col_test_ind += list(col_idx[np.add(rand_idx, i)])
        train_idx = [item for item in range(count) if item not in rand_idx]
        if not len(train_idx) == 0:
          row_train_ind += list(row_idx[np.add(train_idx, i)])
          col_train_ind += list(col_idx[np.add(train_idx, i)])
      i = ptr
    return [row_train_ind, col_train_ind]

  def get_word_size(self):
    return len(self.__word2id)

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

  def get_word_num(self, word):
    if word not in self.__word2id:
      return -1
    return self.__word2id[word]

  def get_word(self, word_no):
    if word_no not in self.__id2word:
      return -1
    return self.__id2word[word_no]

  def get_train_row_list(self):
    return self.__row_train_ind

  def get_train_col_list(self):
    return self.__col_train_ind

  def get_test_row_list(self):
    return self.__row_test_ind

  def get_test_col_list(self):
    return self.__col_test_ind

  def get_validation_row_list(self):
    return self.__row_validation_ind

  def get_validation_col_list(self):
    return self.__col_validation_ind

  def __bisearch_left(self, row_list, val):
    i = bisect.bisect_left(row_list, val)
    if i != len(row_list) and row_list[i] == val:
      return i
    return -1

  # return col_idx of searched row
  def slice_test_row(self, rowId):
    start = self.__bisearch_left(self.__row_test_ind, rowId)
    if start == -1:
      return []
    i = start
    while i < len(self.__row_test_ind) and self.__row_test_ind[i] == rowId:
      i += 1
    return self.__col_test_ind[start: i]

  def slice_validation_row(self, rowId):
    start = self.__bisearch_left(self.__row_validation_ind, rowId)
    if start == -1:
      return []
    i = start
    while i < len(self.__row_validation_ind) and self.__row_validation_ind[i] == rowId:
      i += 1
    return self.__col_validation_ind[start: i]

  # return col_idx of searched row
  def slice_train_row(self, rowId):
    start = self.__bisearch_left(self.__row_train_ind, rowId)
    if start == -1:
      return []
    i = start
    while i < len(self.__row_train_ind) and self.__row_train_ind[i] == rowId:
      i += 1
    return self.__col_train_ind[start: i]
