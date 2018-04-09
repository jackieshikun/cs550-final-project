from sparse_data import sparse_data
import numpy as np

class HFT:
  __overall_mean = 0.0
  __user_bias = []
  __item_bias = []
  __rating_list = []
  __kappa = 1.0

  __n_word = 0
  __n_item = 0
  __n_user = 0
  __K = 5

  # (row,col), [topic number]
  __word_topics = {}
  # __n_word * __K
  __word_topic_cnt = None
  # __n_item * __K
  __item_topic_cnt = None
  # __n_word * __K
  __word_weight = None
  # __n_word
  __back_weight = None

  def __init__(self, data):
    self.__n_item = data.get_col_size()
    self.__n_user = data.get_row_size()
    self.__n_word = data.get_word_size()
    row_list = data.get_train_row_list()
    col_list = data.get_train_col_list()
    self.__word_topic_cnt = np.zeros((self.__n_word, self.__K), dtype=np.float64)
    self.__item_topic_cnt = np.zeros((self.__n_item, self.__K), dtype=np.float64)
    self.__word_weight = np.zeros((self.__n_word, self.__K), dtype=np.float64)
    self.__back_weight = np.zeros(self.__n_word, dtype=np.float64)
    # extract from data
    total_words = 0
    for idx in range(len(row_list)):
      r = row_list[idx]
      c = col_list[idx]
      rv = data.get_val(r, c, 'reviewText')
      rating = data.get_val(r, c, 'rating')
      self.__rating_list.append(rating)
      # random assignment topics
      review_len = len(rv)
      topic_list = np.random.randint(0, self.__K, review_len)
      self.__word_topics.setdefault((r,c), topic_list)
      for i in range(review_len):
        word = rv[i]
        topic = topic_list[i]
        self.__item_topic_cnt[c, topic] += 1
        self.__word_topic_cnt[word, topic] +=1
        total_words += 1
        self.__back_weight[word] += 1
    print(total_words)
    for i in range(self.__n_word):
      self.__back_weight[i] /= total_words
    self.__normalize_word_weight()

    # Initialize biases
    self.__overall_mean = np.sum(self.__rating_list) / len(self.__rating_list)
    # Because of using the row based storage
    # directly count and sum is okay.
    row_counts = np.bincount(row_list)
    row_sums = np.bincount(row_list, weights=self.__rating_list)
    self.__user_bias = row_sums / row_counts
    self.__user_bias = np.add(self.__user_bias, -self.__overall_mean)

    self.__item_bias = np.zeros(self.__n_item)
    col_sum = {}
    for idx in range(len(col_list)):
      col_idx = col_list[idx]
      if col_idx in col_sum:
        col_sum[col_idx][0] += self.__rating_list[idx]
        col_sum[col_idx][1] += 1
      else:
        col_sum.setdefault(col_idx, [self.__rating_list[idx], 1])
    for i in range(self.__n_item):
      rec = col_sum[i]
      self.__item_bias[i] = float(rec[0]) / rec[1]
    self.__item_bias = np.add(self.__item_bias, -self.__overall_mean)

  def __normalize_word_weight(self):
    avg_weight = np.divide(np.sum(self.__word_weight, axis=1), self.__K)
    self.__back_weight = np.add(self.__back_weight, avg_weight)
    self.__word_weight = np.add(self.__word_weight.T, -avg_weight).T

if __name__ == '__main__':
  data = sparse_data("test.json")
  print("start")
  x = HFT(data)
