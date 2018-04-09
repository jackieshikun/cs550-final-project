from sparse_data import sparse_data
import numpy as np
import time

class HFT:
  __optimize_sample = True
  __overall_mean = 0.0
  __user_bias = []
  __item_bias = []
  __rating_list = []
  __kappa = 1.0
  __gamma_user = None
  __gamma_item = None
  __data = None

  __n_word = 0
  __n_item = 0
  __n_user = 0
  __K = 5

  # (row,col), [topic number]
  # optimize {col, [topic, counts]}
  __word_topics = {}
  # __n_word * __K
  __word_topic_cnt = None
  # __n_item * __K
  __item_topic_cnt = None
  # __n_word * __K
  __word_weight = None
  # __n_word
  __back_weight = None

  def __init__(self, data, optimize_sample=True):
    self.__n_item = data.get_col_size()
    self.__n_user = data.get_row_size()
    self.__n_word = data.get_word_size()
    self.__data = data
    self.__optimize_sample = optimize_sample
    row_list = self.__data.get_train_row_list()
    col_list = self.__data.get_train_col_list()
    self.__word_topic_cnt = np.zeros((self.__n_word, self.__K), dtype=np.float64)
    self.__item_topic_cnt = np.zeros((self.__n_item, self.__K), dtype=np.float64)
    self.__word_weight = np.zeros((self.__n_word, self.__K), dtype=np.float64)
    self.__back_weight = np.zeros(self.__n_word, dtype=np.float64)
    self.__gamma_user = np.random.normal(0, 0.1, (self.__n_user, self.__K))
    self.__gamma_item = np.random.normal(0, 0.1, (self.__n_item, self.__K))
    # extract from data
    total_words = 0
    # Assign random topic to unique words
    if self.__optimize_sample == True:
      for item in range(self.__n_item):
        topic_list = np.random.randint(0, self.__K, self.__n_word)
        x_list = []
        for topic in topic_list:
          x_list.append([topic, 0])
        self.__word_topics.setdefault(item, x_list)

    for idx in range(len(row_list)):
      r = row_list[idx]
      c = col_list[idx]
      rv = self.__data.get_val(r, c, 'reviewText')
      rating = self.__data.get_val(r, c, 'rating')
      self.__rating_list.append(rating)
      # random assignment topics
      review_len = len(rv)
      if self.__optimize_sample == True:
        for i in range(review_len):
          word = rv[i]
          topic = self.__word_topics[c][word][0]
          self.__word_topics[c][word][1] += 1
          self.__item_topic_cnt[c, topic] += 1
          self.__word_topic_cnt[word, topic] +=1
          self.__back_weight[word] += 1

      else:
        topic_list = np.random.randint(0, self.__K, review_len)
        self.__word_topics.setdefault((r,c), topic_list)
        for i in range(review_len):
          word = rv[i]
          topic = topic_list[i]
          self.__item_topic_cnt[c, topic] += 1
          self.__word_topic_cnt[word, topic] +=1
          self.__back_weight[word] += 1
      total_words += review_len
    print(total_words)
    for i in range(self.__n_word):
      self.__back_weight[i] /= total_words
    self.__normalize_word_weight()

    t0 = time.time()
    self.__sample_topic()
    print(time.time() - t0)

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

  # Gibbs sampling
  # Performance imporvement for optimization
  # 100K entry
  # before sample optimization 187.9s
  # after sample optimization 3.7s
  def __sample_topic(self):
    if self.__optimize_sample == True:
      for col, topic_list in self.__word_topics.items():
        random_list = np.random.sample(self.__n_word)
        for i in range(self.__n_word):
          old_topic = topic_list[i][0]
          old_cnt = topic_list[i][1]
          if old_cnt == 0:
            continue
          topic_score = np.exp(np.add(np.multiply(self.__kappa, self.__gamma_item[col]), np.add(self.__back_weight[i], self.__word_weight[i])))
          topic_sum = np.sum(topic_score)
          topic_score /= topic_sum
          indicator = random_list[i]
          for new_topic in range(self.__K):
            indicator -= topic_score[new_topic]
            if indicator < 0:
              break
            new_topic += 1
          if new_topic != old_topic:
            self.__word_topic_cnt[i][old_topic] -= old_cnt
            self.__word_topic_cnt[i][new_topic] += old_cnt
            self.__item_topic_cnt[col][old_topic] -= old_cnt
            self.__item_topic_cnt[col][new_topic] += old_cnt
            topic_list[i][0] = new_topic
    else:
      # scores only calculate once
      topic_score_c_w = np.zeros((self.__n_item, self.__n_word, self.__K))
      for col in range(self.__n_item):
        for word in range(self.__n_word):
          topic_score = np.exp(np.add(np.multiply(self.__kappa, self.__gamma_item[col]), np.add(self.__back_weight[word], self.__word_weight[word])))
          topic_sum = np.sum(topic_score)
          topic_score_c_w[col, word] = topic_score / topic_sum
      for rec, topic_list in self.__word_topics.items():
        row = rec[0]
        col = rec[1]
        rv = self.__data.get_val(row, col, 'reviewText')
        rv_len = len(topic_list)
        random_list = np.random.sample(rv_len)
        for i in range(rv_len):
          word = rv[i]
          indicator = random_list[i]
          topic_score = topic_score_c_w[col, word]
          for new_topic in range(self.__K):
            indicator -= topic_score[new_topic]
            if indicator < 0:
              break
            new_topic += 1
          old_topic = topic_list[i]
          if new_topic != old_topic:
            self.__word_topic_cnt[word][old_topic] -= 1
            self.__word_topic_cnt[word][new_topic] += 1
            self.__item_topic_cnt[col][old_topic] -= 1
            self.__item_topic_cnt[col][new_topic] += 1
            topic_list[i] = new_topic

if __name__ == '__main__':
  data = sparse_data("test.json")
  print("start")
  print(data.get_col_size())
  print(data.get_row_size())
  x = HFT(data)
