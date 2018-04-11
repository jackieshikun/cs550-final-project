from sparse_data import sparse_data
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import expit
import numpy as np
import time
from copy import deepcopy

class HFT:
  optimize_sample = True
  overall_mean = 0.0
  user_bias = []
  item_bias = []
  rating_list = []
  kappa = 1.0
  mu = 0.005
  gamma_user = None
  gamma_item = None
  data = None

  n_word = 0
  n_item = 0
  n_user = 0
  K = 5

  # (row,col), [topic number]
  # optimize {col, [topic, counts]}
  word_topics = {}
  # n_word * K
  word_topic_cnt = None
  # n_item * K
  item_topic_cnt = None
  # n_word * K
  word_weight = None
  # n_word
  back_weight = None

  def __init__(self, sparse_data, optimize_sample=True):
    self.n_item = data.get_col_size()
    self.n_user = data.get_row_size()
    self.n_word = data.get_word_size()
    self.data = sparse_data
    self.optimize_sample = optimize_sample
    row_list = self.data.get_train_row_list()
    col_list = self.data.get_train_col_list()
    self.word_topic_cnt = np.zeros((self.n_word, self.K), dtype=np.float64)
    self.item_topic_cnt = np.zeros((self.n_item, self.K), dtype=np.float64)
    self.word_weight = np.zeros((self.n_word, self.K), dtype=np.float64)
    self.back_weight = np.zeros(self.n_word, dtype=np.float64)
    self.gamma_user = np.random.normal(0, 0.1, (self.n_user, self.K))
    self.gamma_item = np.random.normal(0, 0.1, (self.n_item, self.K))
    # extract from data
    total_words = 0
    # Assign random topic to unique words
    if self.optimize_sample == True:
      for item in range(self.n_item):
        topic_list = np.random.randint(0, self.K, self.n_word)
        x_list = []
        for topic in topic_list:
          x_list.append([topic, 0])
        self.word_topics.setdefault(item, x_list)

    for idx in range(len(row_list)):
      r = row_list[idx]
      c = col_list[idx]
      rv = self.data.get_val(r, c, 'reviewText')
      rating = self.data.get_val(r, c, 'rating')
      self.rating_list.append(rating)
      # random assignment topics
      review_len = len(rv)
      if self.optimize_sample == True:
        for i in range(review_len):
          word = rv[i]
          topic = self.word_topics[c][word][0]
          self.word_topics[c][word][1] += 1
          self.item_topic_cnt[c, topic] += 1
          self.word_topic_cnt[word, topic] +=1
          self.back_weight[word] += 1

      else:
        topic_list = np.random.randint(0, self.K, review_len)
        self.word_topics.setdefault((r,c), topic_list)
        for i in range(review_len):
          word = rv[i]
          topic = topic_list[i]
          self.item_topic_cnt[c, topic] += 1
          self.word_topic_cnt[word, topic] +=1
          self.back_weight[word] += 1
      total_words += review_len
    print(total_words)
    for i in range(self.n_word):
      self.back_weight[i] /= total_words

    # Initialize biases
    self.overall_mean = np.sum(self.rating_list) / len(self.rating_list)
    # Because of using the row based storage
    # directly count and sum is okay.
    row_counts = np.bincount(row_list)
    row_sums = np.bincount(row_list, weights=self.rating_list)
    self.user_bias = row_sums / row_counts
    self.user_bias = np.add(self.user_bias, -self.overall_mean)

    self.item_bias = np.zeros(self.n_item)
    col_sum = {}
    for idx in range(len(col_list)):
      col_idx = col_list[idx]
      if col_idx in col_sum:
        col_sum[col_idx][0] += self.rating_list[idx]
        col_sum[col_idx][1] += 1
      else:
        col_sum.setdefault(col_idx, [self.rating_list[idx], 1])
    for i in range(self.n_item):
      rec = col_sum[i]
      self.item_bias[i] = float(rec[0]) / rec[1]
    self.item_bias = np.add(self.item_bias, -self.overall_mean)
    self.sample_topic()
    self.normalize_word_weight()
    self.fit()

  def normalize_word_weight(self):
    avg_weight = np.divide(np.sum(self.word_weight, axis=1), self.K)
    self.back_weight = np.add(self.back_weight, avg_weight)
    self.word_weight = np.add(self.word_weight.T, -avg_weight).T

  # Gibbs sampling
  # Performance imporvement for optimization
  # 100K entry
  # before sample optimization 187.9s
  # after sample optimization 3.7s
  def sample_topic(self):
    if self.optimize_sample == True:
      for col, topic_list in self.word_topics.items():
        random_list = np.random.sample(self.n_word)
        for i in range(self.n_word):
          old_topic = topic_list[i][0]
          old_cnt = topic_list[i][1]
          if old_cnt == 0:
            continue
          topic_score = expit(np.add(np.multiply(self.kappa, self.gamma_item[col]), np.add(self.back_weight[i], self.word_weight[i])))
          topic_sum = np.sum(topic_score)
          topic_score /= topic_sum
          indicator = random_list[i]
          for new_topic in range(self.K):
            indicator -= topic_score[new_topic]
            if indicator < 0:
              break
            new_topic += 1
          if new_topic != old_topic:
            self.word_topic_cnt[i][old_topic] -= old_cnt
            self.word_topic_cnt[i][new_topic] += old_cnt
            self.item_topic_cnt[col][old_topic] -= old_cnt
            self.item_topic_cnt[col][new_topic] += old_cnt
            topic_list[i][0] = new_topic
    else:
      # scores only calculate once
      topic_score_c_w = np.zeros((self.n_item, self.n_word, self.K))
      for col in range(self.n_item):
        for word in range(self.n_word):
          topic_score = expit(np.add(np.multiply(self.kappa, self.gamma_item[col]), np.add(self.back_weight[word], self.word_weight[word])))
          topic_sum = np.sum(topic_score)
          topic_score_c_w[col, word] = topic_score / topic_sum
      for rec, topic_list in self.word_topics.items():
        row = rec[0]
        col = rec[1]
        rv = self.data.get_val(row, col, 'reviewText')
        rv_len = len(topic_list)
        random_list = np.random.sample(rv_len)
        for i in range(rv_len):
          word = rv[i]
          indicator = random_list[i]
          topic_score = topic_score_c_w[col, word]
          for new_topic in range(self.K):
            indicator -= topic_score[new_topic]
            if indicator < 0:
              break
            new_topic += 1
          old_topic = topic_list[i]
          if new_topic != old_topic:
            self.word_topic_cnt[word][old_topic] -= 1
            self.word_topic_cnt[word][new_topic] += 1
            self.item_topic_cnt[col][old_topic] -= 1
            self.item_topic_cnt[col][new_topic] += 1
            topic_list[i] = new_topic

  # overall_mean, kappa, bias_user, bias_item, gamma_user, gamma_item, word_weight
  def get_args(self, arglist):
    idx = 0
    roverall_mean = arglist[idx]
    idx += 1
    rkappa = arglist[idx]
    idx += 1
    rbias_user = arglist[idx:idx+self.n_user]
    idx += self.n_user
    rbias_itme = arglist[idx: idx+self.n_item]
    idx += self.n_item
    rgamma_user = np.zeros((self.n_user, self.K), dtype=np.float64)
    for i in range(self.n_user):
      rgamma_user[i] = arglist[idx: idx + self.K]
      idx += self.K
    rgamma_item = np.zeros((self.n_item, self.K), dtype=np.float64)
    for i in range(self.n_item):
      rgamma_item[i] = arglist[idx: idx + self.K]
      idx += self.K
    rword_weight = np.zeros((self.n_word, self.K), dtype=np.float64)
    for i in range(self.n_word):
      rword_weight[i] = arglist[idx: idx + self.K]
      idx += self.K
    return roverall_mean, rkappa, rbias_user, rbias_itme, rgamma_user, rgamma_item, rword_weight

  def set_args(self, ioverall_mean, ikappa, ibias_user, ibias_item, igamma_user, igamma_item, iword_weight):
    arglist = []
    arglist.append(ioverall_mean)
    arglist.append(ikappa)
    arglist += list(ibias_user)
    arglist += list(ibias_item)
    for i in range(self.n_user):
      arglist += list(igamma_user[i])
    for i in range(self.n_item):
      arglist += list(igamma_item[i])
    for i in range(self.n_word):
      arglist += list(iword_weight[i])
    return arglist

  def set_args_back(self, arglist):
    idx = 0
    self.overall_mean = arglist[idx]
    idx += 1
    self.kappa = arglist[idx]
    idx += 1
    self.user_bias = arglist[idx:idx+self.n_user]
    idx += self.n_user
    self.item_bias = arglist[idx: idx+self.n_item]
    idx += self.n_item
    for i in range(self.n_user):
      self.gamma_user[i] = arglist[idx: idx + self.K]
      idx += self.K
    for i in range(self.n_item):
      self.gamma_item[i] = arglist[idx: idx + self.K]
      idx += self.K
    for i in range(self.n_word):
      self.word_weight[i] = arglist[idx: idx + self.K]
      idx += self.K

  def update_word_weight_para(self, word_weight_new, arglist):
    idx = len(arglist)
    i = self.n_word - 1
    while i >= 0:
      arglist[idx -self.K:idx] = word_weight_new[i]
      idx -= self.K
      i -= 1

  def fit(self):
    i = 0
    while i < 20:
      args = self.set_args(self.overall_mean, self.kappa, self.user_bias, self.item_bias, self.gamma_user, self.gamma_item, self.word_weight)
      parameters = [self]
      t = time.time()
      res, val, d = fmin_l_bfgs_b(evaluation, np.array(args), fprime=derivation, args=parameters)
      self.set_args_back(res)
      print(time.time() - t)
      print(val, d)
      self.sample_topic()
      self.normalize_word_weight()
      i +=1

# we don't need to update the arguments only calculate the derivation
def evaluation(args, *parameters):
  # parse
  obj = parameters[0]
  overall_mean, kappa, bias_user, bias_item, gamma_user, gamma_item, word_weight = obj.get_args(args)

  data = obj.data
  mu = obj.mu
  item_topic_cnt = obj.item_topic_cnt
  word_topic_cnt = obj.word_topic_cnt
  back_weight = obj.back_weight
  row_list = data.get_train_row_list()
  col_list = data.get_train_col_list()
  result = 0.0
  # compute error squre
  for i in range(len(row_list)):
    r = row_list[i]
    c = col_list[i]
    rating = data.get_val(r, c, 'rating')
    err = rating - (overall_mean + bias_user[r] + bias_item[c] + np.dot(gamma_user[r], gamma_item[c].T))
    result += (err * err)
  # compute theta
  # incase overflow, multiply mu at each stage
  lz = np.log(np.sum(expit(np.dot(gamma_item, kappa)), axis=1))
  result -= np.dot(mu, np.sum(np.multiply(item_topic_cnt, np.add(np.dot(kappa, gamma_item).T, -lz).T)))
  # compute phi
  sum_weight = np.add(back_weight, word_weight.T)
  lw = np.log(np.sum(expit(sum_weight), axis=1))
  result -= np.dot(mu, np.sum(np.multiply(word_topic_cnt, np.add(sum_weight.T, -lw))))
  print(kappa)
  return result

def derivation(args, *parameters):
  obj = parameters[0]
  overall_mean, kappa, bias_user, bias_item, gamma_user, gamma_item, word_weight = obj.get_args(args)
  data = obj.data
  mu = obj.mu
  item_topic_cnt = obj.item_topic_cnt
  word_topic_cnt = obj.word_topic_cnt
  back_weight = obj.back_weight
  row_list = data.get_train_row_list()
  col_list = data.get_train_col_list()
  doverall_mean = 0.0
  dbias_user = np.zeros(obj.n_user)
  dbias_item = np.zeros(obj.n_item)
  dgamma_user = np.zeros((obj.n_user, obj.K), dtype=np.float64)
  dgamma_item = np.zeros((obj.n_item, obj.K), dtype=np.float64)
  dword_weight = np.zeros((obj.n_word, obj.K), dtype=np.float64)
  for i in range(len(row_list)):
    r = row_list[i]
    c = col_list[i]
    rating = data.get_val(r, c, 'rating')
    err = rating - (overall_mean + bias_user[r] + bias_item[c] + np.dot(gamma_user[r], gamma_item[c].T))
    err *= 2
    # derivative
    doverall_mean += err
    dbias_user[r] += err
    dbias_item[c] += err
    dgamma_user[r] = np.add(np.dot(gamma_item[c], err), dgamma_user[r])
    dgamma_item[c] = np.add(np.dot(gamma_user[r], err), dgamma_item[c])
  expz = expit(np.dot(gamma_item, kappa))
  sz = np.sum(expz, axis=1)
  # n_item * K
  tmp = np.dot(-mu, np.add(item_topic_cnt, -np.divide(expz.T, sz).T))
  dgamma_item = np.add(dgamma_item, np.dot(kappa, tmp))
  dkappa = np.sum(np.multiply(gamma_item, tmp))

  sum_weight_exp = expit(np.add(back_weight, word_weight.T))
  sw = np.sum(sum_weight_exp, axis=1)
  dword_weight = np.dot(-mu, np.add(word_topic_cnt, -np.divide(sum_weight_exp.T, sw)))

  rargs = obj.set_args(doverall_mean, dkappa, dbias_user, dbias_item, dgamma_user, dgamma_item, dword_weight)
  return np.array(rargs)

if __name__ == '__main__':
  data = sparse_data("test.json")
  x = HFT(data)
