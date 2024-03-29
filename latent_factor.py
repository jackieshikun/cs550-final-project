from sparse_data import sparse_data
from scipy import sparse
from sklearn.metrics import pairwise_distances
import numpy as np
import heapq
import operator
import math
from copy import deepcopy
from HFT import HFT

# Return: val_list, overall mean, user bias, item bias
def preprocess(train_data):
  row_list = train_data.get_train_row_list()
  col_list = train_data.get_train_col_list()
  val_list = []
  for ridx, cidx in zip(row_list, col_list):
    val = train_data.get_val(ridx, cidx, 'rating')
    if val == -1:
      print("[ERROR] getting rating from " + str(ridx) + str(cidx) + " error")
    val_list.append(val)
  overall_mean = np.sum(val_list) / len(val_list)
  # Because of using the row based storage
  # directly count and sum is okay.
  row_counts = np.bincount(row_list)
  row_sums = np.bincount(row_list, weights=val_list)
  # avoid divide by 0
  row_counts[row_counts == 0] = 1
  user_bias = row_sums / row_counts
  user_bias = np.add(user_bias, -overall_mean)

  item_size = train_data.get_col_size()
  item_bias = np.zeros(item_size)
  col_sum = {}
  for idx in range(len(col_list)):
    col_idx = col_list[idx]
    if col_idx in col_sum:
      col_sum[col_idx][0] += val_list[idx]
      col_sum[col_idx][1] += 1
    else:
      col_sum.setdefault(col_idx, [val_list[idx], 1])
  for i in range(item_size):
    if i in col_sum:
      rec = col_sum[i]
      item_bias[i] = float(rec[0]) / rec[1]
    else:
      item_bias[i] = 0.0

  item_bias = np.add(item_bias, -overall_mean)
  return val_list, overall_mean, user_bias, item_bias

#return mean, b_u, b_i, p, q
def fit(train_data, learning_rate_list, regulation_rate_list, epsilon=0.001, max_iter_num=20, factors=5):
  i = 0
  val_list, overall_mean, b_u, b_i = preprocess(train_data)
  row_list = train_data.get_train_row_list()
  col_list = train_data.get_train_col_list()
  user_size = train_data.get_row_size()
  item_size = train_data.get_col_size()

  # mean, std, (shape)
  p = np.random.normal(0, 0.1, (user_size, factors))
  q = np.random.normal(0, 0.1, (item_size, factors))

  b_u_lr = learning_rate_list[0]
  b_i_lr = learning_rate_list[1]
  p_lr = learning_rate_list[2]
  q_lr = learning_rate_list[3]
  b_u_rg = regulation_rate_list[0]
  b_i_rg = regulation_rate_list[1]
  p_rg = regulation_rate_list[2]
  q_rg = regulation_rate_list[3]

  total_err = 0.0
  pre_total_err = 0.0


  trow_list = train_data.get_test_row_list()
  tcol_list = train_data.get_test_col_list()
  vrow_list = train_data.get_validation_row_list()
  vcol_list = train_data.get_validation_col_list()
  for idx in range(len(trow_list)):
    r = trow_list[idx]
    c = tcol_list[idx]
    rating = train_data.get_val(r, c, 'rating')
    err = rating - (overall_mean + b_u[r] + b_i[c] + np.dot(p[r], q[c].T))
    total_err += (err / len(trow_list)* err)
  rmse = math.sqrt(total_err)
  mse = total_err
  print("initial guess test mse", mse)
  print("initial guess test rmse", rmse)
  total_err = 0.0

  for idx in range(len(row_list)):
    r = row_list[idx]
    c = col_list[idx]
    rating = train_data.get_val(r, c, 'rating')
    err = rating - (overall_mean + b_u[r] + b_i[c] + np.dot(p[r], q[c].T))
    total_err += (err / len(row_list) * err)
  rmse = math.sqrt(total_err)
  mse = total_err
  print("initial guess train mse", mse)
  print("initial guesstrain rmse", rmse)
  total_err = 0.0

  while i < max_iter_num:
    print("Processing iteration {}".format(i))
    orign_p = deepcopy(p)
    origin_q = deepcopy(q)
    for idx in range(len(val_list)):
      r = row_list[idx]
      c = col_list[idx]
      val = val_list[idx]
      pr = orign_p[r]
      qc = origin_q[c]
      err = val - (overall_mean + b_u[r] + b_i[c] + np.dot(pr, qc.T))
      total_err += (err / len(val_list) * err)
      b_u[r] += np.dot(b_u_lr, (err - np.dot(b_u_rg, b_u[r])))
      b_i[c] += np.dot(b_i_lr, (err - np.dot(b_i_rg, b_i[c])))
      p[r] += np.dot(p_lr, np.add(np.dot(err, qc), -np.dot(p_rg, pr)))
      q[c] += np.dot(q_lr, np.add(np.dot(err, pr), -np.dot(q_rg, qc)))
    # total_err = math.sqrt(total_err)
    print(total_err)
    if abs(total_err - pre_total_err) <= epsilon:
      break
    pre_total_err = total_err
    total_err = 0.0
    i += 1
    for idx in range(len(vrow_list)):
      r = vrow_list[idx]
      c = vcol_list[idx]
      rating = train_data.get_val(r, c, 'rating')
      err = rating - (overall_mean + b_u[r] + b_i[c] + np.dot(p[r], q[c].T))
      total_err += (err / len(vrow_list) * err)
    rmse = math.sqrt(total_err)
    mse = total_err
    print("latent factor validation mse", mse)
    print("latent factor validation rmse", rmse)
    total_err = 0.0
    for idx in range(len(trow_list)):
      r = trow_list[idx]
      c = tcol_list[idx]
      rating = train_data.get_val(r, c, 'rating')
      err = rating - (overall_mean + b_u[r] + b_i[c] + np.dot(p[r], q[c].T))
      total_err += (err / len(trow_list) * err)
    rmse = math.sqrt(total_err)
    mse = total_err
    print("latent factor test mse", mse)
    print("latent factor test rmse", rmse)
    total_err = 0.0
  total_err = 0.0
  for idx in range(len(row_list)):
    r = row_list[idx]
    c = col_list[idx]
    rating = train_data.get_val(r, c, 'rating')
    err = rating - (overall_mean + b_u[r] + b_i[c] + np.dot(p[r], q[c].T))
    total_err += (err / len(row_list) * err)
  rmse = math.sqrt(total_err)
  mse = total_err
  print("latent factor train mse", mse)
  print("latent factor train rmse", rmse)
  total_err = 0.0
  return overall_mean, b_u, b_i, p, q

def predict(data, mean, b_u, b_i, p, q, top_n=10):
  prediction = []
  prediction_col = []
  for i in range(data.get_row_size()):
    train_col = data.slice_train_row(i)
    pred = []
    pred_col = []
    b_u_i = b_u[i]
    p_i = p[i]
    for j in range(data.get_col_size()):
      if j in train_col:
        continue
      r = mean + b_u_i + b_i[j] + np.dot(p_i, q[j].T)
      pred.append(r)
      pred_col.append(j)
    top_idx = zip(*heapq.nlargest(top_n, enumerate(pred), key=operator.itemgetter(1)))[0]

    pred_sorted = []
    pred_col_sorted = []
    for idx in top_idx:
      pred_sorted.append(pred[idx])
      pred_col_sorted.append(pred_col[idx])
    print(pred_sorted)
    prediction.append(pred_sorted)
    prediction_col.append(pred_col_sorted)
  return prediction, prediction_col

if __name__ == '__main__':
  # data = sparse_data("Electronics_5.json")
  data = sparse_data("Video_Games_5.json")
  # data = sparse_data("test.json")
  # mean, b_u, b_i, p, q = fit(data, [0.005,0.005,0.005,0.005], [0.02,0.02,0.02,0.02], max_iter_num=30)
  HFT(data)
  #print(predict(data, mean, b_u, b_i, p, q))
  #print(data.get_row_size())
  #print(data.get_row_index("AO94DHGC771SJ"))
  #print(data.get_col_index("0528881469"))
  #print(data.get_val(0,0, 'rating'))
  #print(data.get_val(0,0, 'reviewText'))
  #print(data.get_entry_size())
  #print(pairwise_distances(rating_mat))
  #est_dimension(rating_mat)
