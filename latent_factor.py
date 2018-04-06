from sparse_data import sparse_data
from scipy import sparse
from sklearn.metrics import pairwise_distances
import numpy as np

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
  # Because of using the row based storage
  # directly count and sum is okay.
  row_counts = np.bincount(row_list)
  row_sums = np.bincount(row_list, weights=val_list)
  user_bias = row_sums / row_counts
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
    rec = col_sum[i]
    item_bias[i] = float(rec[0]) / rec[1]
  overall_mean = np.sum(val_list) / len(val_list)
  #rating_mat = sparse.csr_matrix((val_list, (row_list, col_list)), dtype=float)
  return val_list, overall_mean, user_bias, item_bias

# Based on the sigma of SVD to decide how many factors to keep in latent factor
def est_dimension(sparse_matrix, energy=0.8):
  U, sigma, VT = sparse.linalg.svds(sparse_matrix)
  sig2 = sigma ** 2
  threshold = sig2.sum() * energy
  dimension = sigma.shape[0]
  while dimension > 0:
    dimension -= 1
    if sig2[:dimension].sum() <= threshold:
      break
  return dimension

#return b_u, b_i, p, q
def fit(train_data, learning_rate_list, regulation_rate_list, error, iter_num):
  i = 0
  rating_csr, overall_mean, user_bias, item_bias = preprocess(train_data)
  factors = est_dimension(csr)
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
  p_rg = regulation_rate_list[3]
  q_rg = regulation_rate_list[4]

  row_list = train_data.get_train_row_list()
  col_list = train_data.get_train_col_list()
  while i < iter_num:
    print("Processing iteration {}".format(i))
    total_err = 0
    for idx in range(len(val_list)):
      r = row_list[idx]
      c = col_list[idx]
      r = val_list[idx]
      pr = r[r]
      qc = q[c]
      err = r - (overall_mean + b_u[r] + b_i[c] + np.dot(pr, qc.T)
      total_err += err ** 2
      b_u[r] += b_u_lr * (err - b_u_rg * b_u[r])
      b_i[c] += b_i_lr * (err - b_i_rg * b_i[c])
      p[r] += p_lr * (err * qc + p_rg * pr)
      q[c] += q_lr * (err * pr + q_rt * qc)
    if total_err <= error:
      break
    i += 1

if __name__ == '__main__':
  data = sparse_data("test.json")
  print(len(data.get_train_row_list()))
  print(len(data.get_train_col_list()))
  print(len(data.get_test_row_list()))
  print(len(data.get_test_col_list()))
  print(data.get_entry_size())
  #print(data.get_row_size())
  #print(data.get_row_index("AO94DHGC771SJ"))
  #print(data.get_col_index("0528881469"))
  #print(data.get_val(0,0, 'rating'))
  #print(data.get_val(0,0, 'reviewText'))
  #print(data.get_entry_size())
  rating_mat, overall_mean, user_bias, item_bias = preprocess(data)
  #print(pairwise_distances(rating_mat))
  #est_dimension(rating_mat)
