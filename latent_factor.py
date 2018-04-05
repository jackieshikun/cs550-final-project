from sparse_data import sparse_data
from scipy import sparse
from sklearn.metrics import pairwise_distances
import numpy as np

# Return: rating_matrix, overall mean, user bias, item bias
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
  rating_mat = sparse.csr_matrix((val_list, (row_list, col_list)), dtype=float)
  return rating_mat, overall_mean, user_bias, item_bias

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

#return p,q,b_i,b_u
def fit(train_data, learning_rate_list, error, iter_num):
  i = 0
  csr = gen_rating_mat(train_data)
  factors = est_dimension(csr)
  users = train_data.get_row_size()
  items = train_data.get_col_size()

  p = np.zeros(users, factors)
  q = np.zeros(items, factors)
  b_u = np.zeros(users, np.double)
  b_i = np.zeros(items, np.double)
  while i < iter_num:
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
  rating_mat, overall_mean, user_bias, item_bias = gen_rating_mat(data)
  #print(pairwise_distances(rating_mat))
  #est_dimension(rating_mat)
