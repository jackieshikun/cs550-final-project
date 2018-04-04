from sparse_data import sparse_data
from scipy import sparse
from sklearn.metrics import pairwise_distances

def gen_rating_mat(file_data):
    row_list = file_data.get_row_list();
    col_list = file_data.get_col_list();
    val_list = []
    for ridx, cidx in zip(row_list, col_list):
        val_list.append(file_data.get_val(ridx, cidx, 'rating'))
    rating_mat = sparse.csr_matrix((val_list, (row_list, col_list)), dtype=float)
    return rating_mat

def est_dimension(sparse_matrix, energy=0.8):
    U, sigma, VT = sparse.linalg.svds(sparse_matrix)
    sig2 = sigma ** 2
    threshold = sig2.sum() * energy
    dimension = sigma.shape[0]
    while dimension > 0:
        dimension -= 1
        if sig2[:dimension].sum() <= threshold:
            break;
    return dimension

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
    #rating_mat = gen_rating_mat(data)
    #print(pairwise_distances(rating_mat))
    #est_dimension(rating_mat)
