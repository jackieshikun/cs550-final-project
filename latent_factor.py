from sparse_data import sparse_data

if __name__ == '__main__':
    data = sparse_data("test.json")
    print(data.get_row_size())
    print(data.get_row("AO94DHGC771SJ"))
    print(data.get_col("0528881469"))
    print(data.get_val(0,0, 'rating'))
    print(data.get_val(0,0, 'reviewText'))
    print(data.get_entry_size())
