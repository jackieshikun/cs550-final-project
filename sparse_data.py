import json
from review_enum import review_enum
class sparse_data:
    __row2user_dict = {}
    __user2row_dict = {}
    __col2item_dict = {}
    __item2col_dict = {}
    #(row,col), [rate, [helpful, unhelpful], review_text]
    __data = {}
    def __init__(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
        row_count = 0
        col_count = 0
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
            self.__data[(self.__user2row_dict[userId], self.__item2col_dict[itemId])] = [float(record['overall']), record['helpful'], record['reviewText']]

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
    def get_col(self, itemId):
        if itemId not in self.__item2col_dict:
            return -1
        return self.__item2col_dict[itemId]

    def get_row(self, userId):
        if userId not in self.__user2row_dict:
            return -1
        return self.__user2row_dict[userId]
