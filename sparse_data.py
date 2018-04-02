import json
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
            rating = float(record['overall'])
            helpful, unhelpful = record['helpful']
            review_text = record['reviewText']
            if userId not in self.__user2row_dict:
                self.__user2row_dict[userId] = row_count
                self.__row2user_dict[row_count] = userId
                row_count += 1
            if itemId not in self.__item2col_dict:
                self.__item2col_dict[itemId] = col_count
                self.__col2item_dict[col_count] = itemId
                col_count += 1
            self.__data[(self.__user2row_dict[userId], self.__item2col_dict[itemId])] = [rating, [helpful, unhelpful], review_text]

    def get_row_size(self):
        return len(self.__row2user_dict)

    def get_col_size(self):
        return len(self.__col2item_dict)

    # -1 means no such record
    def get_rating(self, row_id, col_id):
        if (row_id, col_id) in self.__data:
            return self.__data[(row_id, col_id)][0]
        return -1

    def set_rating(self, row_id, col_id, rating):
        if (row_id, col_id) in self.__data:
            self.__data[(row_id, col_id)][0] = rating
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
