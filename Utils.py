import numpy as np


class Data:
    def __init__(self, X):
        self.batch_start_index = 0
        self.X = X
        self.data_size = len(X)
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        end_index = self.batch_start_index + batch_size
        if end_index > self.data_size:
            # Shuffle the data
            perm = np.arange(self.data_size)
            np.random.shuffle(perm)
            self.X = self.X[perm]

            self.epochs_completed += 1
            self.batch_start_index = 0

        start_index = self.batch_start_index
        batch_X = self.X[start_index: end_index]

        return batch_X


def get_sorted_index(score, order='descending'):
    '''
    :param score:
    :return: index of sorted item in descending order
    e.g. [8,3,4,9] return [3,0,2,1]
    '''
    score_map = []
    size = len(score)
    for i in range(size):
        score_map.append({'index':i, 'score':score[i]})
    if order == "descending":
        reverse = True
    elif order == "ascending":
        reverse = False
    score_map.sort(key=lambda x: x['score'], reverse=reverse)
    keys = [x['index'] for x in score_map]
    return keys


# @nb.njit()
def get_rank(score):
    '''
    :param score:
    :return:
    e.g. input: [0.8, 0.4, 0.6] return [0, 2, 1]
    '''
    sort = np.argsort(score)
    size = score.shape[0]
    rank = np.zeros(size)
    for i in range(size):
        rank[sort[i]] = size - i - 1

    return rank

