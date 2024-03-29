import pandas as pd
from collections import Counter
import numpy as np


class DataLoader:

    # import data, get labels and count number of features, values, and objects
    def __init__(self, data_path, verbose=False):
        input_split = data_path.split("/")
        self.data_name = input_split[len(input_split) - 1].split(".")[0]

        data = pd.read_csv(data_path)
        # data = pd.read_csv(data_path)
        self.data_matrix = data.values
        [self.objects_num, self.all_features_num] = data.shape
        self.all_features_num = self.all_features_num - 1

        self.head = data.columns

        self.cate_name = [f_name for f_name in self.head if f_name.startswith("A")]
        self.nume_name = [f_name for f_name in self.head if f_name.startswith("B")]

        self.cate_features_num = len([f_name for f_name in self.head if f_name.startswith("A")])
        self.nume_features_num = len([f_name for f_name in self.head if f_name.startswith("B")])

        self.values_num_list = np.array([len(np.unique(self.data_matrix[:, i])) for i in range(self.cate_features_num)])
        self.values_num = np.sum(self.values_num_list)
        self.list_of_class = np.array(self.data_matrix[:, self.all_features_num], dtype='int')

        self.cate_data = np.zeros([self.objects_num, self.cate_features_num], dtype=int)
        self.nume_data = np.zeros([self.objects_num, self.nume_features_num])

        self.value_frequency_list = np.zeros(self.values_num)
        self.value_list = []
        self.first_value_index = []
        self.first_value_index.append(0)
        self.value_feature_indicator = []

        self.verbose = verbose
        return

    def data_dorp(self, dorp_list):
        cate_fs_num = self.cate_features_num
        nume_fs_num = self.nume_features_num
        new_dorp_list = []
        cate_list = []
        nume_list = []

        for id in dorp_list:
            if id + 1 <= cate_fs_num:
                if self.cate_features_num > 1:
                    self.cate_features_num -= 1
                    cate_list.append(id)

            else:
                if self.nume_features_num > 1:
                    self.nume_features_num -= 1
                    nume_list.append(id)

        self.cate_name = [self.cate_name[i] for i in range(len(self.cate_name)) if (i not in cate_list)]
        self.nume_name = [self.nume_name[i] for i in range(len(self.nume_name)) if
                          (i not in (np.array(nume_list) - cate_fs_num))]

        new_dorp_list = cate_list + nume_list

        # print(new_dorp_list)

        self.all_features_num = self.all_features_num - len(new_dorp_list)
        self.data_matrix = np.delete(self.data_matrix, new_dorp_list, axis=1)
        self.cate_data = np.delete(self.cate_data, cate_list, axis=1)
        self.nume_data = np.delete(self.nume_data, nume_list, axis=1)
        self.values_num_list = np.array([len(np.unique(self.data_matrix[:, i])) for i in range(self.cate_features_num)])
        self.values_num = np.sum(self.values_num_list)

        self.cate_data = np.zeros([self.objects_num, self.cate_features_num], dtype=int)
        self.nume_data = np.zeros([self.objects_num, self.nume_features_num])

        self.value_frequency_list = np.zeros(int(self.values_num))
        self.value_list = []
        self.first_value_index = []
        self.first_value_index.append(0)
        self.value_feature_indicator = []

    # calculate basic statistical information
    def data_prepare(self):
        # calc first_value_index, count value frequency,
        # generate value list for each feature, indicate the feature index of the values
        for i in range(self.cate_features_num):
            column = self.data_matrix[:, i]
            this_value_list = np.unique(column).tolist()
            feature_value_num = len(this_value_list)
            self.first_value_index.append(self.first_value_index[i] + feature_value_num)
            for j in range(feature_value_num):
                self.value_feature_indicator.append(i)

            frequency_map = Counter(column)
            for jj, item in enumerate(this_value_list):
                frequency = frequency_map.get(item)
                self.value_frequency_list[self.first_value_index[i] + jj] = frequency
            self.value_list.append(this_value_list)

        # process categorical space
        for i in range(0, self.cate_features_num):
            this_value_list = self.value_list[i]
            this_value_index_map = {}
            for j in range(len(this_value_list)):
                this_value_index_map[this_value_list[j]] = self.first_value_index[i] + j
            for k in range(self.objects_num):
                self.cate_data[k][i] = this_value_index_map[self.data_matrix[k][i]]

        # normalise numerical features using max-min normalisation method, normalised features will range from 0 to 1
        for i in range(self.cate_features_num, self.all_features_num):

            column_max = np.max(self.data_matrix[:, i])
            column_min = np.min(self.data_matrix[:, i])
            if column_max - column_min == 0:
                raise ValueError("all values in feature {} ({}) are zero.".format(i, self.head[i]))
            for j in range(self.objects_num):
                self.nume_data[j][i - self.cate_features_num] = \
                    float(self.data_matrix[j][i] - column_min) / float(column_max - column_min)

        return
