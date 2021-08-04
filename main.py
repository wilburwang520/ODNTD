import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from DataLoader import DataLoader
from ODNTD import ODNUTFramework
import os


def main(input_path):
    rounds = 1
    auc_array = np.zeros((rounds, 1))
    time_array = np.zeros((rounds, 1))
    dim_array = np.zeros((rounds, 1))
    data_describe=''
    for m in range(rounds):
        data = DataLoader(input_path)
        data.data_prepare()
        dim = data.all_features_num
        n = len(data.data_matrix)
        labels = data.list_of_class
        data_describe="{}, {}, {}, ".format(data.data_name, n, dim)
        scores_all, totalTime, dim_vec, it = ODNUTFramework(data)
        final_scores_all =  scores_all
        time_array[m] = totalTime
        dim_array[m] = dim_vec[-1]
        roc_auc = roc_auc_score(labels, final_scores_all)
        auc_array[m] = roc_auc

    roc_auc = np.mean(auc_array)
    runtime = np.mean(time_array)
    avgDim = np.mean(dim_array)

    print_text = data_describe+"{:.4}, {:.4}, {:.4}s".format( avgDim, roc_auc, runtime)
    print(print_text)
    doc = open('out.txt', 'a')
    print(print_text, file=doc)
    doc.close()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    input_root = "data/"

    if os.path.isdir(input_root):
        for file_name in os.listdir(input_root):
            if file_name.endswith(".csv"):
                input_path = os.path.join(input_root, file_name)
                main(input_path)
    else:
        input_path = input_root
        main(input_path)
