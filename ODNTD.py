import sys
from datetime import datetime

import numpy as np

import MIX

from feature_selection import ridge_fs4od, mutual_info_fs4od


def ODNUTFramework(data):
    n = len(data.data_matrix)
    batch_size = 64
    episode_max = 10000
    epsilon = 0.06
    k = 0.3
    allt = datetime.now()

    scores, cate_scores, nume_scores = MIX.fit(data, batch_size=batch_size, episode_max=episode_max, epsilon=epsilon,
                                               k=k, verbose=False)

    # Initialization
    cate_flag = True
    nume_flag = True
    nume_minLoss = sys.maxsize
    cate_minLoss = sys.maxsize
    it = 0
    maxIt = 20
    scores_matrix = np.zeros((n, maxIt))
    minSE_vector = np.zeros((maxIt, 1))
    cate_minSE_vector = np.zeros((maxIt, 1))
    nume_minSE_vector = np.zeros((maxIt, 1))
    cate_minSE = 0
    nume_minSE = 0

    dim_vec = np.zeros((maxIt, 1))

    while True:

        if it >= maxIt:
            break
        candidate_scores, outlier_candidate_idx = cutoff(scores)

        if cate_flag:
            cate_removeID, cate_minSE = mutual_info_fs4od(data, cate_scores, outlier_candidate_idx)
            if cate_minSE < cate_minLoss:
                cate_minLoss = cate_minSE
                if cate_minSE == 0:
                    cate_flag = False
                    cate_removeID = []
            else:
                cate_removeID = []
                cate_flag = False
        else:
            cate_removeID = []
            cate_minSE = cate_minSE
        cate_minSE_vector[it] = cate_minSE

        nume_removeID = []
        if nume_flag:
            nume_toRemoveFeatID, nume_minSE = ridge_fs4od(data, nume_scores, outlier_candidate_idx)
            if nume_minSE < nume_minLoss:
                for id in nume_toRemoveFeatID:
                    nume_removeID.append(id + data.cate_features_num)
                nume_minLoss = nume_minSE
                if nume_minSE == 0:
                    nume_flag = False
                    nume_removeID = []
            else:
                nume_removeID = []
                nume_flag = False
                cate_flag = False
        else:
            nume_removeID = []
            nume_minSE = nume_minSE

        nume_minSE_vector[it] = nume_minSE
        toRemoveFeatID = cate_removeID + nume_removeID
        if len(toRemoveFeatID) > 0:
            data.data_dorp(toRemoveFeatID)
            data.data_prepare()
            scores_matrix[:, it] = normLengthNormalization(scores)
            scores, cate_scores, nume_scores = MIX.fit(data, batch_size=batch_size, episode_max=episode_max,
                                                       epsilon=epsilon, k=k,
                                                       verbose=False)

            newD = data.all_features_num
            dim_vec[it] = newD

        else:
            break


        it = it + 1

    scores_matrix[:, it] = normLengthNormalization(scores)

    nume_minSE_norm = normLengthNormalization(nume_minSE_vector[1:it + 1])
    cate_minSE_norm = normLengthNormalization(cate_minSE_vector[1:it + 1])
    for i in range(it):
        minSE_vector[i] = nume_minSE_norm[i] + cate_minSE_norm[i]

    scores_all = weightedSummation(scores_matrix, minSE_vector, it)
    dim_vec = dim_vec[0:it]
    totalTime = datetime.now() - allt
    return scores_all, totalTime.seconds, dim_vec, it


def weightedSummation(scores_matrix, minSE_vector, it):
    totalSE1 = sum(minSE_vector[0:it])
    weights1 = totalSE1 - minSE_vector[0:it]
    weights1 = weights1 / sum(weights1)
    if it == 1:
        weights1 = 1
    weightedScores1 = np.dot(scores_matrix[:, 1: it + 1], weights1)
    return weightedScores1


def normLengthNormalization(scores):
    total = sum(scores)
    if total == 0:
        ns = scores
    else:
        ns = scores / total
    return ns

def cutoff(scores):
    smu = np.mean(scores)
    sdelta = np.std(scores)
    # % sort_score = sort(scores);
    # % th = smu + sdelta * 1.3; % 70% confidence level
    # % th = smu + sdelta * 1.5275; % 70% confidence level
    th = smu + sdelta * 1.7321  # % 75% confidence level
    # % th = smu + sdelta * 2; % 80% confidence level
    # % th = smu + sdelta * 3; % 90% confidence level
    # % th = smu + sdelta * 4.3589; % 95% confidence level
    # % th = scores(outlier_num);
    outlier_candidate_idx = np.where(scores >= th)[0]
    scores_sh = np.array([scores[i] for i in outlier_candidate_idx])

    # % the minimum number of data objects in lasso is 5. This condition is
    # % added to ensure CINFO also works when the Catenlli's inequality-based
    # % cutoff returns less than 5 outlier candidates

    if np.size(outlier_candidate_idx, 0) < 5:
        sort_scores = sorted(scores, reverse=True)
        ids = np.argsort(scores)
        ids = np.flipud(ids)
        scores_sh = sort_scores[0:5]
        outlier_candidate_idx = ids[0:5]
    return scores_sh, outlier_candidate_idx