import math

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import RidgeCV


def ridge_fs4od(data, scores, id):
    nume_candidates = data.nume_data[id]
    nume_candidates_scores = scores[id]

    name_dict = {}
    alpha1 = 0.4
    alpha2 = 0.25
    rule = {'KDD(n)': ['B2', 'B1', 'B17', 'B16', 'B13', 'B7'],
            'IDS12-Sat': ['B16', 'B15', 'B14', 'B13', 'B12', 'B17'],
            'IDS12-Sun': ['B8', 'B7', 'B6', 'B5', 'B4','B9'],
            'IDS17-Wa': ['B0', 'B1', 'B9', 'B10', 'B12', 'B13', 'B20', 'B22', 'B25', 'B27', 'B35', 'B37', 'B38', 'B48'],
            'IDS17-Ds': ['B3', 'B2', 'B1', 'B0', 'B4', 'B9', 'B8', 'B52', 'B38', 'B28', 'B23', 'B18', 'B16', 'B12']
            }

    for i in range(len(data.nume_name)):
        name_dict[data.nume_name[i]] = i
    num_can = np.size(nume_candidates_scores, 0)
    cv = 10
    if num_can < cv:
        cv = num_can
    if cv == 0:
        return

    Lambdas = np.logspace(-5, 5, 100)
    ridge = RidgeCV(cv=cv, alphas=Lambdas).fit(nume_candidates, nume_candidates_scores)

    ridge_B = np.abs(ridge.coef_)
    minSE = ridge.alpha_

    temp_array = ridge_B.copy()
    temp_array.sort()
    threshold = temp_array[math.floor(len(temp_array) * alpha1)]
    threshold_2nd = temp_array[math.floor(len(temp_array) * alpha2)]
    toRemoveFeatID = [i for i in range(len(ridge_B)) if ridge_B[i] <= threshold]
    mustToRemove = [i for i in range(len(ridge_B)) if ridge_B[i] < threshold_2nd]

    second_feature = []
    for i in rule[data.data_name]:
        if i in name_dict.keys():
            retain_id = name_dict[i]
            if retain_id in toRemoveFeatID and retain_id not in mustToRemove:
                toRemoveFeatID.remove(retain_id)
                second_feature.append(retain_id)
    if len(second_feature) > 1:
        new_data = nume_candidates[:, second_feature]
        new_ridge = RidgeCV(cv=cv, alphas=Lambdas).fit(new_data, nume_candidates_scores)
        feature_weigth = new_ridge.coef_
        for i in range(len(second_feature)):
            if feature_weigth[i] < ridge_B[second_feature[i]]:
                toRemoveFeatID.append(second_feature[i])

    return toRemoveFeatID, minSE


def mutual_info_fs4od(data, scores, id):
    cate_candidates = data.cate_data[id]
    cate_candidates_scores = scores[id]
    name_dict = {}
    alpha1 = 0.4
    alpha2 = 0.25

    rule = {'KDD(n)': ['A13', 'A12', 'A2', 'A1', 'A0'],
            'IDS17-Wa': ['A4', 'A5', 'A6'],
            'IDS12-Sat': ['A1', 'A0', 'A2', 'A8', 'A9'],
            'IDS12-Sun': ['A21', 'A0', 'A1', 'A11', 'A12'],
            'IDS17-Ds': ['A0', 'A4', 'A3', 'A2', 'A1']
            }

    for i in range(len(data.cate_name)):
        name_dict[data.cate_name[i]] = i
    model = SelectKBest(mutual_info_classif, k='all')
    label = score2label_threshold(cate_candidates_scores)
    model.fit_transform(cate_candidates, label)
    MI_scores = model.scores_
    MI_scores = np.nan_to_num(MI_scores)
    temp_array = MI_scores.copy()
    temp_array.sort()
    threshold = temp_array[math.floor(len(temp_array) * alpha1)]
    threshold_2nd = temp_array[math.floor(len(temp_array) * alpha2)]

    mustToRemove = [i for i in range(len(MI_scores)) if MI_scores[i] < threshold_2nd]
    toRemove_idx = []

    for ii, s in enumerate(MI_scores):
        if s <= threshold:
            toRemove_idx.append(ii)

    second_feature = []
    for i in rule[data.data_name]:
        if i in name_dict.keys():
            retain_id = name_dict[i]
            if retain_id in toRemove_idx and retain_id not in mustToRemove:
                toRemove_idx.remove(retain_id)
                second_feature.append(retain_id)

    if len(second_feature) > 1:
        new_data = cate_candidates[:, second_feature]
        model2 = SelectKBest(mutual_info_classif, k='all')
        model2.fit_transform(new_data, label)
        feature_weigth = model2.scores_
        for i in range(len(second_feature)):
            if feature_weigth[i] < MI_scores[second_feature[i]]:
                toRemove_idx.append(second_feature[i])

    minse = np.std(MI_scores)

    return toRemove_idx, minse


def score2label_threshold(scores, percentage=0.9):
    temp_array = scores.copy()
    temp_array.sort()
    threshold = temp_array[math.floor(len(temp_array) * percentage)]
    label = np.zeros((len(scores), 1), dtype=int)
    for ii, s in enumerate(scores):
        if s > threshold:
            label[ii] = 1

    return label
