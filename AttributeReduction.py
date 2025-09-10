import time

import scipy.io as scio
from numpy import *
import numpy as np
import warnings
import PySimpleGUI as sg
from scipy.spatial.distance import cdist
from numpy.distutils.fcompiler import none
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
from FDNRS import FDNRS



def AttributeReduction(X, Y, param_for_radius, param_alpha, dataset_name=None, noise_ratio=None):
    """
    :param X: instance space, row denotes instance
    :param Y: multi-label space, row denote a label vector
    :param radius: radius for neighborhood granule
    :param distance_metric: ['chebyshev']
    :return: a ranking set of features
    """
    start_time = time.time()
    num, dim = X.shape
    # retain specific number of features according to the cardinality of feature set
    select_feature_number = None
    if dim > 1000:
        select_feature_number = int(dim * 0.1)
    if 1000 >= dim > 500:
        select_feature_number = int(dim * 0.2)
    if 500 >= dim > 100:
        select_feature_number = int(dim * 0.3)
    if dim <= 100:
        select_feature_number = int(dim * 0.4)
    select_feature_number = select_feature_number + 1

    candidate_features = list(range(dim))
    selected_features = []

    distance_matrix = np.zeros((num, num))
    fdnrs = FDNRS(X=X, Y=Y, radius_param=param_for_radius, dataset_name=dataset_name, noise_ratio=noise_ratio)
    while True:
        flag = None
        max_value = float('-inf')
        temp_distance_matrix2 = np.zeros((num, num))
        for f in candidate_features:
            fdnrs.update_neighborhood_matrix(new_selected_feature=f, distance_matrix=distance_matrix)
            sig_dependency, temp_distance_matrix = fdnrs.calculate_neighborhood_dependency()
            sig_uncertainty = fdnrs.calculate_neighborhood_uncertainty()
            sig = param_alpha*sig_dependency - (1-param_alpha)*sig_uncertainty
            if sig > max_value:
                flag = f
                max_value = sig
                temp_distance_matrix2 = temp_distance_matrix
        # 实时进度条
        sg.one_line_progress_meter('progress bar', len(selected_features), select_feature_number,
                                   "dataset:{} \nnoise ratio:{} \nradius:{} \nparam_alpha:{}\n维度X{},Y{}\nselect:{}".format(
                                       dataset_name, noise_ratio, param_for_radius, param_alpha, str(X.shape),
                                       str(Y.shape), str(flag)))
        selected_features.append(flag)
        candidate_features.remove(flag)
        distance_matrix = temp_distance_matrix2

        if len(selected_features) >= select_feature_number:
            break

    end_time = time.time()
    # concat the ordered feature subset and unchecked feature subset [ordered feature subset，unchecked feature subset]
    final_result = selected_features + candidate_features
    # the initial number of the feature is set to 1 rather than 0
    final_result = list(np.array(final_result) + 1)
    return final_result, end_time - start_time

if __name__ == '__main__':
    datasetName = 'Birds'
    data = scio.loadmat('../../datasets/' + datasetName + '.mat')
    X = data['features'][:, :]
    Y = data['labels'][:, :]
    Ss = MinMaxScaler()  # Normalize data
    X = Ss.fit_transform(X[:, :])
    param_alpha = 0.5
    res, time = AttributeReduction(X=X,Y=Y, param_for_radius=0.4, param_alpha=param_alpha, dataset_name=datasetName)
    print(res)
    print(time)
