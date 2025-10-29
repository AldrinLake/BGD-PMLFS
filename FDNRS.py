# Fuzzy Decision Neighborhood Rough Set for multi-label data
# in this version, chebyshev distance is used as default, and the method is applied to label confidence matrix, i.e., label distribution data
import numpy as np
from scipy.spatial.distance import cdist

temp_file_path = 'temp/'

class FDNRS:
    def __init__(self, X, Y, radius_param, dataset_name, noise_ratio):
        self.X = X
        self.Y = Y
        self.sample_num = X.shape[0]
        self.radius_param = radius_param
        # prevent repeat calculation of matrix
        self.neighborhood_relation_matrix = None

        self.chebyshev_on_each_feature_matrix = np.load("{}/chebyshev_minMax/{}.npy".format(temp_file_path, dataset_name))
        r = np.load("{}/temp_of_label_calculation/related_to_noise_ratio_{}/{}.npz".format(temp_file_path, noise_ratio,dataset_name))
        self.different_label_number_matrix = r['label_different_matrix']
        self.same_label_number_matrix = r['label_same_matrix']

    def update_neighborhood_matrix(self, new_selected_feature, distance_matrix):
        distance_matrix_for_new_feature = self.chebyshev_on_each_feature_matrix[new_selected_feature]
        distance_matrix2 = np.where(distance_matrix < distance_matrix_for_new_feature,distance_matrix_for_new_feature, distance_matrix)
        neighborhood_relation_matrix = distance_matrix2 <= self.radius_param

        self.temp_distance_matrix = distance_matrix2
        self.instance_similarity_matrix = 1 / (1+self.temp_distance_matrix)
        self.neighborhood_relation_matrix = neighborhood_relation_matrix


    def calculate_neighborhood_dependency(self):
        """
        :return: lower approximation
        """
        instance_similarity_matrix = self.instance_similarity_matrix
        neighborhood_relation_matrix = self.neighborhood_relation_matrix
        same_label_number_matrix = self.same_label_number_matrix
        # temp1 = neighborhood_relation_matrix * same_label_number_matrix * instance_similarity_matrix
        temp1 = neighborhood_relation_matrix * same_label_number_matrix
        row_sum = np.sum(temp1, axis=1)
        temp2 = row_sum / np.sum(neighborhood_relation_matrix, axis=1)
        temp2 = np.where(temp2>0.5,temp2,0)  # 0.5 is a fixed parameter
        neighborhood_dependency = np.mean(temp2)
        return neighborhood_dependency, self.temp_distance_matrix



    def calculate_neighborhood_uncertainty(self):
        """
        the neighbors have more different labels with respect to a target sample means a higher neighborhood uncertainty
        :return: neighborhood uncertainty
        """
        instance_similarity_matrix = self.instance_similarity_matrix
        neighborhood_relation_matrix = self.neighborhood_relation_matrix
        different_label_number_matrix = self.different_label_number_matrix
        # temp1 = neighborhood_relation_matrix * different_label_number_matrix * instance_similarity_matrix
        temp1 = neighborhood_relation_matrix * different_label_number_matrix
        row_sum = np.sum(temp1, axis=1)
        temp2 = row_sum / np.sum(neighborhood_relation_matrix, axis=1)
        neighborhood_uncertainty = np.mean(temp2)
        return neighborhood_uncertainty


if __name__ == '__main__':
    X = np.array([
        [0.5],
        [0.4],
        [0.6],
        [0.8]
    ])

    Y = np.array([
        [1,0,0,1],
        [1,1,0,0],
        [1,1,0,0],
        [1,0,0,0]
    ])

    res = np.std(Y,axis=0)
    print(res)
