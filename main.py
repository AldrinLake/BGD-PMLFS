import os
import time
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
import csv
import sys
import socket
import fcntl
import multiprocessing
import numpy as np
import itertools
import AttributeReduction



record_filepath = 'file_for_disambiguation/dataset_record.csv'

reduction_result_file_path = 'file_for_no_disambiguation/result/'

dataset_use_path = 'datasets_use.txt'

dataset_file_url = r'../../PML datasets/synthetic/noise_ratio_{}/{}.mat'

record_head = ['数据集', '样本个数', '特征个数', '标签个数','噪声率' '数据预处理方式', 'param_for_radius', 'param_alpha', '约简后特征个数', '运行时间', '记录时刻', '运行设备']

processing_num = 3

noise_ratio_list = [
    # 0,
    # 0.2, 0.4, 0.6, 0.8,
    # 'real'
]

def main():

    if not os.path.exists(reduction_result_file_path):
        os.makedirs(reduction_result_file_path)

    if not os.path.exists(dataset_use_path):
        print('dataset ' + dataset_use_path + ' not found')
        sys.exit()

        csv_file = csv.writer(open(record_filepath, 'w', newline='', encoding='utf_8_sig'))
        csv_file.writerow(record_head)
    # =========================================================

    # dataset = ['CHD_49', 'Society', ]
    datasets = []
    f = open(dataset_use_path)
    line = f.readline()
    while line:
        # 读取没有被注释掉的数据集
        if line.find('//') == -1:
            datasets.append(line.replace('\n', ''))
        line = f.readline()
    f.close()
    print(datasets)

    preProcessMethod = ['minMax']  # minMax, standard, mM_std, std_mM
    # param_list_for_radius = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
    # param_alpha = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    param_list_for_radius = [0.05,0.25,0.45,0.65,0.85]
    param_alpha = [0.1,0.3,0.5,0.7,0.9]

    temp = list(itertools.product(datasets, noise_ratio_list, preProcessMethod, param_list_for_radius, param_alpha))
    temp.reverse()
    parameters_list = multiprocessing.Manager().list(temp)
    Lock = multiprocessing.Manager().Lock()
    proc_list = []
    for proc_index in range(processing_num):
        parameter_dist = {
            'proc_index': proc_index,
            'parameters_list': parameters_list,
            "lock": Lock
        }
        proc = multiprocessing.Process(target=SingleProcess,args=(parameter_dist,))
        proc_list.append(proc)
        proc.start()
        time.sleep(0.5)
    for proc in proc_list:
        proc.join()
    for proc in proc_list:
        proc.close()


def SingleProcess(parameter_dist):
    processing_index = parameter_dist['proc_index']
    share_lock = parameter_dist['lock']
    params_list = parameter_dist['parameters_list']
    while True:
        share_lock.acquire()
        if len(params_list) == 0:
            break
        params = params_list.pop()
        dataset_name = params[0]
        noise_ratio = params[1]
        preProcessMethod = params[2]
        param_for_radius = params[3]
        param_alpha = params[4]

        share_lock.release()
        print("==== process {}, dataset:{}, noise:{},{}, radius:{},alpha:{}, time:{}".format(processing_index, dataset_name, noise_ratio, preProcessMethod,param_for_radius,param_alpha,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        try:
            # read a dataset
            data = scio.loadmat(dataset_file_url.format(noise_ratio, dataset_name))
            X = data['features'][:, :]
            Y = data['labels'][:, :]
            # normalize
            if preProcessMethod == 'minMax':
                minMax = MinMaxScaler()  # Normalize data
                X = minMax.fit_transform(X[:, :])
            elif preProcessMethod == 'standard':
                Ss = StandardScaler()
                X = Ss.fit_transform(X[:, :])
            else:
                print("without preprosses")
            features_rank, run_time = AttributeReduction.AttributeReduction(X=X,Y=Y, param_for_radius=param_for_radius, param_alpha=param_alpha, dataset_name=dataset_name,noise_ratio=noise_ratio)
            # write the ranked features to txt file
            file_path = "{}/noise_ratio_{}/{}/{}/".format(reduction_result_file_path,noise_ratio, preProcessMethod, dataset_name)
            if os.path.exists(file_path) is False:
                os.makedirs(file_path)
            note = open("{}/{}_{}.txt".format(file_path, param_for_radius,param_alpha), mode='w')
            note.write(','.join(str(i) for i in features_rank))
            note.close()
        except Exception as e:
            print(e)
            with open(record_filepath, 'a', newline='') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                writer = csv.writer(f)
                writer.writerow([dataset_name, X.shape[0], X.shape[1], Y.shape[1],noise_ratio, preProcessMethod,str(param_for_radius),str(param_alpha), 'Exception:' + str(e), '',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), socket.gethostname()])
                fcntl.flock(f, fcntl.LOCK_UN)
            continue
        with open(record_filepath, 'a', newline='') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            writer = csv.writer(f)
            writer.writerow([dataset_name, X.shape[0], X.shape[1], Y.shape[1],noise_ratio, preProcessMethod,str(param_for_radius), str(param_alpha), len(features_rank), run_time, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), socket.gethostname()])
            fcntl.flock(f, fcntl.LOCK_UN)
    share_lock.release()




if __name__ == '__main__':
    main()
