import argparse
import os,sys
import time
root_dir = os.getcwd().replace('\\', '/') + '/../'
print(root_dir)
inputdata_dir=root_dir+'data/ucr/'
sys.path.append(root_dir)
from itertools import combinations
from warnings import simplefilter


import torch

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from scipy import fftpack
import numpy as np
from sklearn.metrics import accuracy_score
from models.KGMTP import KGMTP
from utils.data_loader import  read_dataset,ucr_112_dataset_names
from utils.tools import SparseScaler
simplefilter(action='ignore', category=FutureWarning)
num_features = 50000
parser = argparse.ArgumentParser()

kernel_length = 6
length = 6

value_length = 1
indices = np.array([_ for _ in combinations(np.arange(length), value_length)], dtype=np.int32)
num_kernels = len(indices)
weights1 = np.ones((num_kernels, length), dtype=np.float32)  # see note
weights1 = -1 * weights1
for i in range(num_kernels):
    for j in range(value_length):
        weights1[i][indices[i][j]] = (length - value_length) * 1 / value_length

value_length = 2
indices = np.array([_ for _ in combinations(np.arange(length), value_length)], dtype=np.int32)
num_kernels = len(indices)
weights2 = np.ones((num_kernels, length), dtype=np.float32)  # see note
weights2 = -1 * weights2
for i in range(num_kernels):
    for j in range(value_length):
        weights2[i][indices[i][j]] = (length - value_length) * 1 / value_length

value_length = 3
indices = np.array([_ for _ in combinations(np.arange(length), value_length)], dtype=np.int32)
num_kernels = len(indices)
weights3 = np.ones((num_kernels, length), dtype=np.float32)  # see note
weights3 = -1 * weights3
for i in range(num_kernels):
    for j in range(value_length):
        weights3[i][indices[i][j]] = (length - value_length) * 1 / value_length

value_length = 4
indices = np.array([_ for _ in combinations(np.arange(length), value_length)], dtype=np.int32)
num_kernels = len(indices)
weights4 = np.ones((num_kernels, length), dtype=np.float32)  # see note
weights4 = -1 * weights4
for i in range(num_kernels):
    for j in range(value_length):
        weights4[i][indices[i][j]] = (length - value_length) * 1 / value_length
weights4*=2
value_length = 5
indices = np.array([_ for _ in combinations(np.arange(length), value_length)], dtype=np.int32)
num_kernels = len(indices)
weights5 = np.ones((num_kernels, length), dtype=np.float32)  # see note
weights5 = -1 * weights5
for i in range(num_kernels):
    for j in range(value_length):
        weights5[i][indices[i][j]] = (length - value_length) * 1 / value_length
weights5*=5


weights = np.r_[weights1,weights2,weights3,weights4,weights5]




# numba.set_num_threads(1)
# torch.set_num_threads(1)
arguments = parser.parse_args()

compiled = False
print(f"RUNNING".center(80, "="))
total_train_time=0
for dataset_name in ucr_112_dataset_names:
    datasets_dict = read_dataset(inputdata_dir, dataset_name)
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    input_length = x_train.shape[1]

    x_train_hibert = np.zeros(x_train.shape,dtype=np.float32)
    x_test_hibert = np.zeros(x_test.shape,dtype=np.float32)
    for i in range(x_train.shape[0]):
        x_train_hibert[i] = fftpack.hilbert(x_train[i])
    for i in range(x_test.shape[0]):
        x_test_hibert[i] = fftpack.hilbert(x_test[i])
    train = np.c_[x_train, x_train_hibert]
    test = np.c_[x_test, x_test_hibert]


  
    X_train = raw_X_train = train
    X_test =raw_X_test = test
    raw_y_train = y_train
    raw_y_test = y_test
    X = np.vstack((X_train, X_test))
    Y = np.hstack((y_train, y_test))
    acc = 0
    total_acc = 0
    for k in range(30):
        scaler = SparseScaler()
        classifier = make_pipeline(
            StandardScaler(),
            RidgeClassifierCV(
                alphas=np.logspace(-3, 3, 10),
                normalize=True
        ))



        if (k > 0):
            all_indices = np.arange(len(X))
            training_indices = np.loadtxt("../data/indices112/{}_INDICES_TRAIN.txt".format(dataset_name),
                                          skiprows=k,
                                          max_rows=1).astype(np.int32)
            test_indices = np.setdiff1d(all_indices, training_indices, assume_unique=True)
            X_train, y_train = X[training_indices, :], Y[training_indices]
            X_test, y_test = X[test_indices, :], Y[test_indices]

        x_train_imf = X_train[:, X_train.shape[1] // 2:]
        x_train = X_train[:, :X_train.shape[1] // 2]
        x_train_diff = np.diff(x_train, 1)

        x_test_imf = X_test[:, X_test.shape[1] // 2:]
        x_test = X_test[:, :X_test.shape[1] // 2]
        x_test_diff = np.diff(x_test, 1)

        start=time.perf_counter()
        KGMTP_base = KGMTP(num_features=num_features // 3, weights=weights)
        x_train_transform,x_train_hydra_transform =KGMTP_base.fit(x_train=x_train)

        KGMTP_hilbert = KGMTP(num_features=num_features // 3, weights=weights)
        x_train_transform_imf,x_train_hydra_transform_imf=KGMTP_hilbert.fit(x_train=x_train_imf)

        KGMTP_diff = KGMTP(num_features=num_features // 3, weights=weights)
        x_train_transform_diff,x_train_hydra_transform_diff=KGMTP_diff.fit(x_train=x_train_diff)


        X_training_transform = np.c_[x_train_transform,x_train_transform_imf,x_train_transform_diff]

        X_training_hydra_transform = np.array(scaler.fit_transform(torch.FloatTensor(np.c_[x_train_hydra_transform,x_train_hydra_transform_imf,x_train_hydra_transform_diff])))


        classifier.fit(np.c_[X_training_transform,X_training_hydra_transform],y_train)


        # single_fit_time=time.perf_counter()-start
        # total_train_time+=single_fit_time
        # ensemble_input = open('../results/KGMTP.txt',
        #                       mode='a+')
        #
        # ensemble_input.write(
        #     dataset_name + '\t' + str(single_fit_time) + '\n')
        # ensemble_input.close()
        x_test_transform,x_test_hydra_transform = KGMTP_base.predict(x_test)
        x_test_imf_transform,x_test_imf_hydra_transform =KGMTP_hilbert.predict(x_test_imf)
        x_test_diff_transform,x_test_diff_hydra_transform = KGMTP_diff.predict(x_test_diff)

        X_testing_transform = np.c_[x_test_transform,x_test_imf_transform,x_test_diff_transform]

        X_testing_hydra_transform = np.array(scaler.transform(torch.FloatTensor(np.c_[x_test_hydra_transform,x_test_imf_hydra_transform,x_test_diff_hydra_transform])))


        predictions = classifier.predict(np.c_[X_testing_transform,X_testing_hydra_transform])

        acc = accuracy_score(predictions,y_test)
        total_acc += acc
        print(acc)
    print(dataset_name + '\tavg\t' + str(total_acc/30.0) )
    # print(dataset_name + '\t' + str(single_fit_time))
# print('total_train_time\t' + str(total_train_time))
    ensemble_input = open('../results/KGMTP.txt',
                          mode='a+')

    ensemble_input.write(
        dataset_name + '\t' + str(total_acc / 30.0) + '\n')
    ensemble_input.close()
