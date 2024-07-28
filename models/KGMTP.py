#Panjie Wang, Jiang Wu, Yuan Wei, Taiyong Li
from warnings import simplefilter
from typing import Tuple

import numpy as np

#Panjie Wang, Jiang Wu, Yuan Wei, Taiyong Li Kernel Grouping for Time Series Classification
# with Multiple Transformations and Pooling Operators
from numba import njit, prange, vectorize,float32

simplefilter(action='ignore', category=FutureWarning)


def fit(X, num_features=10_000, max_dilations_per_kernel=32, weights=None):
    _, input_length = X.shape

    num_kernels = len(weights)
    kernel_length = weights.shape[1]
    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel,
                                                          num_kernels, kernel_length)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles, weights)

    return dilations, num_features_per_dilation, biases, weights


def _fit_dilations(input_length, num_features, max_dilations_per_kernel, num_kernels, kernel_length):
    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (kernel_length - 1))
    dilations, num_features_per_dilation = \
        np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
                  return_counts=True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32)


@vectorize("float32(float32,float32)", nopython=True, cache=True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0


@njit("float32[:](float64[:,:],int32[:],int32[:],float32[:],float32[:,:])",
      fastmath=True, parallel=False, cache=True)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles, weights):
    indices1 = np.array([0, 1, 2, 3, 4, 5],
                        dtype=np.int32)
    indices2 = np.array([0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 2, 1, 3, 1, 4, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 3, 5, 4, 5],
                        dtype=np.int32).reshape(15, 2)
    indices3 = np.array(
        [0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 3, 4, 0, 3, 5, 0, 4, 5, 1, 2, 3, 1, 2, 4, 1,
         2,
         5, 1, 3, 4, 1, 3, 5, 1, 4, 5, 2, 3, 4, 2, 3, 5, 2, 4, 5, 3, 4, 5],
        dtype=np.int32).reshape(20, 3)
    indices4 = np.array(
        [0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2, 5, 0, 1, 3, 4, 0, 1, 3, 5, 0, 1, 4, 5, 0, 2, 3, 4, 0, 2, 3, 5, 0, 2, 4, 5, 0,
         3,
         4, 5, 1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 4, 5, 1, 3, 4, 5, 2, 3, 4, 5], dtype=np.int32).reshape(15, 4)

    indices5 = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 5, 0, 1, 2, 4, 5, 0, 1, 3, 4, 5, 0, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                        dtype=np.int32).reshape(6, 5)
    num_examples, input_length = X.shape


    kernel_length = 6

    num_kernels = weights.shape[0]
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        input_length = X.shape[1] + dilation
        A = np.zeros(input_length, dtype=np.float64)
        G = np.zeros(input_length, dtype=np.float64)

        padding = ((kernel_length) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(6):

            feature_index_end = feature_index_start + num_features_this_dilation
            _X1 = X[np.random.randint(num_examples)]
            
            A[:-dilation] = -_X1  
            G[:-dilation] = _X1 + _X1 + _X1 + _X1 + _X1 + _X1  

            C_alpha = np.zeros(input_length, dtype=np.float64)
            C_alpha[:] = A

            C_gamma = np.zeros((6, input_length), dtype=np.float64)
            C_gamma[6 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(6 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(6 // 2 + 1, 6):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0 = indices1[kernel_index]

            C1 = C_alpha + C_gamma[index_0]
            C1=C1.astype(np.float32)
            biases[feature_index_start:feature_index_end] = np.quantile(C1, quantiles[
                                                                            feature_index_start:feature_index_end])

            feature_index_start = feature_index_end
        for kernel_index in range(0, 15):

            feature_index_end = feature_index_start + num_features_this_dilation
            _X1 = X[np.random.randint(num_examples)]
            
            A[:-dilation] = -_X1  
            G[:-dilation] = _X1 + _X1 + _X1  

            C_alpha = np.zeros(input_length, dtype=np.float64)
            C_alpha[:] = A

            C_gamma = np.zeros((6, input_length), dtype=np.float64)
            C_gamma[6 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(6 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(6 // 2 + 1, 6):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1 = indices2[kernel_index]

            C1 = C_alpha + C_gamma[index_0] + C_gamma[index_1]
            C1 = C1.astype(np.float32)
            biases[feature_index_start:feature_index_end] = np.quantile(C1, quantiles[
                                                                            feature_index_start:feature_index_end])

            feature_index_start = feature_index_end
        for kernel_index in range(0, 20):

            feature_index_end = feature_index_start + num_features_this_dilation
            _X1 = X[np.random.randint(num_examples)]
            
            A[:-dilation] = -_X1
            G[:-dilation] = _X1 + _X1  

            C_alpha = np.zeros(input_length, dtype=np.float64)
            C_alpha[:] = A

            C_gamma = np.zeros((6, input_length), dtype=np.float64)
            C_gamma[6 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(6 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(6 // 2 + 1, 6):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2 = indices3[kernel_index]

            C1 = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
            C1 = C1.astype(np.float32)
            biases[feature_index_start:feature_index_end] = np.quantile(C1, quantiles[
                                                                            feature_index_start:feature_index_end])

            feature_index_start = feature_index_end
        for kernel_index in range(0, 15):

            feature_index_end = feature_index_start + num_features_this_dilation
            _X1 = X[np.random.randint(num_examples)]
            
            A[:-dilation] = -_X1 - _X1
            G[:-dilation] = _X1 + _X1 + _X1  

            C_alpha = np.zeros(input_length, dtype=np.float64)
            C_alpha[:] = A

            C_gamma = np.zeros((6, input_length), dtype=np.float64)
            C_gamma[6 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(6 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(6 // 2 + 1, 6):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2, index_3 = indices4[kernel_index]

            C1 = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2] + C_gamma[index_3]
            C1 = C1.astype(np.float32)
            biases[feature_index_start:feature_index_end] = np.quantile(C1, quantiles[
                                                                            feature_index_start:feature_index_end])

            feature_index_start = feature_index_end
        for kernel_index in range(0, 6):

            feature_index_end = feature_index_start + num_features_this_dilation
            _X1 = X[np.random.randint(num_examples)]
            
            A[:-dilation] = -_X1 - _X1 - _X1 - _X1 - _X1  
            G[:-dilation] = _X1 + _X1 + _X1 + _X1 + _X1 + _X1  

            C_alpha = np.zeros(input_length, dtype=np.float64)
            C_alpha[:] = A

            C_gamma = np.zeros((6, input_length), dtype=np.float64)
            C_gamma[6 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(6 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(6 // 2 + 1, 6):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2, index_3, index_4 = indices5[kernel_index]

            C1 = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2] + C_gamma[index_3] + C_gamma[index_4]
            C1 = C1.astype(np.float32)
            biases[feature_index_start:feature_index_end] = np.quantile(C1, quantiles[
                                                                            feature_index_start:feature_index_end])

            feature_index_start = feature_index_end

    return biases



@njit("(float64[:,:],Tuple((int32[:],int32[:],float32[:],float32[:,:])),int32)",
          fastmath=True, parallel=True, cache=True)
def transform(X, parameters, n_features_per_kernel=5) -> Tuple[float32[:, :], float32[:, :]]:
    indices1 = np.array([0, 1, 2, 3, 4, 5],
                        dtype=np.int32)
    indices2 = np.array([0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 2, 1, 3, 1, 4, 1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 3, 5, 4, 5],
                        dtype=np.int32).reshape(15, 2)
    indices3 = np.array(
        [0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 3, 4, 0, 3, 5, 0, 4, 5, 1, 2, 3, 1, 2, 4, 1,
         2,
         5, 1, 3, 4, 1, 3, 5, 1, 4, 5, 2, 3, 4, 2, 3, 5, 2, 4, 5, 3, 4, 5],
        dtype=np.int32).reshape(20, 3)
    indices4 = np.array(
        [0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2, 5, 0, 1, 3, 4, 0, 1, 3, 5, 0, 1, 4, 5, 0, 2, 3, 4, 0, 2, 3, 5, 0, 2, 4, 5, 0,
         3,
         4, 5, 1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 4, 5, 1, 3, 4, 5, 2, 3, 4, 5], dtype=np.int32).reshape(15, 4)

    indices5 = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 5, 0, 1, 2, 4, 5, 0, 1, 3, 4, 5, 0, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                        dtype=np.int32).reshape(6, 5)
    dilations, num_features_per_dilation, biases, weights = parameters
    kernel_length = weights.shape[1]

    num_examples, input_length = X.shape
    num_kernels = len(weights)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)
    features = np.zeros((num_examples, num_features * n_features_per_kernel), dtype=np.float32)
    features_hydra= np.zeros((num_examples, num_kernels * num_dilations*2), dtype=np.float32)

    for example_index in prange(num_examples):
        hydra_feature_index=0

        feature_index_start = 0
        _X = X[example_index]
        for dilation_index in prange(num_dilations):
            dilation = dilations[dilation_index]

            input_length = _X.shape[0] + dilation
            A = np.zeros(input_length, dtype=np.float64)
            G = np.zeros(input_length, dtype=np.float64)
            A[:-dilation] = -_X  
            G[:-dilation] = _X + _X + _X + _X + _X + _X  

            padding = ((kernel_length) * dilation) // 2
            output_length = _X.shape[0] + (2 * padding) - ((kernel_length - 1) * dilation)
            C_max_1 = np.ones(output_length) * -10000
            C_min_1 = np.ones(output_length) * 10000
            C_max_index_1 = np.ones(output_length, dtype=np.int32)
            C_min_index_1 = np.ones(output_length, dtype=np.int32)
            C_hydra_max_1 = np.zeros(6, dtype=np.float32)
            C_hydra_min_1 = np.zeros(6, dtype=np.float32)

            C_max_2 = np.ones(output_length) * -10000
            C_min_2 = np.ones(output_length) * 10000
            C_max_index_2 = np.ones(output_length, dtype=np.int32)
            C_min_index_2 = np.ones(output_length, dtype=np.int32)
            C_hydra_max_2 = np.zeros(15, dtype=np.float32)
            C_hydra_min_2 = np.zeros(15, dtype=np.float32)

            C_max_3 = np.ones(output_length) * -10000
            C_min_3 = np.ones(output_length) * 10000
            C_max_index_3 = np.ones(output_length, dtype=np.int32)
            C_min_index_3 = np.ones(output_length, dtype=np.int32)
            C_hydra_max_3 = np.zeros(20, dtype=np.float32)
            C_hydra_min_3 = np.zeros(20, dtype=np.float32)

            C_max_4 = np.ones(output_length) * -10000
            C_min_4 = np.ones(output_length) * 10000
            C_max_index_4 = np.ones(output_length, dtype=np.int32)
            C_min_index_4 = np.ones(output_length, dtype=np.int32)
            C_hydra_max_4 = np.zeros(15, dtype=np.float32)
            C_hydra_min_4 = np.zeros(15, dtype=np.float32)

            C_max_5 = np.ones(output_length) * -10000
            C_min_5 = np.ones(output_length) * 10000
            C_max_index_5 = np.ones(output_length, dtype=np.int32)
            C_min_index_5 = np.ones(output_length, dtype=np.int32)
            C_hydra_max_5 = np.zeros(6, dtype=np.float32)
            C_hydra_min_5 = np.zeros(6, dtype=np.float32)




            num_features_this_dilation = num_features_per_dilation[dilation_index]
            C_alpha = np.zeros(input_length, dtype=np.float64)
            C_alpha[:] = A

            C_gamma = np.zeros((6, input_length), dtype=np.float64)
            C_gamma[6 // 2, :] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(6 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(6 // 2 + 1, 6):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation
            for kernel_index in prange(6):
                feature_index_end = feature_index_start + num_features_this_dilation
                index_0 = indices1[kernel_index]

                C1 = C_alpha + \
                     C_gamma[index_0]

                C = C1.astype(np.float32)

                for feature_count in range(num_features_this_dilation):
                    feature_index = feature_index_start + feature_count
                    _bias = biases[feature_index]
                    ppv = 0
                    last_val = 0
                    max_stretch = 0.0
                    mean_index = 0
                    mean = 0
                    zero_count = 0
                    for j in range(C.shape[0]):
                        if(feature_count==0):
                            if (C_max_1[j] <= C[j]):
                                C_max_1[j] = C[j]
                                C_max_index_1[j] = kernel_index
                            if (C_min_1[j] >= C[j]):
                                C_min_1[j] = C[j]
                                C_min_index_1[j] = kernel_index
                        if (j < C.shape[0] - 1 and ((C[j] > _bias and C[j + 1] < _bias)
                                or (C[j] < _bias and C[j + 1] > _bias))):
                            zero_count += 1

                        if C[j] > _bias:
                            ppv += 1
                            mean_index += j
                            mean += C[j] + _bias
                        elif C[j] < _bias:
                            stretch = j - last_val

                            if stretch > max_stretch:
                                max_stretch = stretch
                            last_val = j
                    stretch = C.shape[0] - 1 - last_val
                    if stretch > max_stretch:
                        max_stretch = stretch

                    end = feature_index
                    features[example_index, end] = ppv / C.shape[0]
                    end = end + num_features
                    features[example_index, end] = mean / ppv if ppv > 0 else 0
                    end = end + num_features
                    features[example_index, end] = max_stretch
                    end = end + num_features
                    features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                    end = end + num_features
                    features[example_index, end] = zero_count / C.shape[0]
                feature_index_start = feature_index_end
            A = np.zeros(input_length, dtype=np.float64)
            G = np.zeros(input_length, dtype=np.float64)
            A[:-dilation] = -_X  
            G[:-dilation] = _X + _X + _X  


            num_features_this_dilation = num_features_per_dilation[dilation_index]
            C_alpha = np.zeros(input_length, dtype=np.float64)
            C_alpha[:] = A

            C_gamma = np.zeros((6, input_length), dtype=np.float64)
            C_gamma[6 // 2, :] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(6 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(6 // 2 + 1, 6):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation
            for kernel_index in prange(0, 15):
                feature_index_end = feature_index_start + num_features_this_dilation

                index_0, index_1 = indices2[kernel_index]

                C1 = C_alpha + \
                     C_gamma[index_0] + C_gamma[index_1]

                C = C1.astype(np.float32)

                for feature_count in range(num_features_this_dilation):
                    feature_index = feature_index_start + feature_count
                    _bias = biases[feature_index]
                    ppv = 0
                    last_val = 0
                    max_stretch = 0.0
                    mean_index = 0
                    mean = 0
                    zero_count = 0
                    for j in range(C.shape[0]):
                        if (feature_count == 0):
                            if (C_max_2[j] <= C[j]):
                                C_max_2[j] = C[j]
                                C_max_index_2[j] = kernel_index
                            if (C_min_2[j] >= C[j]):
                                C_min_2[j] = C[j]
                                C_min_index_2[j] = kernel_index

                        if (j < C.shape[0] - 1 and ((C[j] > _bias and C[j + 1] < _bias)
                                or (C[j] < _bias and C[j + 1] > _bias))):
                            zero_count += 1

                        if C[j] > _bias:
                            ppv += 1
                            mean_index += j
                            mean += C[j] + _bias
                        elif C[j] < _bias:
                            stretch = j - last_val

                            if stretch > max_stretch:
                                max_stretch = stretch
                            last_val = j
                    stretch = C.shape[0] - 1 - last_val
                    if stretch > max_stretch:
                        max_stretch = stretch

                    end = feature_index
                    features[example_index, end] = ppv / C.shape[0]

                    end = end + num_features
                    features[example_index, end] = mean / ppv if ppv > 0 else 0
                    end = end + num_features
                    features[example_index, end] = max_stretch
                    end = end + num_features
                    features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                    end = end + num_features
                    features[example_index, end] = zero_count / C.shape[0]
                feature_index_start = feature_index_end
            A = np.zeros(input_length, dtype=np.float64)
            G = np.zeros(input_length, dtype=np.float64)
            A[:-dilation] = -_X
            G[:-dilation] = _X + _X  

            padding = ((kernel_length) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]
            C_alpha = np.zeros(input_length, dtype=np.float64)
            C_alpha[:] = A

            C_gamma = np.zeros((6, input_length), dtype=np.float64)
            C_gamma[6 // 2, :] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(6 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(6 // 2 + 1, 6):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation
            for kernel_index in prange(0, 20):
                feature_index_end = feature_index_start + num_features_this_dilation

                index_0, index_1, index_2 = indices3[kernel_index]

                C1 = C_alpha + \
                     C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                C = C1.astype(np.float32)

                for feature_count in range(num_features_this_dilation):

                    feature_index = feature_index_start + feature_count
                    _bias = biases[feature_index]
                    ppv = 0
                    last_val = 0
                    max_stretch = 0.0
                    mean_index = 0
                    mean = 0
                    zero_count = 0
                    for j in range(C.shape[0]):
                        if (feature_count == 0):
                            if (C_max_3[j] <= C[j]):
                                C_max_3[j] = C[j]
                                C_max_index_3[j] = kernel_index
                            if (C_min_3[j] >= C[j]):
                                C_min_3[j] = C[j]
                                C_min_index_3[j] = kernel_index
                        if (j < C.shape[0] - 1 and ((C[j] > _bias and C[j + 1] < _bias)
                                or (C[j] < _bias and C[j + 1] > _bias))):
                            zero_count += 1
                        if C[j] > _bias:
                            ppv += 1
                            mean_index += j
                            mean += C[j] + _bias
                        elif C[j] < _bias:
                            stretch = j - last_val

                            if stretch > max_stretch:
                                max_stretch = stretch
                            last_val = j
                    stretch = C.shape[0] - 1 - last_val
                    if stretch > max_stretch:
                        max_stretch = stretch

                    end = feature_index
                    features[example_index, end] = ppv / C.shape[0]

                    end = end + num_features
                    features[example_index, end] = mean / ppv if ppv > 0 else 0
                    end = end + num_features
                    features[example_index, end] = max_stretch
                    end = end + num_features
                    features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                    end = end + num_features
                    features[example_index, end] = zero_count / C.shape[0]
                feature_index_start = feature_index_end
            A = np.zeros(input_length, dtype=np.float64)
            G = np.zeros(input_length, dtype=np.float64)
            A[:-dilation] = -_X - _X
            G[:-dilation] = _X + _X + _X  

            padding = ((kernel_length) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]
            C_alpha = np.zeros(input_length, dtype=np.float64)
            C_alpha[:] = A

            C_gamma = np.zeros((6, input_length), dtype=np.float64)
            C_gamma[6 // 2, :] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(6 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(6 // 2 + 1, 6):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation
            for kernel_index in prange(0, 15):
                feature_index_end = feature_index_start + num_features_this_dilation

                

                index_0, index_1, index_2, index_3 = indices4[kernel_index]

                C1 = C_alpha + \
                     C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2] + C_gamma[index_3]

                C = C1.astype(np.float32)

                for feature_count in range(num_features_this_dilation):




                    feature_index = feature_index_start + feature_count
                    _bias = biases[feature_index]
                    ppv = 0
                    last_val = 0
                    max_stretch = 0.0
                    mean_index = 0
                    mean = 0

                    zero_count = 0
                    for j in range(C.shape[0]):
                        if (feature_count == 0):
                            if (C_max_4[j] <= C[j]):
                                C_max_4[j] = C[j]
                                C_max_index_4[j] = kernel_index
                            if (C_min_4[j] >= C[j]):
                                C_min_4[j] = C[j]
                                C_min_index_4[j] = kernel_index
                        if (j < C.shape[0] - 1 and ((C[j] > _bias and C[j + 1] < _bias)
                                or (C[j] < _bias and C[j + 1] > _bias))):
                            zero_count += 1

                        if C[j] > _bias:
                            ppv += 1
                            mean_index += j
                            mean += C[j] + _bias
                        elif C[j] < _bias:
                            stretch = j - last_val

                            if stretch > max_stretch:
                                max_stretch = stretch
                            last_val = j
                    stretch = C.shape[0] - 1 - last_val
                    if stretch > max_stretch:
                        max_stretch = stretch

                    end = feature_index
                    features[example_index, end] = ppv / C.shape[0]

                    end = end + num_features
                    features[example_index, end] = mean / ppv if ppv > 0 else 0
                    end = end + num_features
                    features[example_index, end] = max_stretch
                    end = end + num_features
                    features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                    end = end + num_features
                    features[example_index, end] = zero_count / C.shape[0]
                feature_index_start = feature_index_end
            A = np.zeros(input_length, dtype=np.float64)
            G = np.zeros(input_length, dtype=np.float64)
            A[:-dilation] = -_X - _X - _X - _X - _X
            G[:-dilation] = _X + _X + _X + _X + _X + _X  

            padding = ((kernel_length) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]
            C_alpha = np.zeros(input_length, dtype=np.float64)
            C_alpha[:] = A

            C_gamma = np.zeros((6, input_length), dtype=np.float64)
            C_gamma[6 // 2, :] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(6 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(6 // 2 + 1, 6):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation
            for kernel_index in prange(0, 6):
                feature_index_end = feature_index_start + num_features_this_dilation

                

                index_0, index_1, index_2, index_3, index_4 = indices5[kernel_index]

                C1 = C_alpha + \
                     C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2] + C_gamma[index_3] + C_gamma[index_4]

                C = C1.astype(np.float32)

                for feature_count in range(num_features_this_dilation):

                    feature_index = feature_index_start + feature_count
                    _bias = biases[feature_index]
                    ppv = 0
                    last_val = 0
                    max_stretch = 0.0
                    mean_index = 0
                    mean = 0

                    zero_count = 0
                    for j in range(C.shape[0]):
                        if (feature_count == 0):
                            if (C_max_5[j] <= C[j]):
                                C_max_5[j] = C[j]
                                C_max_index_5[j] = kernel_index
                            if (C_min_5[j] >= C[j]):
                                C_min_5[j] = C[j]
                                C_min_index_5[j] = kernel_index

                        if (j < C.shape[0] - 1 and ((C[j] > _bias and C[j + 1] < _bias)
                                or (C[j] < _bias and C[j + 1] > _bias))):
                            zero_count += 1

                        if C[j] > _bias:
                            ppv += 1
                            mean_index += j
                            mean += C[j] + _bias
                        elif C[j] < _bias:
                            stretch = j - last_val

                            if stretch > max_stretch:
                                max_stretch = stretch
                            last_val = j
                    stretch = C.shape[0] - 1 - last_val
                    if stretch > max_stretch:
                        max_stretch = stretch

                    end = feature_index
                    features[example_index, end] = ppv / C.shape[0]
                    end = end + num_features
                    features[example_index, end] = mean / ppv if ppv > 0 else 0
                    end = end + num_features
                    features[example_index, end] = max_stretch
                    end = end + num_features
                    features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                    end = end + num_features
                    features[example_index, end] = zero_count / C.shape[0]
                feature_index_start = feature_index_end
            for i in prange(output_length):
                C_hydra_max_1[C_max_index_1[i]] += C_max_1[i]
                C_hydra_min_1[C_min_index_1[i]] += 1

                C_hydra_max_2[C_max_index_2[i]] += C_max_2[i]
                C_hydra_min_2[C_min_index_2[i]] += 1

                C_hydra_max_3[C_max_index_3[i]] += C_max_3[i]
                C_hydra_min_3[C_min_index_3[i]] += 1

                C_hydra_max_4[C_max_index_4[i]] += C_max_4[i]
                C_hydra_min_4[C_min_index_4[i]] += 1

                C_hydra_max_5[C_max_index_5[i]] += C_max_5[i]
                C_hydra_min_5[C_min_index_5[i]] += 1

            features_hydra[example_index, hydra_feature_index:hydra_feature_index + 6] = C_hydra_max_1
            features_hydra[example_index, hydra_feature_index + 6:hydra_feature_index + 21] = C_hydra_max_2
            features_hydra[example_index, hydra_feature_index + 21:hydra_feature_index + 41] = C_hydra_max_3
            features_hydra[example_index, hydra_feature_index + 41:hydra_feature_index + 56] = C_hydra_max_4
            features_hydra[example_index, hydra_feature_index + 56:hydra_feature_index + 62] = C_hydra_max_5

            hydra_feature_index += num_kernels

            features_hydra[example_index, hydra_feature_index:hydra_feature_index + 6] = C_hydra_min_1
            features_hydra[example_index, hydra_feature_index + 6:hydra_feature_index + 21] = C_hydra_min_2
            features_hydra[example_index, hydra_feature_index + 21:hydra_feature_index + 41] = C_hydra_min_3
            features_hydra[example_index, hydra_feature_index + 41:hydra_feature_index + 56] = C_hydra_min_4
            features_hydra[example_index, hydra_feature_index + 56:hydra_feature_index + 62] = C_hydra_min_5
            hydra_feature_index += num_kernels
    res = (features, features_hydra)
    return res

class KGMTP:

    def __init__(
            self,
            num_features=50000,
            weights=0,
    ):
        self.weights = weights
        self.base_parameters = None
        self.diff1_parameters = None
        self.n_features_per_kernel = 5
        self.num_features = num_features
        self.num_kernels = int(self.num_features / self.n_features_per_kernel)

    def fit(self, x_train):
        self.base_parameters = fit(
            x_train,
            num_features=self.num_kernels,
            weights=self.weights

        )

        x_train_transform, x_train_hydra_transform = transform(
            x_train,
            self.base_parameters,
            self.n_features_per_kernel
        )

        x_train_transform = np.nan_to_num(x_train_transform)
        return x_train_transform,x_train_hydra_transform

    def predict(self, x):
        x_transform,x_hydra_transform = transform(
            x,
            self.base_parameters,
            self.n_features_per_kernel
        )

        x_transform = np.nan_to_num(x_transform)

        return x_transform,x_hydra_transform
