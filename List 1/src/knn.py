import pandas as pd
import numpy as np
from .distances import *

def neighborhood(x_train, y_train, xq, data_type=0, N=None, NC=None, max_a=[], min_a=[]):
    distances = []
    
    if (data_type == 0):
        for i, x in enumerate(x_train):
            d = euclidean_distance(x, xq)
            distances.append([d, y_train[i]])
    elif (data_type == 1):
        for i, x in enumerate(x_train):
            d = vdm_distance(x, xq, N, NC)
            distances.append([d, y_train[i]])
    else:
        for i, x in enumerate(x_train):
            d = hvdm_distance(x, xq, N, NC, max_a, min_a)
            distances.append([d, y_train[i]])
        
    sorted_distances = sorted(distances, key=lambda z:z[0])
    neighbors = np.asarray(sorted_distances)
    
    return neighbors

def knn(x_train, y_train, xq, k_values, data_type=0, N=None, NC=None, max_a=[], min_a=[]):
    pred = []
    neighbors = neighborhood(x_train, y_train, xq, data_type, N, NC, max_a, min_a)

    for k in k_values:
        freq = np.unique(neighbors[:k,-1], return_counts=True)
        pred.append(freq[0][freq[1].argmax()])
        
    return pred

def weighted_knn(x_train, y_train, xq, k_values, data_type=0, N=None, NC=None, max_a=[], min_a=[]):
    pred = []
    distances = neighborhood(x_train, y_train, xq, data_type, N, NC, max_a, min_a)

    for k in k_values:
        classes = {}
        weights = []
        neighbors = distances[:k, :]
        for n in neighbors:
            if (float(n[0]) != 0):
                weights.append([1/float(n[0]),n[1]])
            else:
                weights.append([999999999,n[1]])
        for w in weights:
            if (w[-1] in classes):
                classes[w[-1]] += w[0]
            else:
                classes[w[-1]] = w[0]
        pred.append(max(classes, key=classes.get))
            
    return pred

def cross_validation(data, n_fold, k_values, with_weight=False, data_type=0, dataframe=[]):
    size = data.shape[0]
    k_size = int(size / n_fold)
    x = [data[i:i+k_size,:-1] for i in range(0, size, k_size)]
    y = [data[i:i+k_size,-1] for i in range(0, size, k_size)]
    accs = []
    N = []
    NC = {}
    max_a = []
    min_a = []

    data = np.asarray(data)

    if (data_type != 0):
        N = calculate_N(dataframe)
        NC = calculate_NC(dataframe)
        max_a = [max(data[:,i]) for i in range(data.shape[1])]
        min_a = [min(data[:,i]) for i in range(data.shape[1])]

    for i in range(n_fold):
        preds = []
        x_train = np.asarray(x[:i] + x[i+1:])
        y_train = np.asarray(y[:i] + y[i+1:])
        x_train = np.reshape(x_train, (size-k_size,-1))
        y_train = np.reshape(y_train, size-k_size)
        x_test = np.asarray(x[i])
        y_test = np.asarray(y[i])
        if (with_weight == True):
            for i, xq in enumerate(x_test):
                preds.append(weighted_knn(x_train, y_train, xq, k_values, data_type, N, NC, max_a, min_a))
        else:
            for i, xq in enumerate(x_test):
                preds.append(knn(x_train, y_train, xq, k_values, data_type, N, NC, max_a, min_a))
                
        preds = np.asarray(preds)
        acc = []
        for i in range(len(k_values)):
            acc.append(100*float((preds[:,i] == y_test).sum()) / preds.shape[0])
        
        accs.append(acc)
        
    accs = np.asarray(accs)
    
    return [accs[:,i].sum()/accs.shape[0] for i in range(accs.shape[-1])]