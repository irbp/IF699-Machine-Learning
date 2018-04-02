import numpy as np
import pandas as pd

def euclidean_distance(x1, x2):
    return np.sqrt(((x1 - x2)**2).sum())

def calculate_N(df):
    N = []
    keys = list(df.keys())
    attr = list(df.keys())[:-1]

    for i in attr:
        N.append(dict(df[i].value_counts()))
        
    return N

def calculate_NC(df):
    NC = {}
    keys = list(df.keys())
    attr = keys[:-1]
    classes = keys[-1]
    
    for k in df[classes].unique():
        NC[k] = []
        for a in attr:
            count = dict(df[a].where(df[classes] == k).value_counts())
            NC[k].append(count)
        
    return NC

def vdm_distance(x1, x2, N, NC):
    vdm = []
    q = 1

    for i in range(len(x1)):
        N1 = N[i][x1[i]]
        N2 = N[i][x2[i]]
        res = 0
        for k in NC.keys():
            if (x1[i] in NC[k][i]):
                NC1 = NC[k][i][x1[i]]
            if (x2[i] in NC[k][i]):
                NC2 = NC[k][i][x2[i]]
            res += (abs((NC1 / N1) - (NC2 / N2)) ** q)
        vdm.append(res)
        
    vdm = np.asarray(vdm)
    
    return np.sqrt(vdm.sum())

def hvdm_distance(x1, x2, N, NC, max_a, min_a):
    distances = []
    q = 1
    
    for i in range(len(x1)):
        if (isinstance(x1[i], str)):
            N1 = N[i][x1[i]]
            N2 = N[i][x2[i]]
            res = 0
            for k in NC.keys():
                if (x1[i] in NC[k][i]):
                    NC1 = NC[k][i][x1[i]]
                if (x2[i] in NC[k][i]):
                    NC2 = NC[k][i][x2[i]]
                res += (abs((NC1 / N1) - (NC2 / N2)) ** q)
            distances.append(res)
        else:
            distances.append(abs(x1[i] - x2[i]) / (max_a[i] - min_a[i]))

    distances = np.asarray(distances)
    return np.sqrt((distances**2).sum())