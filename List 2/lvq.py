from random import randrange
from math import sqrt
from math import isnan
import numpy as np

def euclidean_distance(x, xq):
    sum = 0.0
    
    for i in range(len(x)-1):
        if isnan(x[i]) or isnan(xq[i]):
            print(x[i], xq[i])
        sum += float((x[i]-xq[i])**2)

    return sqrt(sum)

def get_nearest_prototype(prototypes, xq, n=1):
    distances = []

    for i, prototype in enumerate(prototypes):
        dist = euclidean_distance(prototype, xq)
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    return [distances[i] for i in range(n)]

def random_prototype(train):
    n_instances = len(train)
    n_attributes = len(train[0])
    prototype = [train[randrange(n_instances)][i] for i in range(n_attributes)]

    return prototype

def verify_window(m):
    w = 0.2
    s = (1-w)/(1+w)
    di = m[0][1]
    dj = m[1][1]

    return min(di/dj, dj/di) > s

def lvq_1(train, n_prototypes, learn_rate, epochs):
    prototypes = [random_prototype(train) for i in range(n_prototypes)]

    for epoch in range(epochs):
        alpha = learn_rate * (1.0-(epoch/float(epochs)))
        for xq in train:
            idx = get_nearest_prototype(prototypes, xq)[0][0]
            for i in range(len(xq)-1):
                error = xq[i] - prototypes[idx][i]
                if prototypes[idx][-1] == xq[-1]:
                    prototypes[idx][i] += (alpha * error)
                else:
                    prototypes[idx][i] -= (alpha * error)

    return prototypes

def lvq_21(train, n_prototypes, learn_rate, epochs):
    prototypes = [random_prototype(train) for i in range(n_prototypes)]

    for epoch in range(epochs):
        alpha = learn_rate * (1.0-(epoch/float(epochs)))
        for xq in train:
            m = get_nearest_prototype(prototypes, xq, n=2)
            mi = m[0][0]
            mj = m[1][0]
            for i in range(len(xq)-1):
                error_i = xq[i] - prototypes[mi][i]
                error_j = xq[i] - prototypes[mj][i]
                if (verify_window(m)):
                    if prototypes[mi][-1] == xq[-1] and prototypes[mj][-1] != xq[-1]:
                        prototypes[mi][i] += (alpha * error_i)
                        prototypes[mj][i] -= (alpha * error_j)
                    elif prototypes[mi][-1] != xq[-1] and prototypes[mj][-1] == xq[-1]:
                        prototypes[mi][i] -= (alpha * error_i)
                        prototypes[mj][i] += (alpha * error_j)

    return prototypes