from random import randrange
from math import sqrt

def euclidean_distance(x, xq):
    sum = 0
    
    for i in range(len(x)-1):
        sum += float((x[i]-xq[i])**2)

    return sqrt(sum)

def get_nearest_prototype(prototypes, xq):
    distances = []

    for i, prototype in enumerate(prototypes):
        dist = euclidean_distance(prototype, xq)
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    return distances[0][0]

def random_prototype(train):
    n_instances = len(train)
    n_attributes = len(train[0])
    prototype = [train[randrange(n_instances)][i] for i in range(n_attributes)]

    return prototype

def train_prototypes(train, n_prototypes, learn_rate, epochs):
    prototypes = [random_prototype(train) for i in range(n_prototypes)]

    for epoch in range(epochs):
        alpha = learn_rate * (1.0-(epoch/float(epochs)))
        print(alpha)
        for xq in train:
            idx = get_nearest_prototype(prototypes, xq)
            for i in range(len(xq)-1):
                error = xq[i] - prototypes[idx][i]
                if prototypes[idx][-1] == xq[-1]:
                    prototypes[idx][i] += (alpha * error)
                else:
                    prototypes[idx][i] -= (alpha * error)

    return prototypes