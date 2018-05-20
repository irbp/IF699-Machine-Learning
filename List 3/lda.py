import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.io import arff

class LDA:
    def __init__(self, X, y):
        self.__X = X
        self.__y = y
        self.__rows = X.shape[0]
        self.__cols = X.shape[1]
        self.__n_cl = len(np.unique(y))

        enc = LabelEncoder()
        label_encoder = enc.fit(self.__y)
        self.__y = label_encoder.transform(self.__y) + 1

    def __sw_matrix(self, mean_vecs):
        S_W = np.zeros((self.__cols, self.__cols))
        for cl, mv in zip(range(1, self.__n_cl+1), mean_vecs):
            class_sc_mat = np.zeros((self.__cols, self.__cols))
            for row in self.__X[self.__y == cl]:
                row, mv = row.reshape(self.__cols, 1), mv.reshape(self.__cols, 1)
                class_sc_mat += (row - mv).dot((row - mv).T)
            S_W += class_sc_mat

        return S_W

    def __sb_matrix(self, mean_vecs):
        overall_mean = np.mean(self.__X, axis=0)

        S_B = np.zeros((self.__cols, self.__cols))
        for i, mean_vec in enumerate(mean_vecs):
            n = self.__X[self.__y==i+1, :].shape[0]
            mean_vec = mean_vec.reshape(self.__cols, 1)
            overall_mean = overall_mean.reshape(self.__cols, 1)
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
            
        return S_B

    def get_components(self, num):
        mean_vecs = []
        for cl in range(1, self.__n_cl+1):
            mean_vecs.append(np.mean(self.__X[self.__y==cl], axis=0))

        S_W = self.__sw_matrix(mean_vecs)
        S_B = self.__sb_matrix(mean_vecs)

        # Calculating eigenvalues and eigenvectors
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        # eig_pairs.sort()
        # eig_pairs.reverse()

        w = [eig_pairs[i][1] for i in range(num)]
        matrix_w = np.array(w).T
        new_X = self.__X.dot(matrix_w)

        return new_X

def main():
    data = arff.loadarff("datasets/jm1.arff")
    df = pd.DataFrame(data[0])
    df.dropna(how="all", inplace=True)
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values

    # Removing NaN from the dataset
    is_nan = ~np.isnan(X).any(axis=1)
    X = X[is_nan]
    y = y[is_nan]

    lda = LDA(X, y)
    new_X = lda.get_components(6)
    print("New components:\n{}".format(new_X))

if __name__ == "__main__":
    main()