import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from scipy.io import arff

DATASETS = ["jm1", "kc1"]
N_COMPONENTS = [1]

class LDA:
    def __init__(self, X, y):
        self.__X = X
        self.__y = y
        self.__rows = X.shape[0]
        self.__cols = X.shape[1]
        self.__n_cl = len(np.unique(y))

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
        eig_vals, eig_vecs = np.linalg.eigh(np.linalg.inv(S_W).dot(S_B))
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

        w = [eig_pairs[i][1] for i in range(num)]
        matrix_w = np.array(w).T
        new_X = self.__X.dot(matrix_w)

        return new_X

def load_dataset(file):
    data = arff.loadarff("datasets/" + file + ".arff")
    df = pd.DataFrame(data[0])
    df.dropna(how="all", inplace=True)
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values

    # Removing NaN from the dataset
    is_nan = ~np.isnan(X).any(axis=1)
    X = X[is_nan]
    y = y[is_nan]

    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y) + 1

    # Balancing dataset with SMOTE
    smote_enn = SMOTEENN(random_state=0)
    X, y = smote_enn.fit_sample(X, y)

    return X, y

def knn(X, y, k=7):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    classifier = KNeighborsClassifier(n_neighbors=7)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    return score

def plot_graph(acc_list, dataset_name):
    fig, ax = plt.subplots()
    index = np.arange(len(acc_list))

    rect = ax.bar(index, acc_list)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("LDA performance: {} dataset".format(dataset_name))
    ax.set_xticks(index)
    ax.set_xticklabels(("1", "21 (Original Dataset)"))

    for i in ax.patches:
        ax.text(i.get_x() + 0.2, i.get_height()-10, \
                str(round((i.get_height()), 2)) + "%", fontsize=11, color='white',
                    rotation=0)

    fig.tight_layout()
    plt.show()

def main():
    for dataset in DATASETS:
        print("-------------Dataset: {}-------------".format(dataset))
        X, y = load_dataset(dataset)

        lda = LDA(X, y)
        acc_l = []

        for n in N_COMPONENTS:
            # Selecting principal components with PCA
            new_X = lda.get_components(n)

            # Avaliating the new dataset
            acc_sum = 0
            for i in range(10):
                acc_sum += knn(new_X, y)
            acc = acc_sum * 100 / 10
            acc_l.append(acc)
            print("The accuracy for {} component(s) is: {:.2f}%".format(n, acc))

        acc_sum = 0
        for i in range(10):
            acc_sum += knn(X, y)
        acc = acc_sum * 100 / 10
        acc_l.append(acc)
        print("The accuracy for the original dataset is: {:.2f}%\n\n".format(acc))
        plot_graph(acc_l, dataset)

if __name__ == "__main__":
    main()