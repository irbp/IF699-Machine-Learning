import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.combine import SMOTEENN
from scipy.io import arff

from sklearn.decomposition import PCA as PCASK

class PCA:
    def __init__(self, X):
        self.__X = X
        self.__X_std = 0

    def __covariance_matrix(self, X):
        mean_vec = np.mean(X, axis=0)
        cov_mat = (X - mean_vec).T.dot(X - mean_vec) / (X.shape[0] - 1)

        return cov_mat

    def __calculate_eigens(self):
        self.__X_std = StandardScaler().fit_transform(self.__X)
        cov_mat = self.__covariance_matrix(self.__X_std)
        eig_vals, eig_vec = np.linalg.eig(cov_mat)

        return eig_vals, eig_vec

    def get_components(self, num):
        eig_vals, eig_vecs = self.__calculate_eigens()
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        eig_pairs.sort()
        eig_pairs.reverse()

        w = [eig_pairs[i][1] for i in range(num)]
        matrix_w = np.array(w).T
        new_X = self.__X_std.dot(matrix_w)

        return new_X

def knn(X, y, k=7):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    classifier = KNeighborsClassifier(n_neighbors=7)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    return score
        
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

    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y)

    # Balancing dataset with SMOTE
    smote_enn = SMOTEENN(random_state=0)
    X, y = smote_enn.fit_sample(X, y)

    # Selecting principal components with PCA
    pca = PCA(X)
    new_X = pca.get_components(6)
    print("Principal components:\n{}".format(new_X))

    # Avaliating the new dataset
    result = knn(new_X, y)
    print("The accuracy is: {:.2f}%".format(result * 100))

    X = StandardScaler().fit_transform(X)
    pca_sklearn = PCASK(n_components=6)
    new_x = pca_sklearn.fit_transform(X)
    print("Principal components sklearn:\n{}".format(new_x))

if __name__ == "__main__":
    main()