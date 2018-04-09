# from lvq import *
# from scipy.io import arff
# import numpy as np
# import pandas as pd

# data = arff.loadarff('datasets/jm1.arff')
# df = pd.DataFrame(data[0])
# df['defects'] = pd.factorize(df['defects'])[0] + 1
# df_norm = (df - df.mean()) / (df.max() - df.min())
# train = df_norm.values
# prototypes = train_prototypes(train, 200, 0.3, 10)