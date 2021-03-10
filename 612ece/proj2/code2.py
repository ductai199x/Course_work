import sys, os
import pickle
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold

with open('mnist_X_train.pkl', 'rb') as f:
    X = pickle.load(f)

with open('mnist_y_train.pkl', 'rb') as f:
    y = pickle.load(f)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

from scipy.ndimage.filters import gaussian_filter

def transform_data(data):
    mask = data > np.max(data)*0.65
    mask = mask.astype(float)
    for i in range(0, mask.shape[0]):
        mask[i] = gaussian_filter(mask[i], sigma=0.7)
    mask = mask > (np.max(mask) - np.mean(mask))/8 + np.mean(mask)
    mask = mask.astype(float)
    return mask

X_train_transformed = transform_data(X_train)
X_test_transformed = transform_data(X_test)

N_COMPONENTS = 40

pca_trans = PCA(n_components=N_COMPONENTS).fit(X_train_transformed)
X_train_trans_proj = pca_trans.transform(X_train_transformed)
X_test_trans_proj = pca_trans.transform(X_test_transformed)

X = np.append(X_train_trans_proj, X_test_trans_proj, axis=0)
y = np.append(y_train, y_test, axis=0)

cl1 = KNeighborsClassifier(n_neighbors=4, p=2)
cl2 = SVC(kernel='rbf', C=10.0, probability=True, max_iter=1000)
cl3 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')

eclf1 = VotingClassifier(estimators=[
    ('knn', cl1), ('svm', cl2), ('logreg', cl3)], voting='soft', weights=[0.92516, 0.96250, 0.85], n_jobs=16)

skf = StratifiedKFold(n_splits=4)
result = []
for train_idx, test_idx in skf.split(X, y):
    X_train_set = X[train_idx]
    X_test_set = X[test_idx]
    y_train_set = y[train_idx]
    y_test_set = y[test_idx]

    eclf1.fit(X_train_set, y_train_set)
    score = eclf1.score(X_test_set, y_test_set)

    print(score)
    result.append(score)
    
print(f"Average score: {np.average(result)}")

