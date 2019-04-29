import numpy as np
import pandas as pd
from scipy.spatial import distance
import math as math
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pickle

# read data
kmer_neg_data = pd.read_csv('kmer_negative_training.csv', header=None, error_bad_lines=False)
kmer_pos_data = pd.read_csv('kmer_positive_training.csv', header=None, error_bad_lines=False)

kmer_full_data = pd.concat([kmer_pos_data, kmer_neg_data], ignore_index=True, join='outer')

features = kmer_full_data.iloc[:, 0: 2066]
label = kmer_full_data.iloc[:, 2066]

# set up hyper parameter search
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# print(features)
# print(label)

x_train, x_test, y_train, y_test = train_test_split(features, label)

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=10, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)

rf.fit(x_train, y_train)

# save the model
filename = 'my_model.sav'
pickle.dump(rf_random, open(filename, 'wb'))




print(rf.score(x_test, y_test))
print(rf_random.best_score_)
