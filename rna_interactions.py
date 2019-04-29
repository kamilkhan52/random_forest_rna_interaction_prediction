import numpy as np
import pandas as pd
from scipy.spatial import distance
import math as math
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# read data
kmer_neg_data = pd.read_csv('kmer_negative_training.csv')
kmer_pos_data = pd.read_csv('kmer_positive_training.csv')

kmer_full_data = pd.concat(kmer_neg_data, kmer_pos_data, ignore_index=True)

features = kmer_neg_data.iloc[:, 0: 2066]
label = kmer_neg_data.iloc[:, 2066]

print(features.head(10))
print(label.head(10))

x_train, x_test, y_train, y_test = train_test_split(features, label)
