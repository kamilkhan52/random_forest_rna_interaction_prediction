import numpy as np
import pandas as pd
from scipy.spatial import distance
import math as math
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from time import time
import re
import pickle
import itertools as it
import khmer
import csv

filename = 'my_model.sav'
# load the model from disk
my_model = pickle.load(open(filename, 'rb'))

# def identify_input():
#     if len(input_data.columns) == 2067:
#         print("kmer")
#     elif input_data[1][1].isalpha():
#         print("sequence")
#     elif not input_data[1][1].isalpha():
#         print("deepbind")
#     else:
#         print("Input type not recognized")

# dictionary of kmers
all_bases = 'ACGT'
keywords = it.product(all_bases, repeat=5)
# print(type(keywords))
all_combinations = list(keywords)
', '.join(keywords)
dictOfKmers = {i: 0 for i in all_combinations}
# print(dictOfKmers)


def convert_deepbind_to_kmer():
    predict_for_seq()
    pass


def predict_for_seq(input_file):
    data_for_prediction = pd.read_csv(input_file, header=None, error_bad_lines=False)
    features = data_for_prediction.iloc[:, 0: 2066]

    # print(features)
    result = my_model.predict(features)
    # print(features.iloc[f, :])
    print("Output for " + str(input_file) + str(result))


def convert_seq_to_kmer():
    seq_number = 1
    full_row = []
    f = open('converted_seq_test.csv', 'w')
    row = dictOfKmers.copy()
    for j in range(0, len(seq_input_data.index)):
        for i in range(0, 2):
            for w in range(0, len(seq_input_data[i][j]) - 4):
                current_string = seq_input_data[i][j][w:w + 5]
                # print(current_string)
                row[tuple(current_string)] = 1

            # print("end of one seq")
            if seq_number < 2:  # if we are on the same row
                for value in row.values():
                    full_row.append(value)
                seq_number += 1
                # print(seq_number)
            else:  # if the this is the last sequence on the row
                for value in row.values():
                    full_row.append(value)
                for t in range(1, 20):
                    full_row.append(5)
                w = csv.writer(f)
                w.writerow(full_row)
                # print("written a row of length: " + str(len(full_row)))
                full_row.clear()
                seq_number = 1
                row.clear()
                row = dictOfKmers.copy()

    predict_for_seq('converted_seq.csv')


# deepbind_input_data = pd.read_csv('deepbind_test.csv', nrows=30, header=None)
# convert_deepbind_to_kmer()
#
seq_input_data = pd.read_csv('seq_test.csv', header=None)
convert_seq_to_kmer()
#
kmer_input_data = pd.read_csv('kmer_test.csv', header=None, error_bad_lines=False)
predict_for_seq('kmer_test.csv')

# Make predictions on inputs
