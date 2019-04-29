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


# def identify_input():
#     if len(input_data.columns) == 2067:
#         print("kmer")
#     elif input_data[1][1].isalpha():
#         print("sequence")
#     elif not input_data[1][1].isalpha():
#         print("deepbind")
#     else:
#         print("Input type not recognized")

def convert_deepbind_to_kmer():
    pass


def convert_seq_to_kmer():
    pass


deepbind_input_data = pd.read_csv('deepbind_test.csv', nrows=30, header=None)
convert_deepbind_to_kmer()

seq_input_data = pd.read_csv('seq_test.csv', nrows=30, header=None)
convert_seq_to_kmer()

kmer_input_data = pd.read_csv('kmer_test.csv', nrows=30, header=None, error_bad_lines=False)
