from fileinput import filename
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from RSE_qubo import plot_results, shift_traces


dir = r"C:\Users\jules\Documents\Technische Wiskunde\Applied quantum algorithms\results"
# list of all files in result directory
files = os.listdir(dir)
# only files that start with "record" and latest file first
files = [f for f in files[::-1] if f.startswith("record")]
# list all resulting files
for i, file in enumerate(files):
    print(i, file, sep=":\t")

# ask user to select file. If no input, choose first file
file_index = input("select a file:")
if file_index == '':
    file_index = 0
else:
    try :
        file_index = int(file_index)
    except ValueError:
        print("invalid input")
        exit()  # exit program


filename = files[file_index]

# unpickle pandas dataframe
record = pd.read_pickle(dir + "\\" + filename)

#unpickle input
with open(dir + "\\" + filename.replace("record", "input"), 'rb') as f:
    d, K, M, N = pickle.load(f)

print(record.loc[record['energy'].idxmin(), :])
best_sample = record.loc[record['energy'].idxmin(), :]

d_shifted = shift_traces(d, K, M, N, best_sample)
plot_results(d, d_shifted, K, M, N)