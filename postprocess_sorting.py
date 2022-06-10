from fileinput import filename
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sorting_qubo as sq


dir = r"C:\Users\jules\Documents\Technische Wiskunde\Applied quantum algorithms\results_sorting"
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
    vecX, penalty_r, penalty_c = pickle.load(f)


print(f"input vector: {vecX}")
print(f"penalty_r, penalty_c: {penalty_r}, {penalty_c}")

best_sample = record.loc[record['energy'].idxmin(), :]
# print(f"best sample: {best_sample}")
print(f"best sample energy: {best_sample['energy']}")
print(f"best sample number of occurrences: {int(best_sample['num_occurrences'])} (out of {np.sum(record['num_occurrences'])})")
# print(f"energy: {sq.energy(P, matR, vecR)}")

array_size = len(vecX)**2
P = np.zeros(array_size)
for i in range(array_size):
    P[i] = best_sample[f'x_{i}']

matR, vecR = sq.initializeSortQUBO(vecX, penalty_r, penalty_c)


print(f"permutation matrix:\n{P.reshape(len(vecX),len(vecX))}")
print(f"result: {P.reshape(len(vecX), len(vecX)) @ vecX}")

# optimal solution
vecY = np.sort(vecX)
P_opt = np.zeros((len(vecX), len(vecX)))
for i in range(len(vecX)):
    P_opt[np.where(vecY == vecX[i])[0], i] = 1

print(f"optimal permutation matrix:\n{P_opt}")
# optimal energy
print(f"optimal energy: {sq.energy(P_opt.flatten('F'), matR, vecR)}")
