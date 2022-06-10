from RSE_qubo import qubo_dict, coefficient_matrices
import sorting_qubo as sq
from sample_problems import test2
from dwave.system import DWaveSampler, EmbeddingComposite
import pickle
import time
import numpy as np

def main():
    
    # number of traces
    M = 8

    # length of traces
    N = 100

    d, K = test2(M,N)
    d, K, M, N, matrices = coefficient_matrices(d, K)
    Q_dict = qubo_dict(d, K, M, N, *matrices)

    sampler = EmbeddingComposite(DWaveSampler())

    sampleset = sampler.sample_qubo(Q_dict, num_reads=200, annealing_time = 200, return_embedding=True)


    filename_record = r"C:\Users\jules\Documents\Technische Wiskunde\Applied quantum algorithms\results" + f"\\record_{time.localtime().tm_year:04d}-{time.localtime().tm_mon:02d}-{time.localtime().tm_mday:02d}_{time.localtime().tm_hour:02d}_{time.localtime().tm_min:02d}.pickle"
    filename_input = r"C:\Users\jules\Documents\Technische Wiskunde\Applied quantum algorithms\results" + f"\\input_{time.localtime().tm_year:04d}-{time.localtime().tm_mon:02d}-{time.localtime().tm_mday:02d}_{time.localtime().tm_hour:02d}_{time.localtime().tm_min:02d}.pickle"
    filename_txt = r"C:\Users\jules\Documents\Technische Wiskunde\Applied quantum algorithms\results" + f"\\info_{time.localtime().tm_year:04d}-{time.localtime().tm_mon:02d}-{time.localtime().tm_mday:02d}_{time.localtime().tm_hour:02d}_{time.localtime().tm_min:02d}.txt"

    # pickle record
    sampleset.to_pandas_dataframe().to_pickle(filename_record)

    # pickle input
    with open(filename_input, 'wb') as f:
        pickle.dump((d, K, M, N), f)

    # write info to file
    with open(filename_txt, 'w') as f:
        for k in sampleset.info:
            f.write(k)
            f.write(": ")
            f.write(str(sampleset.info[k]))
            f.write("\n")


def main_sorting():
    # n = 3
    # vecX = np.array([2,3,1])

    # vecX = np.arange(1,10)
    # np.random.shuffle(vecX)
    # n = len(vecX)

    # vecX = np.array([1, 6, 9, 3, 7, 5, 8, 4, 2])

    vecX = np.array([2, 4, 5, 3])

    penalty_r = 100
    penalty_c = 100
    matR, vecR = sq.initializeSortQUBO(vecX, penalty_r, penalty_c)
    Q_dict = sq.qubo_dict(matR, vecR)

    sampler = EmbeddingComposite(DWaveSampler())

    sampleset = sampler.sample_qubo(Q_dict, num_reads=200, annealing_time = 200, return_embedding=True)

    dir = r"C:\Users\jules\Documents\Technische Wiskunde\Applied quantum algorithms\results_sorting"
    filename_record = dir + f"\\record_{time.localtime().tm_year:04d}-{time.localtime().tm_mon:02d}-{time.localtime().tm_mday:02d}_{time.localtime().tm_hour:02d}_{time.localtime().tm_min:02d}.pickle"
    filename_input = dir + f"\\input_{time.localtime().tm_year:04d}-{time.localtime().tm_mon:02d}-{time.localtime().tm_mday:02d}_{time.localtime().tm_hour:02d}_{time.localtime().tm_min:02d}.pickle"
    filename_txt = dir + f"\\info_{time.localtime().tm_year:04d}-{time.localtime().tm_mon:02d}-{time.localtime().tm_mday:02d}_{time.localtime().tm_hour:02d}_{time.localtime().tm_min:02d}.txt"

    # pickle record
    sampleset.to_pandas_dataframe().to_pickle(filename_record)

    # pickle input
    with open(filename_input, 'wb') as f:
        pickle.dump((vecX, penalty_r, penalty_c), f)

    # write info to file
    with open(filename_txt, 'w') as f:
        for k in sampleset.info:
            f.write(k)
            f.write(": ")
            f.write(str(sampleset.info[k]))
            f.write("\n")

if __name__ == "__main__":
    main_sorting()