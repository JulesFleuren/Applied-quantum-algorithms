import numpy as np

# number of traces
M = 4

# length of traces
N = 100


def test1(M, N):
    
    # number of shifts
    K = 6

    # test array of traces
    d = np.zeros((M,N))
    d[0, 10] = 1
    d[1, 12] = 1
    d[2, 13] = 1
    d[3, 14] = 1
    d[0, 11] = 2
    d[1, 13] = 2
    d[2, 14] = 2
    d[3, 15] = 2
    d[0, 12] = 1
    d[1, 14] = 1
    d[2, 15] = 1
    d[3, 16] = 1
    return d, K

def test2(M, N):
    
    # number of shifts
    K = 10

    # test array of traces
    x = np.linspace(0, 8, N)
    d = np.zeros((M,N))
    np.random.seed(seed=1313)
    shift = np.random.randint(0, K, M-1)
    d[0,:] = np.exp(-(x-4)**2)*np.sin(8*(x))
    for i in range(1,M):
        d[i,:] = np.pad(d[0,:], (0,K-1), 'constant', constant_values=(0,0))[shift[i-1]:shift[i-1]+N]

    return d, K
