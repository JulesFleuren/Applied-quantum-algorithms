import numpy as np
import matplotlib.pyplot as plt
from pyqubo import Binary
import neal
from sample_problems import test1, test2





def coefficient_matrices(d, K):
    # Qubo using one-hot encoding
    # x_i_j = 1 if trace i and shift j are selected
    # A is the linear terms, Q is the quadratic terms, C is the constant
    # index [i*K+j] corresponds to x_i_j


    # number of traces
    M = d.shape[0]

    # length of traces
    N = d.shape[1]


    # pad with zeros
    d = np.pad(d, ((0,0), (K-1,K-1)), 'constant', constant_values=(0,0))
    # array of inner products between d_i(t_a) and d_j(t_b) (or in this case d_i and d_j(t_(a)))
    inner_products_matrix = np.zeros((M,M,2*K-1))
    for i in range(M):
        for j in range(M):
            for a in range(-K+1, K):
                    inner_products_matrix[i,j,a-K+1] = np.dot(d[i,K-1:K-1+N], d[j,a+K-1:a+K-1+N])

    def inner_product(i, j, a, b):
        if a < 0 or a >= K or b < 0 or b >= K:
            raise ValueError('a and b must be between 0 and K-1')
        return inner_products_matrix[i,j,b-a-K+1]


    # penalty factor
    p = 100
    # matrix of coefficients of linear terms (A[i*K+j] corresponds to x_ij)
    A = np.zeros(M * K)
    # matrix of coefficients of quadratic terms
    Q = np.zeros((M*K, M*K))
    # constant term
    C = 0


    # Constraint using penalty term: Each trace must be selected once (last part of equation 5.9)
    for i in range(M):
        for a in range(K):
            for b in range(a+1, K):
                Q[i*K+a, i*K+b] += -2*p

    for i in range(M):
        for a in range(K):
            A[i*K+a] += p

    C += -p*M

    # objective
    for i in range(M):
        for j in range(i+1, M):
            for a in range(K):
                for b in range(K):
                    Q[i*K+a, j*K+b] += 2*inner_product(i,j,a,b)
    return d, K, M, N, (A, Q, C)


def qubo_dict(d, K, M, N, A, Q, C):
# Binary variables
    bin_variables = []
    Q_dict = dict()
    for i in range(M):
        for j in range(K):
            bin_variables.append(f'x_{i}_{j}')



    # Hamiltonian
    # linear terms
    for i in range(M*K):
        Q_dict[(bin_variables[i], bin_variables[i])] = -A[i]


    # quadratic terms
    for i in range(M*K):
        for j in range(M*K):
            Q_dict[(bin_variables[i], bin_variables[j])] = -Q[i,j]

    return Q_dict


def solve_dimod(d, K, M, N, A, Q, C):
    # Binary variables
    bin_variables = []
    for i in range(M):
        for j in range(K):
            bin_variables.append(Binary(f'x_{i}_{j}'))

    # Hamiltonian
    H = -C
    # linear terms
    for i in range(M*K):
        H -= A[i] * bin_variables[i]

    # quadratic terms
    for i in range(M*K):
        for j in range(M*K):
            H -= Q[i,j] * bin_variables[i] * bin_variables[j]
    
    # solve QUBO by dimod sampler
    model = H.compile()
    bqm = model.to_bqm()
    sa = neal.SimulatedAnnealingSampler()
    sampleset = sa.sample(bqm, num_reads=1000)
    decoded_samples = model.decode_sampleset(sampleset)
    best_sample = min(decoded_samples, key=lambda x: x.energy)
    print(best_sample.sample)

    d_shifted = shift_traces(d, K, M, N, best_sample.sample)
    
    return d_shifted
    
def shift_traces(d, K, M, N, sample):
    d_shifted = np.copy(d)
    shifts = np.zeros(M)
    for i in range(M):
        for j in range(K):
            if sample[f'x_{i}_{j}'] == 1:
                print(f'For trace {i}, shift {j} is selected')
                shifts[i] = j
                d_shifted[i,K-1:K-1+N] = d[i,j+K-1:j+K-1+N]
                break
    
    return d_shifted

def plot_results(d, d_shifted, K, M, N):    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.transpose(np.repeat(d[:,K-1:K-1+N], repeats=4, axis=0)), cmap='Greys')
    ax2.imshow(np.transpose(np.repeat(d_shifted[:,K-1:K-1+N], repeats=4, axis=0)), cmap='Greys')
    plt.show()


def main():
    # number of traces
    M = 8

    # length of traces
    N = 100

    d, K = test2(M,N)
    d, K, M, N, matrices = coefficient_matrices(d, K)
    # d_shifted = solve_dimod(d, K, M, N, *matrices)
    plot_results(d, d, K, M, N)

if __name__ == "__main__":
    main()