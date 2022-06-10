import numpy as np
import matplotlib.pyplot as plt
from pyqubo import Binary
import neal


def initializeSortQUBO(vecX, lr = None, lc = None):
    n = len(vecX)

    vecN = np.arange(n) + 1

    matI = np.eye(n)
    vec1 = np.ones(n)

    matN = np.kron(matI, vecN)
    matCr = np.kron(vec1, matI)
    matCc = np.kron(matI, vec1)
    if lr is None or lc is None :
        VecX = vecX / np.sum(vecX)
        lr = lc = n
    matR = lr * matCr .T @ matCr + lc * matCc .T @ matCc
    vecR = vecX @ matN + 2 * vec1 @ ( lr * matCr + lc * matCc )
    return matR, vecR

def qubo_dict(matR, vecR):

    n = len(vecR)

    # Binary variables
    bin_variables = []
    Q_dict = dict()
    for i in range(n):	
        bin_variables.append(f'x_{i}')





    # Hamiltonian
    # quadratic terms
    for i in range(n):
        for j in range(n):
            Q_dict[(bin_variables[i], bin_variables[j])] = matR[i,j]
    
    # linear terms
    for i in range(n):
        Q_dict[(bin_variables[i], bin_variables[i])] += -vecR[i]

    return Q_dict



def solve_dimod(matR, vecR):
    n = len(vecR)

    # Binary variables
    bin_variables = []
    for i in range(n):	
        bin_variables.append(Binary(f'x_{i}'))  



    # Hamiltonian
    H = 0
    # linear terms
    for i in range(n):
        H += -vecR[i] *bin_variables[i]


    # quadratic terms
    for i in range(n):
        for j in range(n):
            H += matR[i,j] * bin_variables[i] * bin_variables[j]

    # solve QUBO by dimod sampler
    model = H.compile()
    bqm = model.to_bqm()
    sa = neal.SimulatedAnnealingSampler()
    sampleset = sa.sample(bqm, num_reads=1000)
    decoded_samples = model.decode_sampleset(sampleset)
    best_sample = min(decoded_samples, key=lambda x: x.energy)
    return best_sample.sample


def energy(vecZ, matR, vecR):
    return vecZ @ matR @ vecZ - vecR @ vecZ


if __name__ == "__main__":
    n = 8
    vecX = np.random.randint(0, 20, n)
    print(vecX)
    matR, vecR = initializeSortQUBO(vecX, 80, 80)

    sample = solve_dimod(matR, vecR)
    P = np.zeros(len(sample))
    print(sample)
    for i in range(len(sample)):
        P[i] = sample[f'x_{i}']

    print(P.reshape(n,n).transpose())
    print(P.reshape(n,n).transpose() @ vecX)


