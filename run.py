import numpy as np


N = 100
REPETITIONS = 10000


def run_experiment():
    r = np.random.randint(2, size=(N, 1))
    A = np.random.randint(2, size=(N, N))
    B = np.random.randint(2, size=(N, N))

    # in order to consider the worst case, we only change first element of A * B
    C = np.matmul(A, B)
    C[0, 0] += 1
    C = np.remainder(C, 2)

    ABr = np.remainder(np.matmul(A, np.matmul(B, r)), 2)
    Cr = np.remainder(np.matmul(C, r), 2)
    return np.allclose(ABr, Cr)


def run():
    false_positives = 0
    for i in range(REPETITIONS):
        if run_experiment():
            false_positives += 1
    print(false_positives / REPETITIONS)


if __name__ == '__main__':
    run()
