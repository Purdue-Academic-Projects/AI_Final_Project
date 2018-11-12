import numpy as np
from timeit import default_timer as timer
from numba import vectorize


@vectorize(['float32(float32, float32)'], target='cuda')
def gpu_pow(a, b):
    return a ** b


@vectorize(['float32(float32, float32)'], target='parallel')
def cpu_para_pow(a, b):
    return a ** b


def cpu_pow(a, b, c):
    for i in range(a.size):
         c[i] = a[i] ** b[i]


def cpu_test():
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)

    start = timer()
    cpu_pow(a, b, c)
    duration = timer() - start

    print(duration)


def cpu_para_test():
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)

    start = timer()
    c = cpu_para_pow(a, b)
    duration = timer() - start

    print(duration)


def gpu_test():
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)

    start = timer()
    c = gpu_pow(a, b)
    duration = timer() - start

    print(duration)


def main():
    cpu_para_test()
    cpu_test()
    gpu_test()


if __name__ == '__main__':
    main()
