import numpy as np
import matplotlib.pyplot as plt

def tanh_schedule(k, n, a, b, alpha=3.0):
    if n <= 1:  # 边界情形
        return float(b)
    x = 2.0*(np.asarray(k, dtype=float)-1.0)/(n-1.0) - 1.0
    ta = np.tanh(alpha)
    s = (np.tanh(alpha*x) + ta) / (2.0*ta)
    return a + (b - a) * s

if __name__ == "__main__":
    n = 1100
    k = np.arange(1, n+1)
    a = 1e-6
    b = 1e-4
    alpha = 10.0

    y = tanh_schedule(k, n, a, b, alpha)

    plt.plot(k, y)
    plt.title('Tanh Schedule')
    plt.xlabel('k')
    plt.ylabel('Value')
    plt.grid()
    plt.show()