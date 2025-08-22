import numpy as np
import matplotlib.pyplot as plt
import math

def tanh_ramp(epoch, total_epochs, a, b, alpha=3.0):
    """
    """
    if total_epochs <= 1:
        return float(b)
    e = max(0, min(int(epoch), total_epochs - 1))
    x = 2.0 * e / (total_epochs - 1) - 1.0
    ta = math.tanh(alpha)
    s = (math.tanh(alpha * x) + ta) / (2.0 * ta)
    return a + (b - a) * s

if __name__ == "__main__":
    y = []
    n = 1100
    for k in range(n-1):
        y.append(tanh_ramp(k+1, n, 0.02, 0.2, alpha=3.0))

    plt.plot(y)
    plt.title('Tanh Schedule')
    plt.xlabel('k')
    plt.ylabel('Value')
    plt.grid()
    plt.show()