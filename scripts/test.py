import numpy as np
import matplotlib.pyplot as plt
import math

def tanh_ramp(epoch, total_epochs=1100, a=1e-6, b=1e-4, alpha=3.0, inflect_at=0.3):
    """
    """
    if total_epochs <= 1:
        return float(b)
    e = max(0, min(int(epoch), total_epochs - 1))
    # 将 epoch 映射到 x∈[-1,1]
    x = 2.0 * e / (total_epochs - 1) - 1.0
    # 把希望的拐点位置转换到 x 轴：x=delta 时是拐点
    delta = 2.0 * inflect_at - 1.0

    # 做移位后的 tanh，并用端点重新归一化到 [0,1]
    s_raw  = math.tanh(alpha * (x - delta))
    s_min  = math.tanh(alpha * (-1.0 - delta))  # x=-1
    s_max  = math.tanh(alpha * ( 1.0 - delta))  # x=+1
    s = (s_raw - s_min) / (s_max - s_min)

    return a + (b - a) * s

if __name__ == "__main__":
    y = []
    n = 1100
    for k in range(n-1):
        y.append(tanh_ramp(k+1, n, 0.01, 0.1, alpha=3.0))

    plt.plot(y)
    plt.title('Tanh Schedule')
    plt.xlabel('k')
    plt.ylabel('Value')
    plt.grid()
    plt.show()