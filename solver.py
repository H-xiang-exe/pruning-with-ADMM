import numpy as np
import torch


class AdmmSolver():
    # 打印损失变化值
    import matplotlib.pyplot as plt

    plt.plot(loss_list)
    plt.title('training loss')
    plt.xlabel('epochs')
    plt.ylabel('lose')
    plt.show()
