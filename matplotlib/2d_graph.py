import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)  # ランダム値の固定をしたい場合は有効にする
N = 50
x = np.random.rand(N)  #50個のランダムな数値
y = np.random.rand(N)

plt.scatter(x, y, alpha=0.5)  #alphaは透過率
plt.show()
