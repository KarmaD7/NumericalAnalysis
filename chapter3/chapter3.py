# %% [markdown]
# # 数值分析 实验3
# 2019011265 计93 丁韶峰

# %% [markdown]
# ## 实验内容
# 
# 考虑 Hilbert 矩阵 $H_n$ 以及全1向量 $x$，用 Cholesky 分解的方法求解 $H_n x = b$，并计算残差和误差。施加扰动，观察残差和误差的变化情况。在不同的$n$下重复实验。

# %% [markdown]
# ## 实验过程
# 
# 引入必要的包：

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# 根据伪代码，实现Cholesky分解。

# %%
def Cholesky(A, n):
  L = np.zeros_like(A)

  for j in range(n):
    L[j][j] = A[j][j]
    for k in range(j):
      L[j][j] -= L[j][k] ** 2
    L[j][j] = np.sqrt(L[j][j])
    for i in range(j + 1, n):
      L[i][j] = A[i][j]
      for k in range(j):
        L[i][j] -= L[i][k] * L[j][k]
      L[i][j] /= L[j][j]

  return L

# %% [markdown]
# 实现解方程$L^{T}Lx = b$的函数：

# %%
def solve(L, b, n):
  # Ly = b
  y = np.zeros_like(b)
  for i in range(n):
    y[i] = b[i]
    for j in range(0, i):
      y[i] -= L[i][j] * y[j]
    y[i] /= L[i][i]

  # LTx = y
  x = np.zeros_like(b)
  for i in reversed(range(n)):
    x[i] = y[i]
    for j in reversed((range(i + 1, n))):
      x[i] -= L[j][i] * x[j]
    x[i] /= L[i][i]
  return x

# %% [markdown]
# 生成矩阵，计算条件数来判断病态性。先在原数据下进行计算，再给一个正态分布的扰动进行计算。比较残差和误差。

# %%
def compute(n):
  H = np.fromfunction(lambda i, j : 1 / (i + j + 1), (n, n))
  ones = np.ones(n)
  print("cond is {}".format(np.linalg.cond(H)))
  b = np.dot(H, ones)
  L = Cholesky(H, n)
  x = solve(L, b, n)
  r = np.max(np.abs(b - np.dot(H, x)))
  delta = np.max(np.abs(ones - x))
  print("no disturbance, r is {}, delta is {}".format(r, delta))
  x = solve(L, b + np.random.normal(0, 1e-7, n), n)
  r = np.max(np.abs(b - np.dot(H, x)))
  delta = np.max(np.abs(ones - x))
  print("with disturbance, r is {}, delta is {}".format(r, delta))

# %% [markdown]
# 在不同的 $n$ 下进行实验，可得结果。

# %%
n_list = [8, 10, 12]
for n in n_list:
  compute(n)

# %% [markdown]
# 观察可知，施加扰动前后，对于不同的$n$，残差都比较小，对于准确的$b$，计算结果还是比较精确的。
# 另一方面，矩阵的条件数很大，病态性很强，且随$n$的增大条件数越来越大。对$b$的轻微扰动会带来非常非常大的误差，且$n$越大误差就越大。

# %% [markdown]
# ## 实验结论
# 
# 对于一些病态性很强的矩阵，对$b$的轻微扰动就会产生很大的误差，且就 Hilbert 矩阵而言，$n$越大误差就越大，用计算机得到这些方程的比较准确的解是相当困难的。


