import math

from matplotlib import pyplot as plt
from numpy.linalg import inv
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

f = open("f11.txt")
y = np.array(list(map(float, f.read().split(" "))))


def dft(x):
    n = len(x)
    c = []
    for k in range(n):
        sub_sum = 0 + 0j
        for m in range(n):
            sub_sum += x[m] * np.exp(-1j * 2 * math.pi * k * m / n)
        c.append(sub_sum / n)
    return c


def extremes(func):
    extrs = []
    for i in range(1, len(func) - 1):
        # maximum
        if func[i - 1] < func[i] and func[i] > func[i + 1]:
            extrs.append(i)
        # minimum
        elif func[i - 1] > func[i] and func[i] < func[i + 1]:
            extrs.append(i)
    return extrs


y_f = dft(y)

f = [15, 15.2]
x = [0.01 * i for i in range(len(y))]
X_ = np.empty((len(y), 4 + len(f)))
for k, x_k in enumerate(x):
    X_[k][0] = x_k ** 3
    X_[k][1] = x_k ** 2
    X_[k][2] = x_k
    for i in range(len(f)):
        X_[k][3 + i] = np.sin(2 * np.pi * f[i] * x_k)
    X_[k][3 + len(f)] = 1
X = np.matrix(X_)
X_t = X.transpose()
a = np.array(np.dot(inv(np.dot(X_t, X)), np.array(np.dot(X_t, y)).flatten())).flatten()
print(a)

a[3], a[3 + len(f)] = a[3 + len(f)], a[3]
predicted = np.empty(len(y))
for i in range(len(y)):
    predicted[i] = a[0] * x[i] ** 3 \
                    + a[1] * x[i] ** 2 \
                    + a[2] * x[i] \
                    + a[len(a) - 1]
    for j in range(len(f)):
        predicted[i] += np.sin(2*np.pi*f[j]*x[i]) * a[3 + j]

residuals = y - predicted
print((residuals**2).mean())

extrs = extremes(y_f[0:len(y_f)//2 + len(y_f) % 2])
print(extrs)

print(extrs)

plt.plot(np.array(dft(y)))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("DFT")
plt.step(0.2, 1)
plt.show()
