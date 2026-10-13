import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normal_equations_method(x1, x2, y):
    """
    Линейная регрессия
    """
    n = len(y)
    sum_x1_sq = np.sum(x1**2)
    sum_x2_sq = np.sum(x2**2)
    sum_x1_x2 = np.sum(x1 * x2)
    sum_x1 = np.sum(x1)
    sum_x2 = np.sum(x2)
    sum_ones = n
    sum_x1_y = np.sum(x1 * y)
    sum_x2_y = np.sum(x2 * y)
    sum_y = np.sum(y)
    A = np.array([
        [2 * sum_x1_sq, 2 * sum_x1_x2, 2 * sum_x1],
        [2 * sum_x1_x2, 2 * sum_x2_sq, 2 * sum_x2],
        [2 * sum_x1, 2 * sum_x2, 2 * sum_ones]
    ])
    b = np.array([2 * sum_x1_y, 2 * sum_x2_y, 2 * sum_y])
    w = np.linalg.solve(A, b) # Решение системы A w = b, где w = [w1, w2, w0]
    return w, A, b

y = np.array([110, 120, 150, 160, 200])
x1 = np.array([50, 55, 70, 75, 90])
x2 = np.array([60, 66, 78, 85, 100])

w, A, b_vec = normal_equations_method(x1, x2, y)

print("\nЛинейная регрессия:")
print("Матрица A:")
print(A)
print("\nВектор b:")
print(b_vec)
print("\nВектор коэффициентов w = [w1, w2, w0]:")
print(w)
print("\nМодель: y ≈ {:.4f} * x1 + {:.4f} * x2 + {:.4f}".format(w[0], w[1], w[2]))

def predict(x1, x2, w):
    return w[0]*x1 + w[1]*x2 + w[2]

x1_range = np.linspace(45, 95, 20)
x2_range = np.linspace(55, 105, 20)
X1, X2 = np.meshgrid(x1_range, x2_range)

Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i,j] = predict(X1[i,j], X2[i,j], w)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X1, X2, Z, alpha=0.7, cmap='viridis', 
                      linewidth=0, antialiased=True, label='Модель')

scatter = ax.scatter(x1, x2, y, 
                    color='red', s=100, label='Исходные данные')

ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_zlabel('y', fontsize=12)
ax.set_title('Линейная регрессия'.format(w[0], w[1], w[2]), fontsize=12)
ax.legend()

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Исходные данные
data = {
    'x1': [50, 55, 70, 75, 90],
    'x2': [60, 66, 78, 85, 100],
    'y': [110, 120, 150, 160, 200]
}

df = pd.DataFrame(data)

X = np.column_stack([
    df['x1'],                    # w1 x1
    df['x2'],                    # w2 x2
    df['x1']**2,                 # w3 (x1^2)
    df['x2']**2,                 # w4 (x2^2)
    df['x1'] * df['x2'],         # w5 (x1*x2)
    np.ones(len(df))             # w0
])

y = df['y'].values

A = np.zeros((6, 6))
b = np.zeros(6)

for i in range(6):
    for j in range(6):
        A[i, j] = np.sum(X[:, i] * X[:, j])
    b[i] = np.sum(X[:, i] * y)

print("Матрица коэффициентов A:")
print(A)
print(f"\nВектор правой части b:")
print(b)

w = np.linalg.solve(A, b)
print(f"\nРешение системы:")
print(f"w₁ = {w[0]:.4f}")
print(f"w₂ = {w[1]:.4f}")
print(f"w₃ = {w[2]:.4f}")
print(f"w₄ = {w[3]:.4f}")
print(f"w₅ = {w[4]:.4f}")
print(f"w₀ = {w[5]:.4f}")

print(f"\nМодель:")
print(f"y = {w[0]:.4f}·x₁ + {w[1]:.4f}·x₂ + {w[2]:.4f}·x₁² + {w[3]:.4f}·x₂² + {w[4]:.4f}·x₁x₂ + {w[5]:.4f}")

def predict(x1, x2, w):
    return w[0]*x1 + w[1]*x2 + w[2]*(x1**2) + w[3]*(x2**2) + w[4]*(x1*x2) + w[5]

x1_range = np.linspace(45, 95, 20)
x2_range = np.linspace(55, 105, 20)
X1, X2 = np.meshgrid(x1_range, x2_range)

Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i,j] = predict(X1[i,j], X2[i,j], w)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X1, X2, Z, alpha=0.7, cmap='viridis', 
                      linewidth=0, antialiased=True, label='Модель')

scatter = ax.scatter(df['x1'], df['x2'], df['y'], 
                    color='red', s=100, label='Исходные данные')

ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_zlabel('y', fontsize=12)
ax.set_title('Полиномиальная регрессия', fontsize=10)
ax.legend()

plt.tight_layout()
plt.show()
