import numpy as np

R1 = np.array([
    [0.8, 0.3, 0.5],
    [0.4, 0.9, 0.2],
    [0.6, 0.1, 0.7]
])

R2 = np.array([
    [0.5, 0.7, 0.2],
    [0.6, 0.4, 0.8],
    [0.3, 0.9, 0.5]
])

def max_min(A, B):
    n, m = A.shape[0], B.shape[1]
    result = np.zeros((n, m))
    for i in range(n):
        for k in range(m):
            result[i, k] = max(min(A[i, j], B[j, k]) for j in range(A.shape[1]))
    return result

def min_max(A, B):
    n, m = A.shape[0], B.shape[1]
    result = np.zeros((n, m))
    for i in range(n):
        for k in range(m):
            result[i, k] = min(max(A[i, j], B[j, k]) for j in range(A.shape[1]))
    return result

def max_product(A, B):
    n, m = A.shape[0], B.shape[1]
    result = np.zeros((n, m))
    for i in range(n):
        for k in range(m):
            result[i, k] = max(A[i, j] * B[j, k] for j in range(A.shape[1]))
    return result

def max_avg_min_prod(A, B):
    n, m = A.shape[0], B.shape[1]
    result = np.zeros((n, m))
    for i in range(n):
        for k in range(m):
            result[i, k] = max(
                (min(A[i, j], B[j, k]) + A[i, j] * B[j, k]) / 2
                for j in range(A.shape[1])
            )
    return result

def intersection(A, B):
    return np.minimum(A, B)

def union(A, B):
    return np.maximum(A, B)

def complement(A):
    return 1 - A

def inverse(A):
    return A.T

print("Max-min:")
print(max_min(R1, R2))
print()
print("Min-max:")
print(min_max(R1, R2))
print()
print("Max-product:")
print(max_product(R1, R2))
print()
print("Max-avg(min, prod):")
print(max_avg_min_prod(R1, R2))
print()
print("Пересечение:")
print(intersection(R1, R2))
print()
print("Объединение:")
print(union(R1, R2))
print()
print("Дополнение R1:")
print(complement(R1))
print()
print("Обратное отношение R1:")
print(inverse(R1))
