import numpy as np

def fuzzy_c_means(X, c, m=2, epsilon=0.01, max_iter=100):
    n = len(X)
    
    U = np.array([
        [0.70, 0.20, 0.10],
        [0.60, 0.30, 0.10],
        [0.40, 0.40, 0.20],
        [0.30, 0.50, 0.20],
        [0.10, 0.60, 0.30],
        [0.10, 0.50, 0.40],
        [0.05, 0.35, 0.60],
        [0.05, 0.25, 0.70],
        [0.02, 0.18, 0.80],
        [0.02, 0.08, 0.90]
    ])
    
    centers = None
    
    for iteration in range(max_iter):
        # Расчет центроидов
        Um = U ** m
        centers = (Um.T @ X) / Um.sum(axis=0, keepdims=True).T
        centers = centers.flatten()
        
        # Расчет расстояний
        distances = np.zeros((n, c))
        for j in range(c):
            distances[:, j] = np.abs(X - centers[j])
        
        # Обновление матрицы принадлежностей
        U_new = np.zeros((n, c))
        for i in range(n):
            zero_indices = np.where(distances[i] == 0)[0]
            if len(zero_indices) > 0:
                U_new[i, zero_indices[0]] = 1
            else:
                for j in range(c):
                    denominator = sum((distances[i][j] / distances[i][k]) ** 2 for k in range(c))
                    U_new[i, j] = 1 / denominator
        
        # Проверка остановки
        if np.max(np.abs(U_new - U)) < epsilon:
            print(f"Сошлось на итерации {iteration + 1}")
            return U_new, centers, iteration + 1
        
        U = U_new
    
    return U, centers, max_iter

X = np.array([1, 2, 4, 5, 7, 8, 10, 11, 13, 14])
c = 3
m = 2
epsilon = 0.01

print("FUZZY C-MEANS КЛАСТЕРИЗАЦИЯ")
print("="*50)
print(f"Точки: {X}")
print(f"Кластеры: {c}, m={m}, ε={epsilon}\n")

U, centers, iterations = fuzzy_c_means(X, c, m, epsilon)

print("\nЦЕНТРОИДЫ:")
for j, center in enumerate(centers):
    print(f"  c{j+1} = {center:.4f}")

print("\nМАТРИЦА ПРИНАДЛЕЖНОСТЕЙ:")
print("  Точка  |  Кл1  |  Кл2  |  Кл3")
print("  " + "-"*30)
for i in range(len(X)):
    print(f"  x{i+1}={X[i]:2d}  | {U[i][0]:.4f} | {U[i][1]:.4f} | {U[i][2]:.4f}")

print("\nРАСПРЕДЕЛЕНИЕ ПО КЛАСТЕРАМ:")
clusters = [[] for _ in range(c)]
for i in range(len(X)):
    clusters[np.argmax(U[i])].append(X[i])
for j in range(c):
    print(f"  Кластер {j+1} (центр={centers[j]:.2f}): {clusters[j]}")

