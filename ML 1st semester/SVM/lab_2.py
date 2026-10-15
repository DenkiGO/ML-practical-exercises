import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def min_distance_between_opposite_points(signal_levels, noise_levels, loss_packets, results):
    """
    Функция вычисляет наименьшее евклидово расстояние между точками противоположных классов
    и выводит параметры этих точек.
    """
    X = np.column_stack((signal_levels, noise_levels, loss_packets))
    y = np.array(results)
    class0_indices = np.where(y == 0)[0]
    class1_indices = np.where(y == 1)[0]
    
    if len(class0_indices) == 0 or len(class1_indices) == 0:
        print("Один из классов пустой.")
        return np.inf
    
    class0 = X[class0_indices]
    class1 = X[class1_indices]
    
    min_dist = np.inf
    closest_pair = None
    closest_indices = None
    
    for i, p1 in enumerate(class1):
        for j, p0 in enumerate(class0):
            dist = np.linalg.norm(p1 - p0)
            if dist < min_dist:
                min_dist = dist
                closest_pair = (p1, p0)
                closest_indices = (class1_indices[i], class0_indices[j])
    
    idx1, idx0 = closest_indices

    S_1 = [signal_levels[idx1], noise_levels[idx1], loss_packets[idx1], 1]
    S_2 = [signal_levels[idx0], noise_levels[idx0], loss_packets[idx0], 1]
    
    return S_1, S_2

def solve_for_a1_a2(S1, S2):
    """
    Решает систему уравнений:
    (S1 · S1) a1 + (S1 · S2) a2 = 1
    (S2 · S1) a1 + (S2 · S2) a2 = -1
    Вычисляет определитель матрицы G = [[S1·S1, S1·S2], [S2·S1, S2·S2]]
    """
    S1 = np.array(S1)
    S2 = np.array(S2)
    
    K11 = np.dot(S1, S1)
    K12 = np.dot(S1, S2)
    K21 = np.dot(S2, S1)
    K22 = np.dot(S2, S2)
    
    G = np.array([[K11, K12], [K21, K22]])
    rhs = np.array([1, -1])
    
    a = np.linalg.solve(G, rhs)
    a1, a2 = a
    
    det_G = np.linalg.det(G)
    
    print(f"Определитель матрицы G: {det_G:.6f}")
    print(f"a1 = {a1:.6f}")
    print(f"a2 = {a2:.6f}")
    
    return a1, a2, det_G

def compute_W(a1, a2, S1, S2):
    """
    Вычисляет вектор W по формуле W = a1 * S1 + a2 * S2
    """
    S1_np = np.array(S1)
    S2_np = np.array(S2)
    W = a1 * S1_np + a2 * S2_np
    print("Вектор W (первые 3 компоненты - веса w, 4-я - смещение b):")
    print(W)
    return W

def plot_hyperplane(ax, W):
    """
    Отображает гиперплоскость w1*x + w2*y + w3*z + b = 0 на 3D графике.
    """
    w1, w2, w3, b = W
    if abs(w3) < 1e-10:
        print("w3 близко к нулю, плоскость параллельна оси Z. Визуализация упрощена.")
        return
    
    # Диапазоны для осей
    x_range = np.linspace(30, 100, 10)
    y_range = np.linspace(80, 100, 10)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (-b - w1 * X - w2 * Y) / w3
    
    # Отображаем плоскость
    ax.plot_surface(X, Y, Z, alpha=0.3, color='green', label='Гиперплоскость')

signal_levels = [35, 45, 65, 75, 55, 85, 80, 50, 90, 60]
noise_levels = [95, 92, 90, 88, 98, 85, 91, 96, 82, 93]
loss_packets = [0, 2, 5, 15, 1, 25, 30, 3, 40, 5]
results = [1, 1, 1, 0, 1, 0, 0, 1, 0, 1]  # 1 = хорошее качество, 0 = плохое качество

S1, S2 = min_distance_between_opposite_points(signal_levels, noise_levels, loss_packets, results)

print("Опорный вектор S1 (класс +1):", S1)
print("Опорный вектор S2 (класс -1):", S2)

a1, a2, det = solve_for_a1_a2(S1, S2)

W = compute_W(a1, a2, S1, S2)

# ==================================================================

colors = ['blue' if r == 1 else 'red' for r in results]
markers = ['^' if r == 1 else 'o' for r in results]

fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(111, projection='3d')

for i in range(len(signal_levels)):
    ax1.scatter(signal_levels[i], noise_levels[i], loss_packets[i], 
                c=colors[i], marker=markers[i], s=100, alpha=0.7, label=f'Point {i+1}')

ax1.set_xlabel('Уровень сигнала')
ax1.set_ylabel('Уровень шума')
ax1.set_zlabel('Потери пакетов')
ax1.set_title('3D вид всех признаков')

handles = [plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Хорошее качество (1)'),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Плохое качество (0)')]
ax1.legend(handles=handles)

plt.tight_layout()

plot_hyperplane(ax1, W)

plt.show()
