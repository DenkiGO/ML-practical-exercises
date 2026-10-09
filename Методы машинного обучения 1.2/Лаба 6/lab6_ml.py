import numpy as np

# Данные (A, B, C, D)
points = {
    'A': np.array([1, 1, 5]),
    'B': np.array([1, 2, -1]),
    'C': np.array([2, 3, 2]),
    'D': np.array([1, 2, 3])
}

# Функция Манхэттенского расстояния (L1)
def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

# Считаем суммы расстояний
distance_sums = {}

for key1 in points:
    total = 0
    for key2 in points:
        if key1 != key2:
            dist = manhattan_distance(points[key1], points[key2])
            total += dist
            print(f"Расстояние {key1}-{key2}: {dist}")
    distance_sums[key1] = total
    print(f"Сумма расстояний для {key1}: {total}\n")

# Находим выброс
max_value = max(distance_sums.values())
outliers = [k for k, v in distance_sums.items() if v == max_value]

print("Суммы расстояний:", distance_sums)
print("Выбросы:", outliers)

print('\n\n')

# Дано
Q25 = 2
Q75 = 5

# Шаг 1: межквартильный размах
IQR = Q75 - Q25

# Шаг 2: границы
lower_bound = Q25 - 1.5 * IQR
upper_bound = Q75 + 1.5 * IQR

# Вывод
print("Q25 =", Q25)
print("Q75 =", Q75)
print("IQR =", IQR)
print("Нижняя граница:", lower_bound)
print("Верхняя граница:", upper_bound)

print("\nИнтервалы выбросов:")
print(f"x < {lower_bound} или x > {upper_bound}")