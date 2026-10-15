import math

# Обучающая выборка
data = [
    {"CPU": 0.85, "RAM": 0.80, "Network": 0.90, "class": "Высокая"},
    {"CPU": 0.90, "RAM": 0.85, "Network": 0.75, "class": "Высокая"},
    {"CPU": 0.50, "RAM": 0.55, "Network": 0.45, "class": "Средняя"},
    {"CPU": 0.45, "RAM": 0.50, "Network": 0.55, "class": "Средняя"},
    {"CPU": 0.15, "RAM": 0.20, "Network": 0.10, "class": "Низкая"},
]

# Новый объект
X = {"CPU": 0.55, "RAM": 0.60, "Network": 0.50}

# Параметры
K = 4
m = 2
eps = 0.01

def euclidean_distance(x1, x2):
    return math.sqrt(
        (x1["CPU"] - x2["CPU"]) ** 2 +
        (x1["RAM"] - x2["RAM"]) ** 2 +
        (x1["Network"] - x2["Network"]) ** 2
    )

distances = []
for i, point in enumerate(data):
    dist = euclidean_distance(X, point)
    distances.append((dist, point["class"], i))

distances.sort(key=lambda x: x[0])

nearest = distances[:K]

weights = []
total_weight = 0
for dist, cls, idx in nearest:
    d_sq = dist ** 2
    w = 1 / (d_sq + eps)
    weights.append((w, cls))
    total_weight += w

class_weights = {}
for w, cls in weights:
    if cls not in class_weights:
        class_weights[cls] = 0
    class_weights[cls] += w

membership = {}
for cls in class_weights:
    membership[cls] = class_weights[cls] / total_weight

max_class = max(membership, key=membership.get)
max_confidence = membership[max_class] * 100

print(f"Результат Fuzzy K-NN: {max_class} с уверенностью {max_confidence:.2f}%")
print(f"Принадлежности: {membership}")