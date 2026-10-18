import math

points = {
    "A": (4, 2),
    "B": (3, 2),
    "C": (1, -1),
    "D": (1, 1),
    "E": (1, 4),
}

centroids = {
    1: points["A"],  # C1
    2: points["D"],  # C2
}

def euclid(p, c):
    """Евклидово расстояние между точкой p=(x,y) и центром c=(x,y)."""
    return math.sqrt((p[0] - c[0])**2 + (p[1] - c[1])**2)

def mean_point(items):
    """Среднее арифметическое координат списка точек [(x,y), ...]."""
    x = sum(p[0] for p in items) / len(items)
    y = sum(p[1] for p in items) / len(items)
    return (x, y)

max_iters = 20
assignments = {name: None for name in points}

for it in range(1, max_iters + 1):
    print(f"\n=== Итерация {it} ===")
    print(f"C1 = {centroids[1]}, C2 = {centroids[2]}")
    print("Точка | d(C1)  | d(C2)  | Кластер")
    print("-----------------------------------")

    changed = False
    new_assignments = {}

    # 1) Назначение кластеров
    for name, p in points.items():
        d1 = euclid(p, centroids[1])
        d2 = euclid(p, centroids[2])
        cluster = 1 if d1 <= d2 else 2  # при равенстве -> кластер 1

        new_assignments[name] = cluster
        if assignments[name] != cluster:
            changed = True

        print(f"{name:>5} | {d1:5.2f} | {d2:5.2f} | {cluster}")

    assignments = new_assignments

    # 2) Пересчёт центров
    clusters = {1: [], 2: []}
    for name, cl in assignments.items():
        clusters[cl].append(points[name])

    new_centroids = {}
    for cl in [1, 2]:
        new_centroids[cl] = mean_point(clusters[cl])

    print("\nНовые центры:")
    print(f"C1 = ({new_centroids[1][0]:.2f}, {new_centroids[1][1]:.2f})")
    print(f"C2 = ({new_centroids[2][0]:.2f}, {new_centroids[2][1]:.2f})")

    # Проверка сходимости (центры не меняются)
    if (abs(new_centroids[1][0] - centroids[1][0]) < 1e-9 and
        abs(new_centroids[1][1] - centroids[1][1]) < 1e-9 and
        abs(new_centroids[2][0] - centroids[2][0]) < 1e-9 and
        abs(new_centroids[2][1] - centroids[2][1]) < 1e-9):
        print("\nАлгоритм сошёлся (центры не изменились).")
        break

    centroids = new_centroids
print("\n=== Итог ===")
print("Кластеры:")
for cl in [1, 2]:
    members = [name for name, c in assignments.items() if c == cl]
    print(f"Кластер {cl}: {members}")

print("\nФинальные центры:")
print(f"C1 = ({centroids[1][0]:.2f}, {centroids[1][1]:.2f})")
print(f"C2 = ({centroids[2][0]:.2f}, {centroids[2][1]:.2f})")

# Код задания №1:
objects = ["A", "B", "C", "D", "E"]

dist = {
    ("A","B"): 3, ("A","C"): 4, ("A","D"): 3, ("A","E"): 5,
    ("B","C"): 4, ("B","D"): 2, ("B","E"): 5,
    ("C","D"): 3, ("C","E"): 3,
    ("D","E"): 3,
}

R = 2

graph = {obj: [] for obj in objects}

for (a, b), d in dist.items():
    if d <= R:
        graph[a].append(b)
        graph[b].append(a)

visited = set()
clusters = []

def dfs(node, cluster):
    visited.add(node)
    cluster.append(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, cluster)

for obj in objects:
    if obj not in visited:
        cluster = []
        dfs(obj, cluster)
        clusters.append(cluster)

print("\nКластеры:")
for i, cluster in enumerate(clusters, 1):
    print(f"Кластер {i}: {cluster}")
