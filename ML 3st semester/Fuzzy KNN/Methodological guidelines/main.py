import math


def euclidean_distance(x, y):
    """
    Евклидово расстояние между двумя векторами признаков
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


def manhattan_distance(x, y):
    """
    Манхэттенское расстояние (сумма модулей разностей)
    """
    return sum(abs(a - b) for a, b in zip(x, y))


def make_sample(name, features, label, classes):
    """
    Создает объект обучающей выборки с четкой меткой класса:
    степень принадлежности своему классу равна 1, остальным - 0
    """
    u = {c: (1.0 if c == label else 0.0) for c in classes}
    return {"name": name, "features": features, "u": u}


def find_neighbors(train, query, K, metric=euclidean_distance):
    """
    Возвращает K ближайших соседей в виде списка пар (расстояние, объект),
    отсортированного по возрастанию расстояния
    """
    distances = [(metric(query, s["features"]), s) for s in train]
    distances.sort(key=lambda pair: pair[0])
    return distances[:K]


def compute_weights(neighbors, m=2, eps=0.01):
    """
    Веса соседей: w_i = 1 / (d_i^(2/(m-1)) + eps)
    При m = 2 показатель степени равен 2, то есть w_i = 1 / (d_i^2 + eps)
    """
    p = 2 / (m - 1)
    return [1 / (d ** p + eps) for d, _ in neighbors]


def fuzzy_membership(neighbors, weights, classes):
    """
    Степень принадлежности объекта к каждому классу:
    u_j(x) = sum(u_ij * w_i) / sum(w_i)
    """
    total = sum(weights)
    result = {}
    for c in classes:
        numerator = sum(s["u"][c] * w for (_, s), w in zip(neighbors, weights))
        result[c] = numerator / total
    return result


def fuzzy_knn(train, query, K, classes, m=2, eps=0.01,
              metric=euclidean_distance, verbose=True):
    """
    Классификация объекта методом Fuzzy K-NN.
    Возвращает словарь степеней принадлежности и название класса-победителя
    """
    if verbose:
        print(f"Запрос: {query}, K = {K}, m = {m}, eps = {eps}")
        print("Расстояния до объектов обучающей выборки:")
        for s in train:
            print(f"  {s['name']}: d = {metric(query, s['features']):.4f}")

    neighbors = find_neighbors(train, query, K, metric)
    weights = compute_weights(neighbors, m, eps)

    if verbose:
        print(f"Ближайшие соседи (K = {K}) и их веса:")
        for (d, s), w in zip(neighbors, weights):
            print(f"  {s['name']}: d = {d:.4f}, w = {w:.4f}")
        print(f"Сумма весов = {sum(weights):.4f}")

    membership = fuzzy_membership(neighbors, weights, classes)
    winner = max(membership, key=membership.get)

    if verbose:
        print("Степени принадлежности:")
        for c in classes:
            print(f"  {c}: {membership[c]:.4f}")
        print(f"Результат: {winner} (уверенность {membership[winner] * 100:.2f}%)")
        print()
    return membership, winner


def keller_init(train, K, classes, metric=euclidean_distance):
    """
    Инициализация степеней принадлежности обучающих объектов по Келлеру.
    Для каждого объекта берутся K ближайших соседей (без самого объекта):
    собственный класс: u = 0.51 + 0.49 * (n_j / K)
    чужие классы:      u = 0.49 * (n_j / K)
    """
    labels = {s["name"]: max(s["u"], key=s["u"].get) for s in train}
    result = []
    for s in train:
        others = [t for t in train if t is not s]
        nearest = find_neighbors(others, s["features"], K, metric)
        counts = {c: 0 for c in classes}
        for _, t in nearest:
            counts[labels[t["name"]]] += 1
        own = labels[s["name"]]
        u = {}
        for c in classes:
            share = 0.49 * counts[c] / K
            u[c] = 0.51 + share if c == own else share
        result.append({"name": s["name"], "features": s["features"], "u": u})
    return result


if __name__ == "__main__":
    # ===== Пример №1: классификация сетевого соединения =====
    cls1 = ["Нормальное", "Подозрительное"]
    train1 = [
        make_sample("A", (0.20, 0.20), "Нормальное", cls1),
        make_sample("B", (0.40, 0.30), "Нормальное", cls1),
        make_sample("C", (0.30, 0.50), "Нормальное", cls1),
        make_sample("D", (0.70, 0.60), "Подозрительное", cls1),
        make_sample("E", (0.90, 0.80), "Подозрительное", cls1),
        make_sample("F", (0.80, 0.50), "Подозрительное", cls1),
    ]
    print("=" * 50)
    print("ПРИМЕР №1")
    print("=" * 50)
    fuzzy_knn(train1, (0.50, 0.40), K=3, classes=cls1, m=2)

    # ===== Пример №2: оценка уровня угрозы =====
    cls2 = ["Низкий", "Средний", "Высокий"]
    raw2 = [
        ("A", (0.10, 0.05, 0.10), "Низкий"),
        ("B", (0.20, 0.15, 0.20), "Низкий"),
        ("C", (0.35, 0.30, 0.40), "Средний"),
        ("D", (0.45, 0.50, 0.35), "Средний"),
        ("E", (0.55, 0.45, 0.55), "Средний"),
        ("F", (0.70, 0.75, 0.65), "Высокий"),
        ("G", (0.90, 0.85, 0.95), "Высокий"),
        ("H", (0.90, 0.95, 0.80), "Высокий"),
    ]
    train2 = [make_sample(n, f, c, cls2) for n, f, c in raw2]
    print("=" * 50)
    print("ПРИМЕР №2")
    print("=" * 50)
    fuzzy_knn(train2, (0.50, 0.55, 0.50), K=5, classes=cls2, m=2)

    # ===== Пример №3: приоритет инцидента (нечеткие метки, m = 3) =====
    cls3 = ["Критический", "Значимый", "Незначительный"]
    raw3 = [
        ("P1", (0.63, 0.54), (0.7, 0.3, 0.0)),
        ("P2", (0.48, 0.66), (0.2, 0.6, 0.2)),
        ("P3", (0.75, 0.70), (0.9, 0.1, 0.0)),
        ("P4", (0.51, 0.62), (0.4, 0.5, 0.1)),
        ("P5", (0.30, 0.10), (0.0, 0.1, 0.9)),
    ]
    train3 = [
        {"name": n, "features": f, "u": dict(zip(cls3, u))} for n, f, u in raw3
    ]
    print("=" * 50)
    print("ПРИМЕР №3")
    print("=" * 50)
    fuzzy_knn(train3, (0.60, 0.50), K=4, classes=cls3, m=3)

    # ===== Инициализация по Келлеру для обучающей выборки примера №1 =====
    print("=" * 50)
    print("ИНИЦИАЛИЗАЦИЯ ПО КЕЛЛЕРУ (выборка примера №1, K = 3)")
    print("=" * 50)
    for s in keller_init(train1, 3, cls1):
        vals = ", ".join(f"{c}: {v:.2f}" for c, v in s["u"].items())
        print(f"  {s['name']}: {vals}")
