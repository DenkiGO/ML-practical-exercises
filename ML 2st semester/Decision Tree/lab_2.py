data = [
    ("A", 1, 1, 1, 0),
    ("B", 0, 0, 1, 1),
    ("C", 1, 1, 0, 0),
    ("D", 1, 1, 0, 1),
]

IDX = {"Obj": 0, "P1": 1, "P2": 2, "P3": 3, "Y": 4}

def gini_for_feature(feature_name):
    """Считает Gini"""
    n = len(data)
    f_i = IDX[feature_name]
    y_i = IDX["Y"]

    # 1) собираем группы по значениям признака (0/1)
    groups = {}
    for row in data:
        v = row[f_i]
        if v not in groups:
            groups[v] = []
        groups[v].append(row)

    total = 0.0

    # 2) считаем вклад каждой ветки: P(P=v) * P0|v * P1|v
    # выводим в порядке v=0, потом v=1 (если есть)
    for v in sorted(groups.keys()):
        rows = groups[v]
        m = len(rows)
        p_v = m / n

        # считаем P(Y=0|v) и P(Y=1|v)
        cnt0 = 0
        cnt1 = 0
        for r in rows:
            if r[y_i] == 0:
                cnt0 += 1
            else:
                cnt1 += 1

        p0 = cnt0 / m
        p1 = cnt1 / m

        contrib = p_v * p0 * p1
        total += contrib

        objs = [r[IDX["Obj"]] for r in rows]
    return total

# ---- Запуск: считаем для P1, P2, P3 ----
g1 = gini_for_feature("P1")
g2 = gini_for_feature("P2")
g3 = gini_for_feature("P3")

best = "P1"
best_val = g1
if g2 < best_val:
    best, best_val = "P2", g2
if g3 < best_val:
    best, best_val = "P3", g3
print("Итог:")
print(" Gini(P1) =", g1)
print(" Gini(P2) =", g2)
print(" Gini(P3) =", g3)
print("Лучший признак (минимальный Gini):", best, "=", best_val)
