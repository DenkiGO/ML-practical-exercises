import numpy as np

dist = np.array([
    [0, 5, 6, 8],
    [5, 0, 7, 2],
    [6, 7, 0, 4],
    [8, 2, 4, 0]
])
alpha, beta, rho, tau0, ants = 1, 2, 0.5, 1, 4
def solve_aco(dist, iterations=2):
    n = len(dist)
    pheromones = np.full((n, n), tau0, dtype=float)
    for it in range(iterations):
        all_paths = []
        for start_node in range(ants):
            path = [start_node]
            visited = {start_node}
            while len(visited) < n:
                i = path[-1]
                # Считаем числитель для каждого города: tau^alpha * (1/d)^beta
                scores = []
                for j in range(n):
                    if j not in visited:
                        score = (pheromones[i,j]**alpha) * ((1.0/dist[i,j])**beta)
                        scores.append((j, score))
                # Выбираем город с макс. значением (как в примере)
                next_node = max(scores, key=lambda x: x[1])[0]
                path.append(next_node)
                visited.add(next_node)
            path.append(start_node)
            d = sum(dist[path[k], path[k+1]] for k in range(n))
            all_paths.append((path, d))
        # Обновление феромона
        pheromones *= (1 - rho)
        for path, d in all_paths:
            for k in range(n):
                pheromones[path[k], path[k+1]] += 1.0/d
                pheromones[path[k+1], path[k]] += 1.0/d
        print(f"Итерация {it+1}, Лучшая длина: {min(p[1] for p in all_paths)}")
solve_aco(dist)
