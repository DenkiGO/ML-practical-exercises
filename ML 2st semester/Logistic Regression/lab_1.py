import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y, p):
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)
    return - (y * np.log(p) + (1 - y) * np.log(1 - p))

def predict(X, w0, w1):
    z = w0 + w1 * X
    return sigmoid(z)

def binary_cross_entropy_from_z(y, z):
    return np.log1p(np.exp(z)) - y * z

# Данные
X = np.array([50, 100, 20, 80, 300])
y = np.array([0, 1, 0, 1, 0])

# Начальные веса
w0, w1 = 0, 0.2
learning_rate = 0.01

# Цикличный проход по 5 примерам
i = -1
for epoch in range(1000):
    # Увеличиваем индекс и сбрасываем при достижении конца
    total_loss = 0

    i += 1
    if i >= len(X):
        i = 0
    
    # Прямой проход
    z = w0 + w1 * X[i]
    p = sigmoid(z)
    
    for j in range(len(X)):
        z1 = w0 + w1 * X[j]
        p1 = sigmoid(z1)
        loss = binary_cross_entropy(y[j], p1)
        total_loss += loss

    # Градиенты
    dL_dp = p - y[i]
    dp_dz = p * (1 - p)
    dL_dz = dL_dp * dp_dz
    
    # Обновление весов
    w0 = w0 - learning_rate * dL_dz
    w1 = w1 - learning_rate * dL_dz * X[i]

    avg_loss = total_loss / len(X)

    if avg_loss < 0.9:
        print(f"Эпоха {epoch+1}, пример {i}: X={X[i]}, y={y[i]}, p={p:.4f}, w0={w0:.6f}, w1={w1:.6f}, avg_loss={avg_loss}")
        break
    
    print(f"Эпоха {epoch+1}, пример {i}: X={X[i]}, y={y[i]}, p={p:.4f}, w0={w0:.6f}, w1={w1:.6f}, avg_loss={avg_loss}")

# Предсказание для письма длиной 90 слов
email_length = 90
probability = predict(email_length, w0, w1)

print(f"\nВероятность отклика на email длиной {email_length} слов: {probability:.4f}")