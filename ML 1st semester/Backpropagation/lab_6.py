import math
import numpy as np

def sigmoid(x):
    """Функция активации сигмоида"""
    return 1 / (1 + math.exp(-x))

def forward_pass(x1, x2, w1, w2, b):
    """Прямое распространение"""
    z = w1 * x1 + w2 * x2 + b
    a = sigmoid(z)
    return z, a

def compute_loss(a, y):
    """Вычисление ошибки"""
    return 0.5 * (a - y) ** 2

def backward_pass(x1, x2, a, y, z):
    """Обратное распространение"""
    dL_da = a - y
    da_dz = a * (1 - a)
    dL_dz = dL_da * da_dz
    
    # Градиенты по весам и смещению
    dL_dw1 = dL_dz * x1
    dL_dw2 = dL_dz * x2
    dL_db = dL_dz
    
    return dL_dw1, dL_dw2, dL_db, dL_dz

def update_weights(w1, w2, b, dL_dw1, dL_dw2, dL_db, learning_rate):
    """Обновление весов"""
    w1_new = w1 - learning_rate * dL_dw1
    w2_new = w2 - learning_rate * dL_dw2
    b_new = b - learning_rate * dL_db
    return w1_new, w2_new, b_new

x1 = 0.4
x2 = 0.6
w1 = 0.2
w2 = 0.3
b = -0.1
y = 1
learning_rate = 0.1

for iteration in range(1, 4):
    
    # Прямое распространение
    z, a = forward_pass(x1, x2, w1, w2, b)
    loss = compute_loss(a, y)
  
    # Обратное распространение
    dL_dw1, dL_dw2, dL_db, dL_dz = backward_pass(x1, x2, a, y, z)

    w1_old, w2_old, b_old = w1, w2, b
    
    # Обновление весов
    w1, w2, b = update_weights(w1, w2, b, dL_dw1, dL_dw2, dL_db, learning_rate)

    # Проверка после обновления
    z_check, a_check = forward_pass(x1, x2, w1, w2, b)
    loss_check = compute_loss(a_check, y)
    
    print(f"Ошибка: {loss:.6f}")

    print(f"\n{'='*50}")
    print(f"ИТЕРАЦИЯ {iteration}")
    print(f"{'='*50}")

print(f"\n{'='*80}")
print(f"w₁ = {w1:.6f}, w₂ = {w2:.6f}, b = {b:.6f}")
z_final, a_final = forward_pass(x1, x2, w1, w2, b)
loss_final = compute_loss(a_final, y)
print(f"Выход сети: a = {a_final:.6f}")
print(f"Ошибка: L = {loss_final:.6f}")
