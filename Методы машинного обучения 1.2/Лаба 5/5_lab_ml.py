import math

def convolution_2d(input_matrix, kernel, stride=1):
    """
    Выполняет свертку входной матрицы с ядром (без bias)
    input_matrix: 2D список (матрица)
    kernel: 2D список (ядро свертки)
    stride: шаг свертки
    возвращает матрицу результатов свертки
    """
    h_in = len(input_matrix)
    w_in = len(input_matrix[0])
    h_k = len(kernel)
    w_k = len(kernel[0])
    
    h_out = (h_in - h_k) // stride + 1
    w_out = (w_in - w_k) // stride + 1
    
    output = [[0.0 for _ in range(w_out)] for _ in range(h_out)]
    
    for i in range(h_out):
        for j in range(w_out):
            total = 0.0
            for ki in range(h_k):
                for kj in range(w_k):
                    total += input_matrix[i + ki][j + kj] * kernel[ki][kj]
            output[i][j] = total
    
    return output

def add_bias(z, bias):
    """
    Добавляет bias к каждому элементу матрицы z
    """
    result = [[z[i][j] + bias for j in range(len(z[0]))] for i in range(len(z))]
    return result

def relu(z):
    """
    Функция активации ReLU
    """
    return [[max(0, z[i][j]) for j in range(len(z[0]))] for i in range(len(z))]

def relu_derivative(z):
    """
    Производная ReLU: 1 если z > 0, иначе 0
    """
    return [[1 if z[i][j] > 0 else 0 for j in range(len(z[0]))] for i in range(len(z))]

def mse_loss(a, y):
    """
    Среднеквадратичная ошибка: L = 1/2 * sum((a - y)^2)
    """
    total = 0.0
    for i in range(len(a)):
        for j in range(len(a[0])):
            diff = a[i][j] - y[i][j]
            total += diff * diff
    return total / 2

def mse_gradient(a, y):
    """
    Градиент MSE: dL/da = a - y
    """
    result = [[a[i][j] - y[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    return result

def elementwise_multiply(m1, m2):
    """
    Поэлементное умножение двух матриц
    """
    result = [[m1[i][j] * m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]
    return result

def sum_matrix(m):
    """
    Сумма всех элементов матрицы
    """
    total = 0.0
    for row in m:
        total += sum(row)
    return total

def compute_weight_gradients(dL_dz, input_matrix, kernel_size):
    """
    Вычисляет градиенты для весов свертки
    dL_dz: матрица градиентов после ReLU
    input_matrix: исходная входная матрица
    kernel_size: размер ядра (h_k, w_k)
    возвращает матрицу градиентов для весов
    """
    h_out = len(dL_dz)
    w_out = len(dL_dz[0])
    h_k, w_k = kernel_size
    
    weight_grad = [[0.0 for _ in range(w_k)] for _ in range(h_k)]
    
    for i in range(h_out):
        for j in range(w_out):
            for ki in range(h_k):
                for kj in range(w_k):
                    weight_grad[ki][kj] += dL_dz[i][j] * input_matrix[i + ki][j + kj]
    
    return weight_grad

def update_parameters(W, b, dW, db, alpha):
    """
    Обновление параметров по градиентному спуску
    """
    W_new = [[W[i][j] - alpha * dW[i][j] for j in range(len(W[0]))] for i in range(len(W))]
    b_new = b - alpha * db
    return W_new, b_new

def print_matrix(name, m):
    """
    Вывод матрицы в консоль
    """
    print(f"{name}:")
    for row in m:
        print([round(x, 4) for x in row])
    print()

def train_step(input_matrix, W, b, y, alpha):
    """
    Один шаг обучения сверточной сети
    
    input_matrix: входная матрица 3x3
    W: ядро свертки 2x2
    b: bias
    y: целевые значения 2x2
    alpha: скорость обучения
    
    возвращает обновленные W, b и значение ошибки
    """
    
    print("=" * 50)
    print("ШАГ 1: Свертка")
    Z_conv = convolution_2d(input_matrix, W)
    print_matrix("Z_conv", Z_conv)
    
    print("ШАГ 2: Добавление bias")
    Z = add_bias(Z_conv, b)
    print_matrix("Z", Z)
    
    print("ШАГ 3: ReLU")
    a = relu(Z)
    print_matrix("a", a)
    
    print("ШАГ 4 (Обратное распространение):")
    
    # Вычисление ошибки
    L = mse_loss(a, y)
    print(f"L (MSE) = {L:.4f}")
    
    # Градиент по a
    dL_da = mse_gradient(a, y)
    print_matrix("dL/da", dL_da)
    
    # Производная ReLU
    dReLU = relu_derivative(Z)
    print_matrix("da/dz", dReLU)
    
    # Градиент по z
    dL_dz = elementwise_multiply(dL_da, dReLU)
    print_matrix("dL/dz", dL_dz)
    
    # Градиент по bias
    db = sum_matrix(dL_dz)
    print(f"dL/db = {db:.4f}")
    
    # Градиенты по весам
    dW = compute_weight_gradients(dL_dz, input_matrix, (len(W), len(W[0])))
    print_matrix("dL/dW", dW)
    
    # Обновление параметров
    W_new, b_new = update_parameters(W, b, dW, db, alpha)
    print_matrix("W_new", W_new)
    print(f"b_new = {b_new:.4f}")
    
    return W_new, b_new, L

# ============ Исходные данные ============

# Входная матрица 3x3
X = [
    [2, 1, 0],
    [1, 2, 1],
    [0, 1, 2]
]

# Начальные веса свертки 2x2
W = [
    [0.4, -0.4],
    [0.3, -0.2]
]

# Начальный bias
b = 0.7

# Целевые значения (выход)
y = [
    [0.8, 0.8],
    [0.8, 0.8]
]

# Скорость обучения
alpha = 0.1

print("========== ИСХОДНЫЕ ДАННЫЕ ==========")
print_matrix("X (вход)", X)
print_matrix("W (начальные веса)", W)
print(f"b = {b}")
print_matrix("y (целевые значения)", y)
print(f"alpha = {alpha}")
print()

# ============ Обучение ============

print("\n" + "=" * 50)
print("1-я ИТЕРАЦИЯ ОБУЧЕНИЯ")
print("=" * 50)

W, b, L = train_step(X, W, b, y, alpha)
print(f"\nРезультат 1-й итерации: L = {L:.4f}")

print("\n" + "=" * 50)
print("2-я ИТЕРАЦИЯ ОБУЧЕНИЯ")
print("=" * 50)
W, b, L = train_step(X, W, b, y, alpha)
print(f"\nРезультат 2-й итерации: L = {L:.4f}")

print("\n" + "=" * 50)
print("3-я ИТЕРАЦИЯ ОБУЧЕНИЯ")
print("=" * 50)
W, b, L = train_step(X, W, b, y, alpha)
print(f"\nРезультат 3-й итерации: L = {L:.4f}")

print("\n" + "=" * 50)
print("4-я ИТЕРАЦИЯ ОБУЧЕНИЯ")
print("=" * 50)
W, b, L = train_step(X, W, b, y, alpha)
print(f"\nРезультат 4-й итерации: L = {L:.4f}")
