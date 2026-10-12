x_points = [-1, 0, 1]
y_points = [4, 1, 0]

x_target = 0.5

def lagrange_interpolation(x_points, y_points, x):
    n = len(x_points)
    result = 0.0
    for i in range(n):
        L_i = 1.0
        for j in range(n):
            if i != j:
                L_i *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += y_points[i] * L_i
    return result

result_lagrange = lagrange_interpolation(x_points, y_points, x_target)
print(f"Метод Лагранжа: {result_lagrange}")

def divided_differences(x_points, y_points):
    """
    Вычисляет таблицу разделенных разностей
    """
    n = len(x_points)
    # Создаем таблицу разделенных разностей
    table = [[0] * n for _ in range(n)]
    
    # Первый столбец - значения y
    for i in range(n):
        table[i][0] = y_points[i]
    
    # Вычисляем разделенные разности
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x_points[i+j] - x_points[i])
    
    return table

def newton_interpolation(x_points, y_points, x):
    n = len(x_points)
    table = divided_differences(x_points, y_points)
    
    # Первая строка таблицы содержит коэффициенты многочлена Ньютона
    coefficients = table[0]
    
    # Вычисляем значение многочлена Ньютона
    result = coefficients[0]
    product = 1.0
    
    for i in range(1, n):
        product *= (x - x_points[i-1])
        result += coefficients[i] * product
    
    return result

result_newton = newton_interpolation(x_points, y_points, x_target)

print("\nТаблица разделенных разностей:")
table = divided_differences(x_points, y_points)
n = len(x_points)
for i in range(n):
    print(f"x{i} = {x_points[i]:2}: ", end="")
    for j in range(n - i):
        print(f"{table[i][j]:8.3f}", end=" ")
    print()

print(f"Метод Ньютона:  {result_newton}")
