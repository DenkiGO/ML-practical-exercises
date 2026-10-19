import numpy as np

a1=np.array([-1,1,1,1,-1,-1,1,1,1])
a2=np.array([-1,-1,-1,1,-1,-1,1,1,-1])
a3=np.array([1,1,-1,1,-1,-1,1,1,-1])
a4=np.array([1,1,-1,1,-1,1,1,-1,1])
b=np.array([1,-1,-1,1,-1,-1,1,1,-1])

patterns = [a1, a2, a3, a4]
names = ["a1", "a2", "a3", "a4"]

np.set_printoptions(linewidth=120)

def sign_vec(v):
    """sgn(x): если x>=0 -> +1, иначе -1"""
    return np.where(v >= 0, 1, -1)

def as_grid(v):
    """Печать как 3x3"""
    return v.reshape(3, 3)

print("=== ОБУЧАЮЩИЕ ОБРАЗЫ (3x3) ===")
for nm, p in zip(names, patterns):
    print(f"{nm} = {p}")
    print(as_grid(p), "\n")

print("=== ПОВРЕЖДЁННЫЙ ОБРАЗ b (3x3) ===")
print("b =", b)
print(as_grid(b))
print()

# ====== ШАГ 1. СТРОИМ W = sum( a a^T ) ======
W = np.zeros((9, 9), dtype=int)

print("=== ШАГ 1: Внешние произведения a^p (a^p)^T и суммирование ===")
for nm, p in zip(names, patterns):
    outer = np.outer(p, p)
    print(f"\n--- {nm} * {nm}^T (outer product) ---")
    print(outer)
    W = W + outer
    print(f"\nСумма W после добавления {nm}:")
    print(W)

print("\n=== ШАГ 2: Зануляем диагональ (w_ii = 0) ===")
W_hop = W.copy()
np.fill_diagonal(W_hop, 0)
print(W_hop)

# ====== ШАГ 3. ОДИН ПАКЕТНЫЙ ШАГ: net = W*b, x1 = sign(net) ======
print("\n=== ШАГ 3: Один шаг восстановления (синхронно) ===")
net0 = W_hop @ b
print("net = W * b =")
print(net0)

x1 = sign_vec(net0)
print("\nx^(1) = sign(net) =")
print(x1)
print("x^(1) как 3x3:")
print(as_grid(x1))

# ====== ШАГ 4. Итерации до сходимости (синхронно) ======
print("\n=== ШАГ 4: Итерации до сходимости (синхронное обновление) ===")
state = b.copy()
for t in range(1, 11):
    net = W_hop @ state
    new_state = sign_vec(net)
    print(f"\nИтерация {t}:")
    print("Текущее состояние x =")
    print(state)
    print("net = W*x =")
    print(net)
    print("sign(net) =")
    print(new_state)
    print("sign(net) как 3x3:")
    print(as_grid(new_state))

    if np.array_equal(new_state, state):
        print("\n>>> Сходимость: состояние не изменилось.")
        break
    state = new_state

print("\n=== ИТОГ (ХОПФИЛД) ===")
print("Восстановленный образ =", state)
print(as_grid(state))

# ====== ШАГ 5. Определяем, какой образ восстановлен ======
print("\n=== ШАГ 5: Сравнение с обучающими образами ===")
matched = False
for i, p in enumerate(patterns, 1):
    eq = np.array_equal(state, p)
    print(f"Совпадает с a{i}? -> {eq}")
    if eq:
        print(f">>> Повреждено изображение a{i}")
        matched = True

if not matched:
    print(">>> Точного совпадения с a1..a4 нет (возможен ложный аттрактор).")
