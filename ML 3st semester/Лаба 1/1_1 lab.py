A = {18: 0, 20: 0.15, 23: 0.4, 26: 0.7, 28: 0.95, 30: 1, 32: 0.85, 34: 0.5, 36: 0.2, 38: 0}
suppA = {x for x, mu in A.items() if mu > 0}
hA = max(A.values())
A_norm = {x: mu / hA for x, mu in A.items()}
print("A =", A)
print("suppA =", suppA)
print("h(A) =", hA)
print("A_norm =", A_norm)