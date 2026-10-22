import math

def run_adaboost_iterations():
    data = {
        1: [1, -1, -1, -1],
        2: [-1, 1, -1, 1],
        3: [-1, -1, 1, -1],
        4: [-1, 1, 1, 1],
        5: [1, -1, 1, 1],
        6: [1, 1, -1, 1],
        7: [1, 1, 1, 1]
    }
    
    n_objects = len(data)
    n_classifiers = 3
    
    u = [1/n_objects] * n_objects
    
    final_classifier_weights = []
    
    print("--- Начало алгоритма AdaBoost ---")
    print(f"Начальные веса объектов: {[round(x, 3) for x in u]}\n")

    for iter_num in range(1, 4):
        print(f"Итерация {iter_num}:")
        
        c_idx = iter_num - 1
        
        bad = 0
        for i in range(n_objects):
            obj_data = data[i+1]
            c_val = obj_data[c_idx]
            y_val = obj_data[3]
            
            if c_val != y_val:
                bad += u[i]
        
        print(f"  Ошибка классификатора C{iter_num} (Bad): {round(bad, 3)}")
        
        if bad > 0:
            w = 0.5 * math.log((1 - bad) / bad)
        else:
            w = 1.0
            
        final_classifier_weights.append(w)
        print(f"  Вес классификатора w_{iter_num}: {round(w, 3)}")
        
        new_u = []
        for i in range(n_objects):
            obj_data = data[i+1]
            c_val = obj_data[c_idx]
            y_val = obj_data[3]
            
            val = u[i] * math.exp(-w * y_val * c_val)
            new_u.append(val)
            
        sum_u = sum(new_u)
        u = [x / sum_u for x in new_u]
        
        print(f"  Веса объектов после нормировки: {[round(x, 3) for x in u]}\n")

    print("--- Финальная классификация ---")
    w1, w2, w3 = final_classifier_weights
    
    for i in range(1, n_objects + 1):
        c1, c2, c3, y_true = data[i]
        # Итоговое выражение: w1*C1 + w2*C2 + w3*C3
        score = w1 * c1 + w2 * c2 + w3 * c3
        y_pred = 1 if score > 0 else -1
        
        status = "ВЕРНО" if y_pred == y_true else "ОШИБКА"
        print(f"Объект ({i}): Score = {round(score, 3)} -> y = {y_pred} (Истина: {y_true}) [{status}]")

if __name__ == "__main__":
    run_adaboost_iterations()

