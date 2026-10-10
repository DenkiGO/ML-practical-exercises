from tabulate import tabulate
import math

signal_levels = [35, 45, 65, 75, 55, 85, 80, 50, 90, 60]
noise_levels = [95, 92, 90, 88, 98, 85, 91, 96, 82, 93]
loss_packets = [0, 2, 5, 15, 1, 25, 30, 3, 40, 8]
results = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]

test = [60, 85, 20]

table_data = []
for i in range(10):
    table_data.append([
        i + 1,
        signal_levels[i],
        noise_levels[i],
        loss_packets[i],
        results[i]
    ])

headers = ["№", "Уровень сигнала", "Уровень шума", "Потеря пакетов", "Итог"]

print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))

def function_manheton(data, dataset):
   num_failed_login = data[0]
   protocol = data[1]
   duration_second = data[2]

   resul = []

   for i in range(len(dataset[0])):
    res = 0
    for j in range(len(data)):
      res += abs(data[j] - dataset[j][i])
    resul.append(res)

   ranking = [sorted(range(len(resul)), key=lambda i: abs(resul[i])).index(i) + 1 for i in range(len(resul))]

   return resul, ranking

def function_evklida(data, dataset):
   num_failed_login = data[0]
   protocol = data[1]
   duration_second = data[2]

   resul = []

   for i in range(len(dataset[0])):
    res = 0
    for j in range(len(data)):
      res += (data[j] - dataset[j][i])**2
    resul.append(math.sqrt(res))

   ranking = [sorted(range(len(resul)), key=lambda i: abs(resul[i])).index(i) + 1 for i in range(len(resul))]

   return resul, ranking

def function_chepusheva(data, dataset):
   num_failed_login = data[0]
   protocol = data[1]
   duration_second = data[2]

   resul = []

   for i in range(len(dataset[0])):
    res = []
    for j in range(len(data)):
      res.append(abs(data[j] - dataset[j][i]))
    resul.append(max(res))

   ranking = [sorted(range(len(resul)), key=lambda i: abs(resul[i])).index(i) + 1 for i in range(len(resul))]
   return resul, ranking

func_menh = function_manheton(test, [signal_levels, noise_levels, loss_packets])

table_data = []
for i in range(10):
    table_data.append([
        i + 1,
        signal_levels[i],
        noise_levels[i],
        loss_packets[i],
        results[i],
        func_menh[0][i],
        func_menh[1][i]
    ])

for i in range(len(func_menh[1])):
   if func_menh[1][i] == 1:
      res = i
      break

table_data.append([
    11,
    test[0],
    test[1],
    test[2],
    results[res]
])

headers = ["№", "Уровень сигнала", "Уровень шума", "Потеря пакетов", "Итог", "D", "Rank"]
print("\n\nМанхэттен\n")
print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))

func_evklid = function_evklida(test, [signal_levels, noise_levels, loss_packets])

table_data = []
for i in range(10):
    table_data.append([
        i + 1,
        signal_levels[i],
        noise_levels[i],
        loss_packets[i],
        results[i],
        func_evklid[0][i],
        func_evklid[1][i]
    ])

for i in range(len(func_evklid[1])):
   if func_evklid[1][i] == 1:
      res = i
      break

table_data.append([
    11,
    test[0],
    test[1],
    test[2],
    results[res]
])

headers = ["№", "Уровень сигнала", "Уровень шума", "Потеря пакетов", "Итог", "D", "Rank"]
print("\n\nЕвклида\n")
print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))

func_chepush = function_chepusheva(test, [signal_levels, noise_levels, loss_packets])

table_data = []
for i in range(10):
    table_data.append([
        i + 1,
        signal_levels[i],
        noise_levels[i],
        loss_packets[i],
        results[i],
        func_chepush[0][i],
        func_chepush[1][i]
    ])

for i in range(len(func_chepush[1])):
   if func_chepush[1][i] == 1:
      res = i
      break

table_data.append([
    11,
    test[0],
    test[1],
    test[2],
    results[res]
])

headers = ["№", "Уровень сигнала", "Уровень шума", "Потеря пакетов", "Итог", "D", "Rank"]
print("\n\nЧепушева\n")
print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))
