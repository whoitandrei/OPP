import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Чтение данных из файла output.txt
with open('output.txt', 'r') as file:
    lines = file.readlines()
    time_v1 = list(map(float, lines[0].strip().split()))  # Время для варианта 1
    time_v2 = list(map(float, lines[1].strip().split()))  # Время для варианта 2

# Определение количества процессов
processes = [1, 2, 4, 8, 16]

# Проверка длин массивов
if len(time_v1) != len(processes) or len(time_v2) != len(processes):
    raise ValueError("Количество значений времени должно соответствовать количеству процессов (5)")

# Вычисление ускорения и эффективности
def calculate_metrics(times):
    speedup = [times[0] / t for t in times]  # Ускорение: T1 / Tp
    efficiency = [s / p * 100 for s, p in zip(speedup, processes)]  # Эффективность: S / p * 100%
    return speedup, efficiency

speedup_v1, efficiency_v1 = calculate_metrics(time_v1)
speedup_v2, efficiency_v2 = calculate_metrics(time_v2)

# Создание таблиц с использованием pandas
data_v1 = {
    'Processes': processes,
    'Time (s)': time_v1,
    'Speedup': speedup_v1,
    'Efficiency (%)': efficiency_v1
}
df_v1 = pd.DataFrame(data_v1)

data_v2 = {
    'Processes': processes,
    'Time (s)': time_v2,
    'Speedup': speedup_v2,
    'Efficiency (%)': efficiency_v2
}
df_v2 = pd.DataFrame(data_v2)

# Вывод таблиц
print("Таблица для варианта 1:")
print(df_v1.to_string(index=False))
print("\nТаблица для варианта 2:")
print(df_v2.to_string(index=False))

# Построение графиков
plt.figure(figsize=(12, 12))

# Индексы для равномерного распределения на оси X
x_indices = np.arange(len(processes))

# 1. График зависимости времени от количества процессов
plt.subplot(3, 1, 1)
plt.plot(x_indices, time_v1, marker='o', label='Вариант 1')
plt.plot(x_indices, time_v2, marker='o', label='Вариант 2')
plt.title('Зависимость времени от количества процессов')
plt.xlabel('Количество процессов')
plt.ylabel('Время (с)')
plt.grid(True)
plt.xticks(x_indices, processes)  # Устанавливаем метки как [1, 2, 4, 8, 16] на равных интервалах
plt.legend()

# 2. График зависимости ускорения от количества процессов
plt.subplot(3, 1, 2)
plt.plot(x_indices, speedup_v1, marker='o', label='Вариант 1')
plt.plot(x_indices, speedup_v2, marker='o', label='Вариант 2')
plt.title('Зависимость ускорения от количества процессов')
plt.xlabel('Количество процессов')
plt.ylabel('Ускорение')
plt.grid(True)
plt.xticks(x_indices, processes)  # Устанавливаем метки как [1, 2, 4, 8, 16] на равных интервалах
plt.legend()

# 3. График зависимости эффективности от количества процессов
plt.subplot(3, 1, 3)
plt.plot(x_indices, efficiency_v1, marker='o', label='Вариант 1')
plt.plot(x_indices, efficiency_v2, marker='o', label='Вариант 2')
plt.title('Зависимость эффективности от количества процессов')
plt.xlabel('Количество процессов')
plt.ylabel('Эффективность (%)')
plt.grid(True)
plt.xticks(x_indices, processes)  # Устанавливаем метки как [1, 2, 4, 8, 16] на равных интервалах
plt.legend()

plt.tight_layout()
plt.show()