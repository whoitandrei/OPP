import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Исходные данные
data = """
2500 : 36.1394 18.2855 10.2251 6.7740 3.4840
2000 : 20.5696 10.3121 5.8513 3.9393 2.0575
1500 : 8.5090 4.3476 2.5114 1.6602 0.8703
1000 : 2.2138 1.1339 0.6225 0.4193 0.2143
"""

# Параметры процессов
processes = np.array([1, 2, 4, 8, 16])

# Парсинг данных
results = {}
for line in data.strip().split('\n'):
    filename, values = line.split(':')
    results[filename.strip()] = list(map(float, values.strip().split()))

# Функция для вычисления метрик
def calculate_metrics(name, times):
    """Вычисление метрик и создание таблицы"""
    if len(times) != len(processes):
        raise ValueError(f"Неверное количество значений для {name}")

    # Рассчитываем метрики
    speedup = [times[0]/t for t in times]
    efficiency = [(s/p)*100 for s, p in zip(speedup, processes)]
    
    # Создаем DataFrame
    df = pd.DataFrame({
        'Processes': processes,
        'Time (s)': times,
        'Speedup': [round(s, 2) for s in speedup],
        'Efficiency (%)': [round(e, 1) for e in efficiency]
    })
    
    return df, speedup, efficiency

# Подготовка данных для графиков
metrics = {}
for name, times in results.items():
    df, speedup, efficiency = calculate_metrics(name, times)
    metrics[name] = {'df': df, 'speedup': speedup, 'efficiency': efficiency}
    
    # Вывод таблицы
    print(f"\n{name}:")
    print(df.to_string(index=False, float_format='%.4f'))

# Настройка стиля графиков
markers = ['o', 's', '^', 'D', 'v']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# График времени выполнения
plt.figure(figsize=(12, 7))
plt.title("Зависимость времени выполнения от количества процессов", fontsize=14)
plt.xlabel("Количество процессов", fontsize=12)
plt.ylabel("Время (сек)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xscale('log', base=2)
plt.xticks(processes, labels=processes)

for (name, values), marker, color in zip(results.items(), markers, colors):
    plt.plot(processes, values, 
             marker=marker, 
             linestyle='-', 
             linewidth=2,
             markersize=8,
             color=color,
             label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('performance_time.png', dpi=300, bbox_inches='tight')
plt.close()

# График ускорения
plt.figure(figsize=(12, 7))
plt.title("Зависимость ускорения от количества процессов", fontsize=14)
plt.xlabel("Количество процессов", fontsize=12)
plt.ylabel("Ускорение", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xscale('log', base=2)
plt.xticks(processes, labels=processes)

for (name, data), marker, color in zip(metrics.items(), markers, colors):
    plt.plot(processes, data['speedup'], 
             marker=marker, 
             linestyle='-', 
             linewidth=2,
             markersize=8,
             color=color,
             label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('performance_speedup.png', dpi=300, bbox_inches='tight')
plt.close()

# График эффективности
plt.figure(figsize=(12, 7))
plt.title("Зависимость эффективности от количества процессов", fontsize=14)
plt.xlabel("Количество процессов", fontsize=12)
plt.ylabel("Эффективность (%)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xscale('log', base=2)
plt.xticks(processes, labels=processes)

for (name, data), marker, color in zip(metrics.items(), markers, colors):
    plt.plot(processes, data['efficiency'], 
             marker=marker, 
             linestyle='-', 
             linewidth=2,
             markersize=8,
             color=color,
             label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('performance_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()