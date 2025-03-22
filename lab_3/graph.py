import matplotlib.pyplot as plt
import numpy as np

# Исходные данные
data = """
2000: 34.7221 17.5626 11.0486 6.1721 4.5503
1000: 1.8524 0.9398 0.5239 0.4178 0.1826
500: 0.2407 0.1236 0.0612 0.0333 0.0194
1500: 11.8879 5.9589 3.3788 1.8135 1.2117
2500: 44.9142 22.6378 14.2196 7.9745 5.7297
"""

# Параметры процессов (T=1,2,4,8,16)
processes = np.array([1, 2, 4, 8, 16])

# Парсинг данных
results = {}
for line in data.strip().split('\n'):
    filename, values = line.split(':')
    results[filename.strip()] = list(map(float, values.strip().split()))

# Настройка графика
plt.figure(figsize=(12, 7))
plt.title("Зависимость времени выполнения от количества процессов", fontsize=14)
plt.xlabel("Количество процессов", fontsize=12)
plt.ylabel("Время выполнения (сек)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xscale('log', base=2)
plt.xticks(processes, labels=processes)

# Построение графиков для каждого файла
markers = ['o', 's', '^', 'D', 'v']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

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

# Сохранение и отображение
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()