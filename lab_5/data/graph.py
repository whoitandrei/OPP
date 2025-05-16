import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Загрузка данных из CSV
df = pd.read_csv('scaling_results.csv')  # Поменяй на имя твоего файла

# Убедимся, что данные отсортированы по числу процессов
df = df.sort_values('np')

processes = df['np'].to_numpy()
times = df['time_sec'].to_numpy()

# Вычисляем метрики
speedup = times[0] / times
efficiency = (speedup / processes) * 100

# Создаём DataFrame с метриками для вывода
metrics_df = pd.DataFrame({
    'Processes': processes,
    'Time (s)': times,
    'Speedup': np.round(speedup, 2),
    'Efficiency (%)': np.round(efficiency, 1)
})

print(metrics_df.to_string(index=False, float_format='%.4f'))

# --- Построение графиков ---

# Настройки стиля
plt.rcParams.update({'font.size': 12})

def plot_with_zero_base(x, y, title, xlabel, ylabel, filename, color):
    """Универсальная функция построения графиков с осью Y от 0"""
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log', base=2)
    plt.xticks(x, labels=x)
    
    # Устанавливаем нижнюю границу оси Y в 0
    y_min = 0
    y_max = max(y) * 1.1  # Добавляем 10% запаса сверху
    plt.ylim(y_min, y_max)
    
    plt.plot(x, y, marker='o', color=color, linestyle='-', linewidth=2, markersize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# График времени выполнения
plot_with_zero_base(
    processes, times,
    "Время выполнения в зависимости от числа процессов",
    "Количество процессов",
    "Время (сек)",
    'performance_time.png',
    'blue'
)

# График ускорения
plot_with_zero_base(
    processes, speedup,
    "Ускорение в зависимости от числа процессов",
    "Количество процессов",
    "Ускорение",
    'performance_speedup.png',
    'green'
)

# График эффективности (здесь уже есть ограничение 0-105%)
plot_with_zero_base(
    processes, efficiency,
    "Эффективность в зависимости от числа процессов",
    "Количество процессов",
    "Эффективность (%)",
    'performance_efficiency.png',
    'red'
)