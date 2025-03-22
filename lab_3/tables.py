import pandas as pd

# Исходные данные
data = """
2000: 34.7221 17.5626 11.0486 6.1721 4.5503
1000: 1.8524 0.9398 0.5239 0.4178 0.1826
500: 0.2407 0.1236 0.0612 0.0333 0.0194
1500: 11.8879 5.9589 3.3788 1.8135 1.2117
2500: 44.9142 22.6378 14.2196 7.9745 5.7297
"""

# Параметры процессов
processes = [1, 2, 4, 8, 16]

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
    
    return df

# Обработка данных
for line in data.strip().split('\n'):
    try:
        name, values = line.split(':')
        name = name.strip()
        times = list(map(float, values.strip().split()))
        
        # Создаем и выводим таблицу
        df = calculate_metrics(name, times)
        print(f"\n{name}:")
        print(df.to_string(index=False, float_format='%.4f'))
        
    except Exception as e:
        print(f"Ошибка обработки строки '{line}': {str(e)}")