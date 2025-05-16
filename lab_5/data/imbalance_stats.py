import pandas as pd
import matplotlib.pyplot as plt

# Читаем данные
df = pd.read_csv('imbalance.csv')

# Сортируем по возрастанию дисбаланса (по желанию)
df = df.sort_values('imbalance')

# Создаём диаграмму
plt.figure(figsize=(10, 6))
plt.bar(df['distribution'], df['imbalance'], color='teal')

plt.title('Load Imbalance by Distribution Strategy')
plt.xlabel('Distribution Strategy')
plt.ylabel('Load Imbalance (max_time / min_time)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Подписи над столбцами с точным значением
for i, val in enumerate(df['imbalance']):
    plt.text(i, val + 0.05, f"{val:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig('load_imbalance.png', dpi=300)
plt.show()
