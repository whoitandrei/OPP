import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
df = pd.read_csv("distributions_stats.csv")

# Убедимся, что всё как надо
assert "distribution" in df.columns
assert "rank" in df.columns
assert "completed_weight" in df.columns
assert "initial_weight" in df.columns

for dist in df['distribution'].unique():
    sub = df[df['distribution'] == dist].sort_values("rank")

    ranks = sub["rank"].to_numpy()
    initial = sub["initial_weight"].to_numpy()
    completed = sub["completed_weight"].to_numpy()

    x = np.arange(len(ranks))  # позиции по оси X
    width = 0.35               # ширина столбца

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, initial, width, label="Initial Weight", color="steelblue")
    plt.bar(x + width/2, completed, width, label="Completed Weight", color="orange")

    plt.title(f"Weight per Process – {dist}")
    plt.xlabel("Process Rank")
    plt.ylabel("Weight")
    plt.xticks(x, ranks)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"weight_comparison_{dist.lower()}.png")
    plt.close()

print("✔️ Диаграммы сохранены как weight_comparison_<strategy>.png")
