import matplotlib.pyplot as plt
import numpy as np

with open("output.txt", "r") as file:
    data = file.readline().strip()  

measurements = list(map(float, data.split()))

x_positions = np.arange(len(measurements))
x_labels = [1, 2, 4, 8, 16, 24]

plt.figure(figsize=(10, 6))
plt.plot(x_positions, measurements, marker='o', linestyle='-', color='blue')

plt.title("График измерений")
plt.xlabel("Количество фрагментов")
plt.ylabel("Измеренные значения")

plt.xticks(x_positions, x_labels, rotation=45)

plt.grid(True)
plt.tight_layout()
plt.show()