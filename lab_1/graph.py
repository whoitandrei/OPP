# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

with open("measurments.txt", "r") as file:
    lines = [line.strip() for line in file.readlines()]

x_labels = [1, 2, 4, 8, 16, 24]
x_positions = np.arange(len(x_labels))

plt.figure(figsize=(10, 6))

for i, line in enumerate(lines):
    measurements = list(map(float, line.split()))
    plt.plot(
        x_positions,
        measurements,
        marker='o',
        linestyle='-',
        label='Program {}'.format(i+1),  # Изменено на англоязычную метку
        color=['blue', 'orange'][i]
    )

plt.title("Performance Comparison")
plt.xlabel("Number of Threads")
plt.ylabel("Time (seconds)")
plt.xticks(x_positions, x_labels, rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()