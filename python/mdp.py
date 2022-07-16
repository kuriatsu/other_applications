#! /usr/bin/python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

plt.plot([1, 1], [0, 1], color="red", linewidth=2)
plt.plot([1, 2], [2, 2], color="red", linewidth=2)
plt.plot([2, 2], [2, 1], color="red", linewidth=2)
plt.plot([2, 3], [1, 1], color="red", linewidth=2)

plt.text(0.5, 2.5, "s0", size=14, ha="center")
plt.text(1.5, 2.5, "s1", size=14, ha="center")
plt.text(2.5, 2.5, "s2", size=14, ha="center")
plt.text(0.5, 1.5, "s3", size=14, ha="center")
plt.text(1.5, 1.5, "s4", size=14, ha="center")
plt.text(2.5, 1.5, "s5", size=14, ha="center")
plt.text(0.5, 0.5, "s6", size=14, ha="center")
plt.text(1.5, 0.5, "s7", size=14, ha="center")
plt.text(2.5, 0.5, "s8", size=14, ha="center")
plt.text(0.5, 2.3, "start", size=14, ha="center")
plt.text(2.5, 0.3, "goal", size=14, ha="center")

ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

plt.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="off", right="off", left="off", labelleft="off")
line, = ax.plot([0.5], [2.5], marker="o", color="g", markersize=60)
