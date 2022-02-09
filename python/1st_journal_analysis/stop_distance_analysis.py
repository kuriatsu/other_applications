#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# x = speed y=deceleration start position / intervention time (80m recognition range)
speed_list = np.arange(0.0, 80.0, 0.1)/3.6
mu = 0.7
safety_margin = 10
G = np.arange(0.1, mu+0.05, 0.05)
recognition_range = 80.0

stop_distance = speed_list.reshape(-1, 1) ** 2 / (2 * 9.8 * G)
int_time = (recognition_range - stop_distance - safety_margin) / (speed_list.reshape(-1, 1)+0.0001)
df_dist  = pd.DataFrame(stop_distance)
# df_dist = df_dist.set_axis(speed_list, axis="index")
df_dist = df_dist.set_axis(G, axis="columns")
df_dist["speed"] = speed_list*3.6
df_time = pd.DataFrame(int_time)
df_time = df_time.set_axis(G, axis="columns")
df_time["speed"] = speed_list*3.6

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
sns.lineplot(x="speed", y="stopping_distance", data=df_dist.melt(id_vars="speed", var_name="G", value_name="stopping_distance"), hue="G", ax=ax1)
ax = sns.lineplot(x="speed", y="intervention_time", data=df_time.melt(id_vars="speed", var_name="G", value_name="intervention_time"), hue="G", ax=ax2, dashes=[(2,2)]*len(G), style="G", legend=False)
ax1.set_ylim(0.0, 150.0)
ax2.set_ylim(0.0, 10.0)
ax1.set_xlabel("Velocity [km/h]")
ax1.set_ylabel("Deceleration Start Distance [m]")
ax2.set_ylabel("Intervention Time [s] (80m detection range)")

ax = sns.lineplot(x="speed", y="stopping_distance", data=df_dist.melt(id_vars="speed", var_name="G", value_name="stopping_distance"), hue="G")
ax.set_ylim(0.0, 150.0)
ax.set_xlabel("Velocity [km/h]")
ax.set_ylabel("Deceleration Start Distance [m]")

ax = sns.lineplot(x="speed", y="intervention_time", data=df_time.melt(id_vars="speed", var_name="G", value_name="intervention_time"), hue="G")
ax.set_ylim(0.0, 10.0)
ax.set_xlabel("Velocity [km/h]")
ax.set_ylabel("Intervention Time [s] (80m detection range)")

# detection range vs intervention time
detection_range = np.arange(0.0, 150.0, 0.1)
G_list = np.array([0.1, 0.2, 0.3])
vel_list = np.array([20.0, 40.0, 60.0])
safety_margin = 10
recognition_time = 0.5

result = []
fig, ax = plt.subplots()
for G in G_list:
    for vel in vel_list:
        stop_distance = (vel/3.6)**2 / (2*9.8*G)
        int_time = (detection_range - stop_distance - safety_margin) / vel - recognition_time
        buf_result = {
            "G": G,
            "vel" : vel,
            "int_time": int_time,
            }
        result.append(buf_result)
        sns.lineplot(x=detection_range, y=int_time, ax=ax)


# speed vs intervention time
vel_list = np.arange(0.0, 80.0, 0.1)
detection_range = [80, 120, 200]
G_list = np.array([0.1, 0.2, 0.3])
safety_margin = 10
recognition_time = 0.5
result = pd.DataFrame(columns=["vel", "int_time", "G", "Range[m]"])
for G in G_list:
    for range in detection_range:
        stop_distance = (vel_list/3.6) ** 2 / (2 * 9.8 * G)
        int_time = (range - stop_distance - safety_margin) / (vel_list+0.0001) - recognition_time
        buf = pd.DataFrame(columns=result.columns)
        buf["vel"] = vel_list
        buf["int_time"] = int_time
        buf["G"] = G
        buf["Range[m]"] = range
        result = pd.concat([result, buf], ignore_index=True)

palette = sns.color_palette("mako_r", 3)
fig, ax = plt.subplots()
sns.lineplot(data=result, x="vel", y="int_time", hue="G", style="Range[m]", dashes=True, ax=ax, palette=palette)
ax.set_ylim(0.0, 7.0)
ax.set_xlabel("Velocity [km/h]", fontsize=14)
ax.set_ylabel("Intervention Time [s]", fontsize=14)
plt.show()
