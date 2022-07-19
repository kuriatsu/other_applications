#! /usr/bin/python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from IPython.display import HTML

# 迷路作成
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


def simple_convert_into_pi_from_theta(theta):
    # 方策パラメータを行動方策piに変換。今回は、単純に各状態で各方策の割合を計算
    pi = np.zeros(theta.shape)
    for i in range(0, pi.shape[0]):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])

    pi = np.nan_to_num(pi)
    return pi


def get_next_state(pi, s):
    """方策から次の方向を選択し、状態sを返す
    pi : 方策パラメータ
    s : 現在状態
    ~return~
    s_next : 次の状態
    """
    direction = ["up", "right", "down", "left"]
    next_direction = np.random.choice(direction, p=pi[s, :]) # 方策の確率に従って、directionが選択される

    if next_direction == "up":
        s_next = s-3
    elif next_direction == "right":
        s_next = s+1
    elif next_direction == "down":
        s_next = s+3
    elif next_direction == "left":
        s_next = s-1

    return s_next

def goal_maze(pi):
    """goal(s=8)になるまで迷路を探索する
    pi : 方策
    ~return~
    trajectory : 探索経路履歴
    """
    s = 0
    trajectory = [0]
    while(1):
        next_s = get_next_state(pi, s)
        trajectory.append(next_s)
        if next_s == 8:
            break
        else:
            s = next_s

    return trajectory

def anim_traj(trajectory):
    def init():
        line.set_data([],[])
        return (line,)
    def animate(i):
        state = trajectory[i]
        x = (state % 3) + 0.5
        y = 2.5 - int(state / 3)
        line.set_data(x, y)
        return (line, )

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(trajectory), interval=200, repeat=False)
    return anim


# 方策（ランダムに進む際の制約条件）作成
# 行は状態（存在するマス）、列は↑、→、↓、←、nanは進めない
theta_0 = np.array([
    [np.nan, 1, 1, np.nan], #0
    [np.nan, 1, np.nan, 1], #1
    [np.nan, np.nan, 1, 1], #2
    [1, 1, 1, np.nan],      #3
    [np.nan, np.nan, 1, 1], #4
    [1, np.nan, np.nan, np.nan], #5
    [1, np.nan, np.nan, np.nan], #6
    [1, 1, np.nan, np.nan], #7
    ]) #8はゴールなので無し

pi_0 = simple_convert_into_pi_from_theta(theta_0)
trajectory = goal_maze(pi_0)
anim = anim_traj(trajectory)
HTML(anim.to_html5_video())
