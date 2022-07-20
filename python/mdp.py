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


def softmax_convert_into_pi_from_theta(theta):
    """ソフトマックスで方策パラメータを行動方策（割合）に変換
    """
    beta = 1.0
    pi = np.zeros(theta.shape)

    for i in range(0, pi.shape[0]):
        # simpleではなく、softmax (log) で割合を計算
        pi[i, :] = np.exp(beta*theta[i, :]) / np.nansum(np.exp(beta*(theta[i, :])))

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
        action = 0
        s_next = s-3
    elif next_direction == "right":
        action = 1
        s_next = s+1
    elif next_direction == "down":
        action = 2
        s_next = s+3
    elif next_direction == "left":
        action = 3
        s_next = s-1

    return [action, s_next]


def update_theta(theta, pi, history):
    """方策勾配法で方策を更新
    方策勾配定理を近似的に実装したREINFORCE algorithmを実装
    softmaxを用いることで、解析的に導出しやすいのと、万が一（なることはないけど）thetaが負になっても確率(log(pi))計算が可能
    """
    eta = 0.1 # 学習率
    T = len(history) - 1 # ゴールまでの総ステップ

    # deltaを更新するもととなるtheta
    delta_theta = theta.copy()

    for i in range(0, theta.shape[0]): # 行
        for j in range(0, theta.shape[1]): # 方策パラメータ
            if not (np.isnan(theta[i, j])): # 状態パラメータがnan(進めない)の場合は、学習しないのでskip
                SA_i = [SA for SA in history if SA[0] == i] # state i に到達した履歴
                SA_ij = [SA for SA in history if SA == [i, j]] # state i で行動jをした履歴
                N_i = len(SA_i) # state i に到達した回数
                N_ij = len(SA_ij) # state i で行動j をとった回数
                delta_theta[i, j] = (N_ij + pi[i, j] * N_i) / T # 方策勾配法(1)

    new_theta = theta + eta * delta_theta # 方策勾配法(2)
    return new_theta

def goal_maze(pi):
    """goal(s=8)になるまで迷路を探索する
    pi : 方策
    ~return~
    history : 探索経路履歴 (状態、その状態でとった行動)
    """
    s = 0
    history = [[0, np.nan]] # state, action
    while(1):
        [action, next_s] = get_next_state(pi, s)
        history[-1][1] = action # next_stateから行動を逆算したので、現状態（格納済み）のactionとして保存
        history.append([next_s, np.nan])
        if next_s == 8:
            break
        else:
            s = next_s

    return history


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

# 最初にトライ
pi_0 = softmax_convert_into_pi_from_theta(theta_0)
history = goal_maze(pi_0)
trajectory = [row[0] for row in history]
anim = anim_traj(trajectory)
HTML(anim.to_html5_video())

# 方策勾配法で最適化
stop_epsilon = 10**-8
is_continue = True
count = 1
theta = theta_0
pi = pi_0
while is_continue:
    history = goal_maze(pi)
    new_theta = update_theta(theta, pi, history)
    new_pi = softmax_convert_into_pi_from_theta(new_theta)
    print(np.sum(np.abs(new_pi - pi)))
    print(f"goal step num : {len(history)-1}")

    if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
        is_continue =  False
    else:
        theta = new_theta
        pi = new_pi

trajectory = [row[0] for row in history]
anim = anim_traj(trajectory)
HTML(anim.to_html5_video())
