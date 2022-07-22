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


##########################
# シンプルにランダム探索
##########################

def simple_convert_into_pi_from_theta(theta):
    # 方策パラメータを行動方策piに変換。今回は、単純に各状態で各方策の割合を計算
    pi = np.zeros(theta.shape)
    for i in range(0, pi.shape[0]):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])

    pi = np.nan_to_num(pi)
    return pi

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


# ランダム方策でトライ
pi_0 = simple_convert_into_pi_from_theta(theta_0)
history = goal_maze(pi_0)
trajectory = [row[0] for row in history]
anim = anim_traj(trajectory)
HTML(anim.to_html5_video())


####################
# 方策勾配法で最適化
####################
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

# 方策勾配法の実行
stop_epsilon = 10**-8
is_continue = True
count = 1
theta = theta_0
pi = softmax_convert_into_pi_from_theta(theta_0)

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

####################
# 価値反復法
####################
def goal_maze_Q(Q, epsilon, eta, gamma, pi_0):
    """Sarsaで迷路を解く関数
    ~args~
    Q: 行動価値
    epsilon : ランダムな行動をとる確率
    eta : 学習率
    pi_0: ランダムな行動をとる場合の方策パラメータ
    ~return~
    history : 行動と経路
    Q : 行動価値関数
    """
    s = 0
    history = [[0, np.nan]]

    while(1):
        [a, s_next] = get_next_state_greddy(s, Q, epsilon, pi_0)
        history[-1][1] = a
        history.append([s_next, np.nan])
        print(s)
        if s_next == 8:
            # goalに到達したら、報酬を与える。次の行動は無し
            r = 1
            a_next = np.nan
        else:
            # 到達しなければ、行動価値関数を計算するために次の行動を暫定的に決める。（行動はしない）
            # 暫定的な行動選択はε-greddyで行う
            r = 0
            [a_next, _] = get_next_state_greddy(s_next, Q, epsilon, pi_0)

        # 価値観数を更新
        Q = Sarsa(s, a, r, s_next, a_next, Q, eta, gamma)
        if s_next == 8:
            break
        else:
            s = s_next

    return [history, Q]

def Sarsa(s, a, r, s_next, a_next, Q, eta, gamma):
    """Sarsaによる行動価値観数の更新
    (状態s, aの行動価値関数Q[s, a])=(現状態sの即時報酬r)+(行動の結果得られる状態s_nextでの行動価値関数値Q[s_next, a_next]*時間割引ritu
    率gannma)
    であるべき(マルコフ過程なので、1つ先の行動価値のみを考えればOK)
    学習が不十分な時は、等式が成り立たず、TD誤差(右辺-左辺)が生まれる
    → Q[s,a] = Q[s,a]+η*TD誤差 で更新(η:学習率)
    ~args~
    s:現状態
    a:現状態に至るためにとった行動
    r:即時報酬
    s_next: 次状態
    a_next: 次状態に至るためにとる行動
    Q: 行動価値観数
    eta: 学習率
    gamma: 時間割引率
    ~return~
    Q : 更新された行動価値関数
    """
    if s_next == 8: # ゴールした場合、行動価値関数=状態価値関数
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] = eta * (r + gamma * Q[s_next, a_next] - Q[s, a])

    return Q


def get_next_state_greddy(s, Q, epsilon, pi_0):
    """方策から次の方向をε-greddy法で選択し、行動a, 状態sを返す
    s : 現在状態
    Q : 行動価値観数
    epsilon : 価値反復法において、行動価値観数に従わずにランダムに行動する確率
    pi_0 : 初期方策パラメータ
    ~return~
    s_next : 次の状態
    action : これから採用する行動
    """
    direction = ["up", "right", "down", "left"]

    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi_0[s, :]) # 方策の確率に従って、directionが選択される
    else:
        next_direction = direction[np.nanargmax(Q[s, :])]

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

# Sarsaで迷路探索
Q = np.random.rand(theta_0.shape[0], theta_0.shape[1]) * theta_0 # 行動価値観数。初期値はランダムに決定しておく
pi_0 = simple_convert_into_pi_from_theta(theta_0)
eta = 0.1
gamma = 0.9
epsilon = 0.5
v = np.nanmax(Q, axis=1) # 状態ごとの価値の最大
is_continue = True
episode = 1

while is_continue:
    print(f"episode {episode}")

    epsilon = epsilon / 2
    [history, Q] = goal_maze_Q(Q, epsilon, eta, gamma, pi_0)

    new_v = np.nanmax(Q, axis=1)
    print(np.sum(np.abs(new_v - v))) # 状態価値の変化
    v = new_v

    episode += 1
    if episode > 100:
        break

trajectory = [row[0] for row in history]
anim = anim_traj(trajectory)
HTML(anim.to_html5_video())
