#! /usr/bin/python3
#! -*- coding : utf-8 -*-


# touple sort
# animal_list = [
#     ("ライオン", 58),
#     ("チーター", 110),
#     ("シマウマ", 60),
#     ("トナカイ", 80)
# ]
#
# faster_list = sorted(animal_list, key=lambda ani: ani[1], reverse=True)
#
# for i in faster_list: print(i)

# generator
# def genPrime(maximum):
#     num = 2
#     while num < maximum:
#         is_prime = True
#
#         for i in range(2, num):
#             if num % i == 0:
#                 is_prime = False
#
#         if is_prime:
#             yield num;
#
#         num+=1
#
#
# it = genPrime(50)
# for i in it:
#     print(i, end=", ")

# encrypt decrypt
# from Crypto.Cipher import AES
# import base64
#
# message = input("input sentence for scecret cord")
# password = input("input password")
#
# iv = "LdfasDFF34w434dg"
# mode = AES.MODE_CBC
#
# def mkpad(s, size):
#     s = s.encode("utf-8")
#     pad = b' ' * (size - len(s) % size)
#     return s + pad
#
# def encrypt(password, data):
#     password = mkpad(password, 16)
#     password = password[:16]
#     data = mkpad(data, 16)
#
#     aes = AES.new(password, mode, iv)
#     data_cipher = aes.encrypt(data)
#     return base64.b64encode(data_cipher).decode("utf-8")
#
# def decrypt(password, encdata):
#     password = mkpad(password, 16)
#     password = password[:16]
#     aes = AES.new(password, mode, iv)
#     encdata = base64.b64decode(encdata)
#     data = aes.decrypt(encdata)
#     return data.decode("utf-8")
#
# enc = encrypt(password, message)
# dec = decrypt(password, enc)
#
# print("暗号化: ", enc)
# print('復号化: ', dec)

# find files from path
# import os
# import sys
# import fnmatch
# import datetime
# import math
#
# if len(sys.argv) <= 1:
#     print("[USAGE] findfile [--uname][--wild][--desc] name")
#     sys.exit(0)
#
# search_mode = "name"
# search_func = lambda target, name : (target == name)
# name = ""
# desc_mode = False
# for v in sys.argv:
#     if v == "--name":
#         search_mode = "name"
#         search_func = lambda target, name : (target == name)
#     elif v == "--wild":
#         search_mode = "wild"
#         search_func = lambda target, pat : fnmatch.fnmatch(target, pat)
#     elif v == "--desc":
#         desc_mode = True
#     else:
#         name = v
#
# print("option")
# print("| search mode :", search_mode, name)
# print("| desc_mode :", desc_mode)
#
# for root, dirs, files in os.walk("."):
#     for fname in files:
#         path = os.path.join(root, fname)
#         b = search_func(fname, name)
#         if b == False:
#             continue
#         if desc_mode:
#             info = os.stat(path)
#             kb = math.ceil(info.st_size / 1024)
#             mt = datetime.datetime.fromtimestamp(info.st_mtime)
#             s = "{0}, {1}kb, {2}".format(path, kb, mt.strftime("%Y-%m-%d"))
#             print(s)
#         else:
            # print(path)


# kalman filter
import matplotlib.pyplot as plt
import numpy as np

def localKalmanFilter(observe, state_previous, variance_previous, noise_state, noise_observe):
    # print(observe)
    # 状態方程式。ローカルレベルモデルなので、状態と観測値は同じ。観測方程式を立てる必要がない。本当は予測された状態から予測された観測値を算出する必要がある。
    # 更に、真の状態は常に一定の値
    state_predicted = state_previous
    # 状態の予測誤差の分散
    variance_predicted = variance_previous + noise_state
    # カルマンゲイン　観測値を元に現状態の状態の予測誤差の分散の補正をするが、状態の信頼性が小さければ分散が大きくなり、補正も大きくなる。観測理の信頼性が小さければ、分散も大きくなり補正はしなくなる
    k_gain = variance_predicted / (variance_predicted + noise_observe)
    # 状態を補正。ローカルレベルモデルでなければ、(観測値-予測された状態)　ではなく、 (観測値-観測方程式で計算された、予測された観測値)になるが、今回は同じなのでこうなっている。
    state_filtered = state_predicted  + k_gain * (observe - state_predicted)
    # 予測誤差の分散を補正
    variance_filtered = (1 - k_gain) * variance_predicted

    return state_filtered, variance_filtered

data_list = [
1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140, 995, 935, 1110, 994, 1020, 960, 1180, 799, 958, 1140, 1100,
1210, 1150, 1250, 1260, 1220, 1030, 1100, 774, 840, 874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969, 831, 726,
456, 824, 702, 1120, 1100, 832, 764, 821, 768, 845, 864, 862, 698, 845, 744, 796, 1040, 759, 781, 865, 845,
944, 984, 897, 822, 1010, 771, 676, 649, 846, 812, 742, 801, 1040, 860, 874, 848, 890, 744, 749, 838, 1050,
918, 986, 797, 923, 975, 815, 1020, 906, 901, 1170, 912, 746, 919, 718, 714, 740
]
print(len(data_list))

state_list = [0]
variance_list = [1000]

for i in range(0, 200):
    print(i)
    state, variance = localKalmanFilter(data_list[i], state_list[i], variance_list[i], noise_state=10, noise_observe=1000)
    state_list.append(state)
    variance_list.append(variance)
    if i >= len(data_list)-1:
        data_list.append(state)
# print(len(state_list), len(data_list))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(1871, 2071), np.array(data_list[:-1]), label="data", color="coral")
ax.plot(np.arange(1871, 2071), np.array(state_list[1:]), label="data", color="turquoise")
plt.show()
