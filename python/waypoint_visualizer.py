#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

def getData(filename, *col):
    file = open(filename, 'r')
    reader = csv.reader(file)
    header = next(reader)
    data = []

    for row in reader:
        if row[0] != 'x':
            data.append(valuesAt(row, col))

    file.close()

    return data


def valuesAt(list, args):

    return [float(list[index]) for index in args]


def makeGraph(data_list):

    color_palette = ['coral', 'olivedrab', 'turquoise', 'royalblue', 'fuchsia', 'gray', 'midnightblue', 'gold', 'lime', 'orchid', 'indigo']
    # print(data_list)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, data in enumerate(data_list):
        data = np.array(data)
        ax.scatter(data[:, 0], data[:, 1], label = i, color=color_palette[i], marker='.', s=10)

    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.legend(loc='lower right', fontsize=12)
    plt.show()


def findClosestPoint(data_list):

    previous_index = [0] * len(data_array)
    average_points = []
    print(average_points)
    for i, driver_0_point in enumerate(data_list):

        selected_index = [0] * len(data_list)
        selected_index[0] = i
        closest_point_num = 0
        point_sum = np.array([0.0, 0.0])

        for driver_id in range(1, len(data_list)):

            smallest_dist = 5.0
            closest_point = np.array([0.0, 0.0])

            for j, point in enumerate(data_array[driver_id][previous_index[driver_id]:previous_index[driver_id] + 50]):

                dist = np.sqrt((driver_0_point[0] - point[0]) ** 2 + (driver_0_point[1] - point[1]) ** 2)

                if dist < smallest_dist:

                    smallest_dist = dist
                    previous_index[driver_id] += j
                    selected_index[driver_id] = [previous_index[driver_id], dist]
                    closest_point = point[0:2]

            if(closest_point[0] != 0.0):
                closest_point_num += 1
                point_sum += closest_point


        average_points.append(point_sum / closest_point_num)
    # print(average_points)
    return average_points


if __name__ == "__main__":

    args = sys.argv
    data_list = []

    for arg in args[1:]:
        data_list.append(getData(arg, 0, 1, 4))

    data_array = np.array(data_list)
    # print(data_array)
    np.append(data_array, findClosestPoint(data_array))
    # print(data_array.shape)
    makeGraph(data_array)

    # print(data_list)
