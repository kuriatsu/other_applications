#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

def getData(filename, mode):
    file = open(filename, 'r')
    reader = csv.reader(file)
    header = next(reader)
    data = []

    for row in reader:

        if mode == 0:
            data.append(valuesAt(row, 0, 8, 9))
        elif mode == 1:
            data.append(valuesAt(row, 0, 1))
        elif mode == 2:
            data.append(valuesAt(row, 0, 1, 2))
        elif mode == 3:
            data.append(valuesAt(row, 0, 6, 9))
        elif mode == 4:
            data.append(valuesAt(row, 0, 7))
    file.close()

    return data


def alignData(operation_mode, odom_list, pose_list, vel_list, operation_list, obstacle_list):

    aligned_data = []
    pose_itr = 0
    vel_itr = 0
    operation_itr = 0
    obstacle_itr = 0

    mileage = 0
    dist = 100
    vel = None
    shift = None
    accel = None
    brake = None
    obstacle = None

    obstacle_pose = [-42.2010040283, 34.2079963684]

    for odom_itr, odom in enumerate(odom_list):

        if pose_list[pose_itr][0] <= odom[0] and pose_list[-1][0] >= odom[0]:

            dist = np.sqrt((pose_list[pose_itr][1] - obstacle_pose[0]) ** 2 + (pose_list[pose_itr][2] - obstacle_pose[1]) ** 2)
            pose_itr += 1

        if operation_mode == 'ras':

            if operation_list[operation_itr][0] <= odom[0] and operation_list[-1][0] > odom[0] and operation_list[operation_itr] != 0.0:

                shift = np.sqrt((operation_list[operation_itr][1] - obstacle_pose[0]) ** 2 + (operation_list[operation_itr][2] - obstacle_pose[1]) ** 2)
                print(shift)
                operation_itr += 1

        elif operation_mode == 'joy':

            if operation_list[operation_itr][0] <= odom[0] and operation_list[-1][0] > odom[0]:

                if operation_list[operation_itr][2] == 1.0:
                    accel = None
                else:
                    accel = (1.0 - operation_list[operation_itr][2]) * 0.5

                if operation_list[operation_itr][1] == 1.0:
                    brake = None
                else:
                    brake = (1.0 - operation_list[operation_itr][1]) * 0.5

                operation_itr += 1

        if vel_list[vel_itr][0] <= odom[0] and vel_list[-1][0] >= odom[0]:

            vel = vel_list[vel_itr][1]
            vel_itr += 1

        if obstacle_list[obstacle_itr][0] <= odom[0] and obstacle_list[-1][0] >= odom[0]:

            obstacle = obstacle_list[obstacle_itr][1]
            obstacle_itr += 1

        mileage += np.sqrt((odom[1] - odom_list[odom_itr-1][1])**2 + (odom[2] - odom_list[odom_itr-1][2])**2)
        aligned_data.append([odom[0], mileage, vel, dist, shift, accel, brake, obstacle])
        shift = None
        obstacle = None

    return aligned_data


def valuesAt(list, *args):
    return [float(list[index]) for index in args]


def cutData(aligned_data):

    range_frag = 0
    round = 0
    close_data_list = []
    closest_data_list = []
    cutted_data = []
    start_time = 0

    for data in aligned_data:
        if data[3] < 1.0:
            if range_frag == 0:
                close_data_list.clear()
                range_frag = 1
            close_data_list.append(data)

        else:
            if range_frag == 1:
                range_frag = 0
                closest_data_list.append(findValueFromArray(close_data_list, 3, 'min', 1.0))


    range_frag = 0
    for data in aligned_data:
        if closest_data_list[round][1] - 10.0 < data[1] < closest_data_list[round][1] + 10.0:
            if range_frag == 0:
                range_frag = 1
                start_time = data[0]
                start_mileage = data[1]
                cutted_data.clear()

            time = (data[0] - start_time) / 1000000000
            mileage = data[1]-start_mileage

            cutted_data.append([time, mileage, data[2], data[3], data[4], data[5], data[6], data[7]])

        else:
            if range_frag == 1:
                range_frag = 0
                print(round)
                outData(cutted_data, round)
                if round + 1 < len(closest_data_list):
                    round += 1


def findValueFromArray(array, row, mode, start_value):
    '''
    ~~args~~
    array : 2D list
    row : index of list in array for serach
    mode : min or max
    start_value : first value for comparison
    '''
    temp_value = start_value
    temp_list = []

    for list in array:
        if mode == 'min':
            if list[row] < temp_value:
                temp_value = list[row]
                temp_list = list
        if mode == 'max':
            if list[row] > temp_value:
                temp_value = list[row]
                temp_list = list

    return temp_list

def outData(cutted_data, round):

    filelist = ['1.csv', '2.csv', '3.csv', '4.csv', '5.csv', '6.csv']
    with open(filelist[round], 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(cutted_data)

    file.close()


if __name__ == "__main__":

    args = sys.argv
    odom = getData(args[2], 2)
    pose = getData(args[3], 2)
    vel = getData(args[4], 1)
    obstacle = getData(args[6], 4)

    if args[1] == 'ras':
        operation = getData(args[5], 0)

    elif args[1] == 'joy':
        operation = getData(args[5], 3)

    aligned_data = alignData(args[1], odom, pose, vel, operation, obstacle)
    cutted_data = cutData(aligned_data)
