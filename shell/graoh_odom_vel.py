#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import csv
import sys
import numpy as np

def getData(filename, mode):
    file = open(filename, 'r')
    reader = csv.reader(file)
    header = next(reader)
    data = []

    for row in reader:

        if mode == 0:
            data.append(valuesAt(row, 0, 4))
        elif mode == 1:
            data.append(valuesAt(row, 0, 1))
        elif mode == 2:
            data.append(valuesAt(row, 0, 1, 2))
        elif mode == 3:
            data.append(valuesAt(row, 0, 6, 9))
    file.close()

    return data


def getObstacleDataInOdom(pose_list, odom_list):
    '''
    extract index from odom_list which is close to the obstacle position
    args:
    pose_list
    odom_list

    return:
    obstacle_index_list: from odom_list
    '''
    obstacle_pose = [-42.2010040283, 34.2079963684]
    prev_dist = 0
    approaching_frag = 0
    closest_pose_list = []
    closest_odom_list = []
    closest_pose_list_index = 0

    for pose_index, pose in enumerate(pose_list):

        dist = np.sqrt((pose[1] - obstacle_pose[0]) ** 2 + (pose[2] - obstacle_pose[1]) ** 2)

        if dist < 10:
            if dist < prev_dist:
                approaching_frag = 1

            elif dist > prev_dist and approaching_frag == 1:
                closest_pose_list.append(pose)
                approaching_frag = 0

        prev_dist = dist

    for odom in odom_list:
        # print(closest_pose_list_index, odom[0])
        if odom[0] > closest_pose_list[closest_pose_list_index][0]:
            closest_pose_list_index += 1
            if closest_pose_list_index == len(closest_pose_list):
                break
            closest_odom_list.append(odom[1])

    return closest_odom_list


def alignData(odom_list, closest_odom_list, vel_list, shift_list, joy_list):

    align_data = []
    closest_odom_list_index = 0
    shift_list_index = 0
    range_frag = 0
    begin_odom = 0
    shift_frag = 0
    pedestrian_frag = 0
    joy_frag = 0

    for odom_index, odom in enumerate(odom_list):

        if (closest_odom_list[closest_odom_list_index] - 10.0) < odom[1] < (closest_odom_list[closest_odom_list_index] + 10.0):
            if range_frag == 0:
                begin_odom = odom[1]
                begin_time = odom[0]
                range_frag = 1

            for joy_index, joy in enumerate(joy_list):
                if joy[1] != 1.0 and joy_list[joy_index-1][1] != 1.0 and joy_list[joy_index-1][0] < odom[0] < joy[0]:
                    joy_frag = -1

                if joy[2] != 1.0 and joy_list[joy_index-1][2] != 1.0 and joy_list[joy_index-1][0] < odom[0] < joy[0]:
                    joy_frag = 1

            for shift_index, shift in enumerate(shift_list):
                if shift[1] == shift_list[shift_index-1][1] < 1000000000 and shift_list[shift_index-1][0] < odom[0] < shift[0]:
                    shift_frag = 1
                else:
                    shift_frag = 0

            if (closest_odom_list[closest_odom_list_index] - 0.5) < odom[1] < (closest_odom_list[closest_odom_list_index] + 0.5):
                pedestrian_frag = 1
            else:
                pedestrian_frag = 0

            align_data.append([closest_odom_list_index, (odom[0] - begin_time)/10000000000, odom[1] - begin_odom, vel_list[odom_index][1], shift_frag, joy_frag, pedestrian_frag])
            print(odom[0] - begin_time ,odom[1] - begin_odom)
        else:
            if range_frag == 1:
                closest_odom_list_index += 1
                if closest_odom_list_index == len(closest_odom_list):
                    break
                range_frag = 0

    # print(np.array(align_data)[:, 2])


def valuesAt(list, *args):
    return [float(list[index]) for index in args]


if __name__ == "__main__":

    args = sys.argv
    odom = getData(args[1], 1)
    pose = getData(args[2], 2)
    vel = getData(args[3], 1)
    shift = getData(args[4], 0)
    joy = getdata(args[5], 3)
    closest_odom_list = getObstacleDataInOdom(pose, odom)
    alignData(odom, closest_odom_list, vel, shift)
