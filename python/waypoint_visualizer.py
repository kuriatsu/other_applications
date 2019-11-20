#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

def getData(filename, *col):
    """read csv file
    ~args~
    filename : name of the output file
    col : list of columns where we want to extract
    ~return~
    data : list of data
    """

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
    """get data of the specified colomn
    """
    return [float(list[index]) for index in args]


def makeGraph(data_list, label_list):
    """write graph
    ~args~
    data_list : list of data
    label_list : list of label name for legend
    """

    data_array = np.array(data_list)
    ## plot colors#
    # color_palette = ['coral', 'coral', 'coral', 'coral', 'coral', 'coral', 'coral', 'midnightblue', 'midnightblue', 'midnightblue', 'indigo']
    color_palette = ['coral', 'olivedrab', 'turquoise', 'royalblue', 'fuchsia', 'gray', 'midnightblue', 'gold', 'lime', 'orchid', 'indigo']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, data in enumerate(data_list):
        data = np.array(data)
        ax.scatter(data[:, 0], data[:, 1], label = label_list[i], color=color_palette[i], marker='.', s=10)

    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.legend(loc='lower right', fontsize=12)
    plt.show()


def getAverageWaypoint(data_list):
    """find closest points from each list based on arbitaly list calcurate average and deviateion from waypoints
    ~args~
    data_list : all data of waypoint [[driver1], [driver2], ...]
    ~return~
    average_waypoints : list of average waypoint
    """

    last_index_list = [0] * len(data_list) # list of index in each driver's waypoint as the start index of searching closest point
    average_waypoints  = [] # list for average waypoint

    # search closest waypoint around a target point of one driver from other drivers
    for i, driver_0_point in enumerate(data_list[0]):

        closest_point_list  = [driver_0_point] # list of the found points of each drivers

        # search from each drivers
        for driver_id in range(1, len(data_list)):

            max_dist      = 2.0 # maximam distance from target point to searching point
            closest_point = np.array([0.0, 0.0]) # found point

            # search closest point. start from a point which extracted at last roop to 50 points forward
            for j, point in enumerate(data_list[driver_id][last_index_list[driver_id]:last_index_list[driver_id]+50]):

                dist = np.sqrt((driver_0_point[0] - point[0]) ** 2 + (driver_0_point[1] - point[1]) ** 2)

                if dist < max_dist:

                    # update values
                    max_dist      = dist
                    last_index_list[driver_id] += j
                    # get point
                    closest_point = point

            # if something is found, add to the list of closest points
            if(closest_point[0] != 0.0):

                closest_point_list.append(closest_point)

        # calcurate an average point from points
        average_waypoints.append(calcAverageOfPoints(closest_point_list))

    return average_waypoints


def calcAverageOfPoints(point_list):
    """calcurate average of the waypoints
    ~args~
    point_list : sample waypoints of each drivers

    ~return~
    position_ave : average point
    """
    # initialization
    position_sum = [0.0, 0.0, 0.0, 0.0]
    position_ave = [0.0, 0.0, 0.0, 0.0]

    # add values to get average
    for point in point_list:
        position_sum[0] = position_sum[0] + point[0]
        position_sum[1] = position_sum[1] + point[1]
        position_sum[2] = position_sum[2] + point[2]
        position_sum[3] = position_sum[3] + point[3]

    # calc average
    position_ave[0] = position_sum[0] / len(point_list) # x
    position_ave[1] = position_sum[1] / len(point_list) # y
    position_ave[2] = position_sum[2] / len(point_list) # yaw
    position_ave[3] = position_sum[3] / len(point_list) # vel

    return position_ave


def csvOut(filename, out_waypoint_list):
    """output to csv file
    ~args~
    filename : name of the output file
    out_waypoint_list : waypoint list for output
    """
    # open file
    file = open(filename, 'w')
    writer = csv.writer(file, lineterminator='\n')
    # add first row
    writer.writerow(["x", "y", "z", "yaw", "velocity", "change_flag"])

    # write data according to the format
    for out_waypoint in out_waypoint_list:

        out_waypoint.insert(2, "0.0") # add z axis data
        out_waypoint.append(0) # add change flag data
        writer.writerow(out_waypoint)

    file.close()


if __name__ == "__main__":

    args = sys.argv
    data_list = []

    # add data to list from csv
    for arg in args[1:]:
        data_list.append(getData(arg, 0, 1, 3, 4))

    # add calcurated average points to list
    average_waypoint = getAverageWaypoint(data_list)
    data_list.append(average_waypoint)

    ## output only average waypoints ##
    # data_list = getAverageWaypoint(data_list)
    # data_array = np.array([getAverageWaypoint(data_list)])

    # draw data
    args.append("ave") # for index of the graph
    makeGraph(data_list, args[1:])

    # out data
    csvOut("out.csv", average_waypoint)
