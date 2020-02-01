#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import matplotlib.animation as animation
import argparse

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


def makeAnimeGraph(data_list):
    """write graph
    ~args~
    data_list : list of all points
    label_list : list of label name for legend
    """

    # data_array = np.array(data_list)
    ## plot colors#
    # color_palette = ['coral', 'coral', 'coral', 'coral', 'coral', 'coral', 'coral', 'midnightblue', 'midnightblue', 'midnightblue', 'indigo']
    color_palette = ['coral', 'olivedrab', 'turquoise', 'royalblue', 'fuchsia', 'gray', 'midnightblue', 'gold', 'lime', 'orchid', 'indigo']

    fig = plt.figure()

    for i in range(1, len(data_list[0]), 10):
        data = np.array(data_list[0][0:i])
        print(i)
        print("number:{}, x={}, y={}".format(i, data_list[0][i][0], data_list[0][i][1]))
        im = plt.scatter(data[:, 0], data[:, 1], marker='.', alpha=0.2)
        plt.xlabel("x", fontsize=20)
        plt.ylabel("y", fontsize=20)
        plt.xlim(-47350, -47150)
        plt.ylim(57500, 57700)
        plt.pause(0.01)
        plt.clf()

    # plt.show()



def getAverageWaypoint(data_list):
    """Search closest waypoints from other drivers around a target point of first driver of data_list. Then calcurate average point.
    ~args~
    data_list : all data of waypoint [driver1 [point]...], [driver2 [point]...], ...]
    ~return~
    average_waypoints : list of average waypoint
    """

    last_index_list    = [0] * len(data_list) # list of index in each driver's waypoint as the start index of searching closest point
    average_waypoints  = [] # list for average waypoint

    # search closest waypoint around a target point of one driver from other drivers
    for i, driver_0_point in enumerate(data_list[0]):
        closest_point_list  = [driver_0_point] # list of the found points of each drivers

        # search from each drivers
        for driver_id in range(1, len(data_list)):
            max_dist      = 2.0 # maximam distance from target point to searching point
            closest_point = np.array([0.0, 0.0]) # found point

            # search closest point. start from a point which extracted at last loop to 50 points forward
            for j, point in enumerate(data_list[driver_id][last_index_list[driver_id]:last_index_list[driver_id]+50]):
                dist = np.sqrt((driver_0_point[0] - point[0]) ** 2 + (driver_0_point[1] - point[1]) ** 2)

                if dist < max_dist:
                    max_dist                   = dist  # update values for next loop
                    last_index_list[driver_id] += j    # update values for next loop
                    closest_point              = point # get point

            # if something is found, add to the list of closest points
            if(closest_point[0] != 0.0):
                closest_point_list.append(closest_point)

        average_waypoints.append(calcAverageOfPoints(closest_point_list)) # calcurate an average point from points

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

    file = open(filename, 'w') # open file
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(["x", "y", "z", "yaw", "velocity", "change_flag"]) # add first row

    # write data according to the format
    for out_waypoint in out_waypoint_list:

        out_waypoint.insert(2, "0.0") # add z axis data
        out_waypoint.append(0) # add change flag data
        writer.writerow(out_waypoint)

    file.close()


if __name__ == "__main__":

    argparser = argparse.ArgumentParser( description = __doc__)
    argparser.add_argument(
        '-b','--base_waypoint',
        metabar='/path/to/file.csv',
        default=None,
        help='Fundamental waypoint for calcurating average waypoint. Select most beautiful one.')
    argparser.add_argument(
        '-f', '--file',
        metabar='/path/to/file1.csv /path/to/file2.csv ...',
        default=None,
        nargs='+',
        required=true,
        help='Waypoint files for calcurating average waypoint.')
    argparser.add_argument(
        '--only_average',
        action='store_true',
        help='Visualize only average waypoints.')
    argparser.add_argument(
        '--only_visualize',
        action='store_true',
        help='Do not calcurate average.')
    argparser.add_argument(
        '--animation',
        action='store_true',
        help='visualize first waypoint with animation.')

    args = argparser.parse_args()
    data_list = []
    legend_list = []

    if args.base_waypoint is not None:
        data_list.append(getData(args.base_waypoint, 0, 1, 3, 4))
        legend_list.append(args.base_waypoint)

    if args.file is not None:
        for filename in args.file:
            data_list.append(getData(filename, 0, 1, 3, 4))
            legend_list.append(filename)

    if !args.only_visualize:
        data_list.append(getAverageWaypoint(data_list))
        legend_list.append('average')
        # out data
        csvOut("out.csv", average_waypoint)

    if args.only_average:
        data_list = data_list[-1]
        legend_list = legend_list[-1]

    if args.animation:
        makeAnimeGraph(data_list[1])
        sys.exit()


    makeGraph(data_list, legend_list) # draw data
