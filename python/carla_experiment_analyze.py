#! /usr/bin/python
# -*- coding:utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv

def summarizeData(extracted_data):

    intervene_time = [['experiment_type', 'world_id', 'intervene_time', 'distance to stopline', 'max_vel', 'min_vel']]
    accuracy_data = [['experiment_type', 'world_id', 'intervene_distance', 'is_correct']]

    fig = plt.figure()
    ax_dict = {'control': fig.add_subplot(2,2,1), 'ui': fig.add_subplot(2,2,2), 'button': fig.add_subplot(2,2,3), 'touch': fig.add_subplot(2,2,4)}
    for axes in ax_dict.values():
        axes.invert_xaxis()
        axes.set_xlim([50, -20])
        axes.set_ylim([0, 40])
        axes.set_xlabel("Mileage [m]", fontsize=20)
        axes.set_ylabel("Velocity [m/s]", fontsize=20)

    cmap = plt.get_cmap("tab10")

    for world_id, profile in extracted_data.items():
        print(world_id)

        # summarize data -get intervene time
        if profile.get('actor_action') in ['static', 'pose']:
            arr_data = np.array(profile.get('data'))[1:, :] # skip first column

            if profile.get('experiment_type') in ['control', 'ui']:
                intervene_column_index = np.where( (arr_data[:, 5] == 'throttle&brake') | (arr_data[:, 5] == 'throttle') )[0][0]
                intervene_time.append( [ profile.get('experiment_type'), world_id, arr_data[intervene_column_index, 0], arr_data[intervene_column_index, 4 ] ] )

            elif profile.get('experiment_type') in ['touch', 'button']:
                intervene_column_index = np.where( (arr_data[:, 5] == 'touch') | (arr_data[:, 5] == 'button') )[0][0]
                intervene_time.append( [ profile.get('experiment_type'), world_id, arr_data[intervene_column_index, 0], arr_data[intervene_column_index, 4 ] ] )

            else:
                intervene_column_index = np.where(arr_data[:, 5] != None)[0][0]
                intervene_time.append( [ profile.get('experiment_type'), world_id, arr_data[intervene_column_index, 0], arr_data[intervene_column_index, 4], arr_data[intervene_column_index, 5] ] )

            # get min vel and max vel
            intervene_time[-1].append(np.amax(arr_data[:, 1]) * 3.6)
            intervene_time[-1].append(np.amin(arr_data[:, 1]) * 3.6)
            writeMotionGraphOnPlt(ax_dict.get(profile.get('experiment_type')), arr_data[:, 4], arr_data[:, 1] * 3.6, arr_data[:, 5] != None, cmap(world_id%10))

        # get accuracy of intervention
        elif profile.get('actor_action') == 'cross':
            arr_data = np.array(profile.get('data'))[1:, :] # skip first column
            intervene_column_index = np.where(arr_data[:, 5] != None)[0][0]
            accuracy_data.append([
                profile.get('experiment_type'),
                world_id,
                arr_data[intervene_column_index][4],
                arr_data[intervene_column_index][1] > 1.0
                ])

    plt.show()
    return intervene_time, accuracy_data


def saveCsv(data, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def writeMotionGraphOnPlt(axes, x, y, area, cmap_color):

    axes.plot(x, y, color=cmap_color, alpha=0.5)

    for index, value in enumerate(area):
        if value:
            axes.fill([x[index], x[index], x[index] + 0.5, x[index] + 0.5], [0, 40, 40, 0], color=cmap_color, alpha=0.1)

    plt.legend(loc='lower right', fontsize=12)


def main():

    pickle_file = '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/Town04_data.pickle'
    intervene_time_out = '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/Town01_summalize_2.csv'
    intervene_acc_out = '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/Town01_accracy_2.csv'

    with open(pickle_file, 'rb') as f:
        extracted_data = pickle.load(f)

    intervene_time, intervene_acc = summarizeData(extracted_data)
    saveCsv(intervene_time, intervene_time_out)
    saveCsv(intervene_acc, intervene_acc_out)


if __name__ == '__main__':
    main()
