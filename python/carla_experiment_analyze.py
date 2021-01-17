#! /usr/bin/python
# -*- coding:utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv

def summarizeData(extracted_data):

    intervene_time = [['experiment_type', 'world_id', 'intervene_time', 'distance to stopline', 'max_vel', 'min_vel', 'intervene_count']]
    accuracy_data = [['experiment_type', 'world_id', 'intervene_time', 'intervene_distance', 'is_correct']]
    face_turn_result = [['experiment_type', 'world_id', 'actor_action', 'count']]

    fig = plt.figure()
    # for 202012experiment/ data
    # ax_dict = {'control': fig.add_subplot(2,2,1), 'ui': fig.add_subplot(2,2,2), 'button': fig.add_subplot(2,2,3), 'touch': fig.add_subplot(2,2,4)}
    ax_dict = {'baseline': fig.add_subplot(2,2,1), 'control': fig.add_subplot(2,2,2), 'button': fig.add_subplot(2,2,3), 'touch': fig.add_subplot(2,2,4)}
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

            if np.where(arr_data[:, 5] != None)[0].size == 0:
                print('skiped no intervention')
                continue

            # for 202012experiment/ data
            # if profile.get('experiment_type') in ['ui', 'control']:
            if profile.get('experiment_type') in ['baseline', 'control']:
                intervene_column_index = np.where( (arr_data[:, 5] == 'throttle&brake') | (arr_data[:, 5] == 'throttle') )[0][0]
                intervene_time.append( [ profile.get('experiment_type'), world_id, arr_data[intervene_column_index, 0], arr_data[intervene_column_index, 4 ], None, None, None] )

            elif profile.get('experiment_type') in ['touch', 'button']:

                # intervene_column_index = np.where( (arr_data[:, 5] == 'throttle&brake') | (arr_data[:, 5] == 'throttle') )[0][0]
                # get intervene count to find how dificult to touch
                intervene_column_index_list = np.where( (arr_data[:, 5] == 'touch') | (arr_data[:, 5] == 'button') )[0]
                intervene_column_index = intervene_column_index_list[0]

                last_intervene_time = arr_data[intervene_column_index, 0]
                intervene_count = 1
                for column in intervene_column_index_list:
                    print(arr_data[column, 0])
                    if (arr_data[column, 0] - last_intervene_time) > 0.5:
                        intervene_count += 1
                        last_intervene_time = arr_data[column, 0]
                        print(intervene_count, last_intervene_time)

                intervene_time.append( [ profile.get('experiment_type'), world_id, arr_data[intervene_column_index, 0], arr_data[intervene_column_index, 4 ], None, None, intervene_count ] )

            else:
                intervene_column_index = np.where(arr_data[:, 5] != None)[0][0]
                intervene_time.append( [ profile.get('experiment_type'), world_id, arr_data[intervene_column_index, 0], arr_data[intervene_column_index, 4], None, None, arr_data[intervene_column_index, 5] ] )

            # get min vel and max vel
            intervene_time[-1][4] = np.amax(arr_data[:, 1]) * 3.6
            intervene_time[-1][5] = np.amin(arr_data[:, 1]) * 3.6
            writeMotionGraphOnPlt(ax_dict.get(profile.get('experiment_type')), arr_data[:, 4], arr_data[:, 1] * 3.6, arr_data[:, 5] != None, cmap(world_id%10))

        # get accuracy of intervention
        elif profile.get('actor_action') == 'cross':

            arr_data = np.array(profile.get('data'))[1:, :] # skip first column
            intervene_index = np.where(arr_data[:, 5] != None)
            if intervene_index[0].size == 0:
                accuracy_data.append([
                    profile.get('experiment_type'),
                    world_id,
                    None,
                    True
                ])
            else:
                intervene_column_index = intervene_index[0][0]
                accuracy_data.append([
                    profile.get('experiment_type'),
                    world_id,
                    arr_data[intervene_column_index][0],
                    arr_data[intervene_column_index][4],
                    arr_data[intervene_column_index][1] < 1.0
                ])



        if profile.get('actor_action') in ['static', 'pose', 'cross']:
            last_face_direction = arr_data[0, 6]
            face_turn_count = 0
            pub_rate = 2

            for index, face_direction in enumerate(arr_data[:, 6]):
                # print(face_direction)
                if face_direction != last_face_direction:
                    face_direction_count_length = min(pub_rate // 2, len(arr_data)-index)
                    # print(arr_data[index:index + face_direction_count_length, 6])
                    face_direction_count = np.where( arr_data[index:index + face_direction_count_length, 6] == face_direction )[0].size
                    if face_direction_count == face_direction_count_length:
                        face_turn_count += 1
                        last_face_direction = face_direction

            face_turn_result.append([profile.get('experiment_type'), profile.get('world_id'), profile.get('actor_action'), face_turn_count])


    plt.show()
    return intervene_time, accuracy_data, face_turn_result


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

    pickle_file = '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/yokoyama/Town01.pickle'
    intervene_time_out = '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/yokoyama/Town01_summalize.csv'
    intervene_acc_out = '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/yokoyama/Town01_accracy.csv'
    face_turn_out = '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/yokoyama/Town01_face.csv'

    with open(pickle_file, 'rb') as f:
        extracted_data = pickle.load(f)

    intervene_time, intervene_acc, face_turn = summarizeData(extracted_data)
    saveCsv(intervene_time, intervene_time_out)
    saveCsv(intervene_acc, intervene_acc_out)
    saveCsv(face_turn, face_turn_out)


if __name__ == '__main__':
    main()
