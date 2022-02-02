#! /usr/bin/python
# -*- coding:utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

# actor_data = {
#     'world_id': target_object.object.id,
#     'data': offsetMileage(profile_data), <- offset to make [0 = target position]
#     'experiment_type': experiment_type,
#     'actor_action' : actor_action
#     }

# profile_data=[time, ego_vel, ego_mileage, target_ego_dist, wp_dist, intervene_type]

def summarizeData(extracted_data):

    out_list = [[
        'subject',
        'experiment_type',
        'world_id',
        'actor_action',
        'first_intervene_time',
        'last_intervene_time',
        'first_intervene_distance',
        'last_intervene_distance',
        'initial_vel',
        'max_vel',
        'min_vel',
        'std_vel',
        'mean_vel',
        'max_acc',
        'min_acc',
        'intervene_vel',
        'face_turn_count',
        ]]


    intervene_type = {
        'baseline' : 'BASELINE',
        'control' : 'CONTROL',
        'button' : 'BUTTON',
        'touch' : 'TOUCH',
    }


    for world_id, profile in extracted_data.items():
        if profile.get('experiment_type') is None:
            continue

        arr_data = np.array(profile.get('data'))[1:, :] # skip first column
        if profile.get('experiment_type') in ['baseline', 'control'] and "button" in arr_data[:, 5]:
            print('skip wrong intervention baseline/control')
            continue
        if profile.get('experiment_type') in ['button', 'touch'] and ("throttle" in arr_data[:, 5] or "throttle&brake" in arr_data[:, 5]):
            print('skip wrong intervention button/touch')
            continue


        intervene_start_column_index = None
        intervene_end_column_index = None

        first_intervene_time = None
        last_intervene_time = None
        first_intervene_distance = None
        last_intervene_distance = None
        initial_vel = 50.0
        min_vel = None
        max_vel = None
        std_vel = None
        mean_vel = None
        intervene_vel = None
        face_turn_count = None
        max_acc = None
        min_acc = None

        # intervention index
        intervene_index_list = np.where(arr_data[:, 5] != None)[0]
        if len(intervene_index_list) > 0:
            # intervene_start_column_index = intervene_index_list[0]
            # intervene_end_column_index   = intervene_index_list[-1]
            if profile.get('experiment_type') in ['baseline', 'control']:
                # print(np.where( (arr_data[:, 5] == 'throttle&brake') | (arr_data[:, 5] == 'throttle') ))
                intervene_start_column_index = np.where( (arr_data[:, 5] == 'throttle&brake') | (arr_data[:, 5] == 'throttle') )[0][0]
                intervene_end_column_index   = np.where( (arr_data[:, 5] == 'throttle&brake') | (arr_data[:, 5] == 'throttle') )[0][-1]

            elif profile.get('experiment_type') in ['touch', 'button']:
                intervene_start_column_index = np.where( (arr_data[:, 5] == 'touch') | (arr_data[:, 5] == 'button') )[0][0]
                intervene_end_column_index = np.where( (arr_data[:, 5] == 'touch') | (arr_data[:, 5] == 'button') )[0][-1]

            first_intervene_time = arr_data[intervene_start_column_index, 0]
            first_intervene_distance = arr_data[intervene_start_column_index, 4]

            last_intervene_time = arr_data[intervene_start_column_index, 0]
            last_intervene_distance = arr_data[intervene_start_column_index, 4]
            intervene_vel = arr_data[intervene_start_column_index, 1] * 3.6

            # acc
            acc_start_index = intervene_start_column_index
            acc_end_index = np.where(arr_data[:, 4] <= -20.0)[0][0]
            max_acc = 0.0
            min_acc = 0.0
            for i in range(acc_start_index, acc_end_index):
                if (arr_data[i, 0] - arr_data[i-1, 0]) == 0.0:
                    continue
                acc = (arr_data[i, 1] - arr_data[i-1, 1]) / (arr_data[i, 0] - arr_data[i-1, 0])
                if acc > max_acc:
                    max_acc = acc
                if acc < min_acc:
                    min_acc = acc

            max_vel = np.amax(arr_data[np.where(arr_data[acc_start_index:acc_end_index, 2]>0.0)+acc_start_index, 1]) * 3.6
            min_vel = np.amin(arr_data[np.where(arr_data[acc_start_index:acc_end_index, 2]>0.0)+acc_start_index, 1]) * 3.6
            std_vel = np.std(arr_data[acc_start_index:acc_end_index, 1])
            mean_vel = np.mean(arr_data[acc_start_index:acc_end_index, 1])
            # print(world_id, acc_start_index, acc_end_index, arr_data[acc_start_index:acc_end_index, 1]*3.6, np.where(arr_data[acc_start_index:acc_end_index, 2]>0.0))

        else:
            min_vel = np.amin(arr_data[:, 1]) * 3.6

        # face turn count
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

        out_list.append( [
            sys.argv[1],
            intervene_type[profile.get('experiment_type')],
            world_id,
            profile.get('actor_action'),
            first_intervene_time,
            last_intervene_time,
            first_intervene_distance,
            last_intervene_distance,
            initial_vel,
            max_vel,
            min_vel,
            std_vel,
            mean_vel,
            max_acc,
            min_acc,
            intervene_vel,
            face_turn_count,
            ])

    return out_list


def saveCsv(filename, data):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def main():

    pickle_file = '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/' + sys.argv[1] + '/Town01.pickle'
    out_file = '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/' + sys.argv[1] + '/summary_rm_wrong.csv'

    with open(pickle_file, 'rb') as f:
        extracted_data = pickle.load(f)

    out_list = summarizeData(extracted_data)
    saveCsv(out_file, out_list)


if __name__ == '__main__':
    main()
