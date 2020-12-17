#! /usr/bin/python
# -*- coding:utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv

analyzed_data = [['experiment_type', 'world_id', 'intervene_time', 'distance to stopline']]

with open('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/Town01_data.pickle', 'rb') as f:
    data = pickle.load(f)

for world_id, profile in data.items():
    # if world_id == 232:
    #     print([col[5] for col in profile.get('data')])
    if profile.get('actor_action') in ['static', 'pose']:
        print('experiment_type: ' + profile.get('experiment_type') + ' world_id: ' + str(world_id))
        for index, profile_data in enumerate(profile.get('data')):

            if profile_data[5] is None:
                continue

            if index == 0:
                continue

            if profile.get('experiment_type') == 'touch' and profile_data[5] == 'touch':
                intervene_time = profile_data[0]
                analyzed_data.append([profile.get('experiment_type'), world_id, intervene_time, profile_data[3]])
                # print('touch:' + str(intervene_time))
                # print('mileage: ' + str(profile_data[3]))
                break

            if profile.get('experiment_type') == 'control' and 'throttle' in profile_data[5]:
                intervene_time = profile_data[0]
                analyzed_data.append([profile.get('experiment_type'), world_id, intervene_time, profile_data[3]])
                # print('control:' + str(intervene_time))
                # print('mileage: ' + str(profile_data[3]))
                break

            if profile.get('experiment_type') == 'ui' and 'throttle' in profile_data[5]:
                intervene_time = profile_data[0]
                analyzed_data.append([profile.get('experiment_type'), world_id, intervene_time, profile_data[3]])
                # print('ui:' + str(intervene_time))
                # print('mileage: ' + str(profile_data[3]))
                break

            if profile.get('experiment_type') == 'button' and profile_data[5] in ['button', 'touch']:
                intervene_time = profile_data[0]
                analyzed_data.append([profile.get('experiment_type'), world_id, intervene_time, profile_data[3]])
                # print('touch:' + str(intervene_time))
                # print('mileage: ' + str(profile_data[3]))
                break

print(analyzed_data)
with open('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/Town01_summalize.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(analyzed_data)

motion_profile = [['experiment_type', 'world_id', 'time', 'velocity', 'mileage', 'intervene']]

for world_id, profile in data.items():
    # if world_id == 232:
    #     print([col[5] for col in profile.get('data')])
    if profile.get('actor_action') in ['static', 'pose']:
        print('experiment_type: ' + profile.get('experiment_type') + ' world_id: ' + str(world_id))
        for index, profile_data in enumerate(profile.get('data')):

            if index == 0:
                continue

            motion_profile.append([profile.get('experiment_type'), world_id, profile_data[0], profile_data[1] * 3.6, -profile_data[2], (profile_data[5] is not None)])


print(motion_profile)
with open('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/Town01_motion_profile.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(motion_profile)
