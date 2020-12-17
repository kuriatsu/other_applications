#! /usr/bin/python
# -*- coding:utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/Town04_data.pickle', 'rb') as f:
    data = pickle.load(f)

for world_id, profile in data.items():

    if profile.get('actor_action') in ['static', 'pose']:
        print(profile.get('experiment_type'))
        for profile_data in profile.get('data'):
            if profile.get('experiment_type') == 'touch' and profile_data[5] == 'touch':
                intervene_type = profile_data[0]
                print('touch' + str(intervene_type))
                break

            if profile.get('experiment_type') == 'control' and profile_data[5] == 'throttle':
                intervene_type = profile_data[0]
                print('control' + str(intervene_type))
                break

            if profile.get('experiment_type') == 'ui' and profile_data[5] == 'throttle':
                intervene_type = profile_data[0]
                print('ui' + str(intervene_type))
                break

            if profile.get('experiment_type') == 'button' and profile_data[5] == 'button':
                intervene_type = profile_data[0]
                print('button' + str(intervene_type))
                break
