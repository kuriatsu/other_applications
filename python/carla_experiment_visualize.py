#!/usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
# from scipy import stats
import statsmodels.api as sm

summary_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/Town01_summalize.csv')
time_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/aso_time_Town01.csv')


profile_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/Town01_motion_profile.csv')
profile_df = profile_df.set_index('experiment_type')
# mileage_vel = profile_df.loc[['touch'], ['world_id', 'mileage', 'velocity']]



fig = plt.figure()
sns.lineplot(x='mileage', y='velocity', data=profile_df.loc[['control'], ['world_id', 'mileage', 'velocity']], hue='world_id', alpha=0.5, palette="muted")
plt.show()
sns.lineplot(x='mileage', y='velocity', data=profile_df.loc[['ui'], ['world_id', 'mileage', 'velocity']], hue='world_id', alpha=0.5,  palette="muted")
plt.show()
sns.lineplot(x='mileage', y='velocity', data=profile_df.loc[['button'], ['world_id', 'mileage', 'velocity']], hue='world_id', alpha=0.5,  palette="muted")
plt.show()
sns.lineplot(x='mileage', y='velocity', data=profile_df.loc[['touch'], ['world_id', 'mileage', 'velocity']], hue='world_id', alpha=0.5,  palette="muted")
plt.show()

sns.barplot(x='experiment_type', y='time', data=time_df)
plt.show()

sns.barplot(x='experiment_type', y='intervene_time', data=summary_df)
plt.show()
sns.barplot(x='experiment_type', y='distance to stopline', data=summary_df)
plt.show()
