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
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats import multicomp
from statsmodels.stats.multitest import multipletests
summary_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/Town01_summalize.csv')
time_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/aso_time_Town01.csv')


profile_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/aso/Town01_motion_profile.csv')
profile_df = profile_df.set_index('experiment_type')
# mileage_vel = profile_df.loc[['touch'], ['world_id', 'mileage', 'velocity']]

time_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/time.csv')
toptime_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/toptime.csv')
nasa_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/nasa-tlx.csv')


fig = plt.figure()
ax = fig.add_subplot(111)

for profile in profile_df.loc[['control'], ['mileage', 'intervene', 'world_id']]:

    print(profile)
    if profile[5]:
        ax.fill(np.array([profile[4], profile[4] + 0.1, profile[4] + 0.1, profile[4]]), np.array([0, 0, 40, 40], alpha=0.1))

sns.lineplot(x='mileage', y='velocity', data=profile_df.loc[['control'], ['world_id', 'mileage', 'velocity']], hue='world_id', alpha=0.5, palette="muted")
plt.show()
sns.lineplot(x='mileage', y='velocity', data=profile_df.loc[['ui'], ['world_id', 'mileage', 'velocity']], hue='world_id', alpha=0.5,  palette="muted")
# plt.show()
sns.lineplot(x='mileage', y='velocity', data=profile_df.loc[['button'], ['world_id', 'mileage', 'velocity']], hue='world_id', alpha=0.5,  palette="muted")
# plt.show()
sns.lineplot(x='mileage', y='velocity', data=profile_df.loc[['touch'], ['world_id', 'mileage', 'velocity']], hue='world_id', alpha=0.5,  palette="muted")
# plt.show()

sns.barplot(x='experiment_type', y='time', data=time_df)
# plt.show()

sns.barplot(x='experiment_type', y='intervene_time', data=summary_df)
# plt.show()
sns.barplot(x='experiment_type', y='distance to stopline', data=summary_df)
# plt.show()

accuracy_df = pd.DataFrame(data=[[5.0/6.0, 1.0, 1.0, 1.0]], columns=['control', 'ui', 'button', 'touch'])
sns.barplot(data=accuracy_df)
# plt.show()


##### time graph #####
sns.barplot(x='subject', y="time",data=time_df)
plt.show()

model = ols('time ~ C(experiment_type) + C(scenario) + C(subject)', data=time_df).fit()
print(model.summary())

anova = sm.stats.anova_lm(model, typ=2)
print(anova)
time_df
multicomp_result = multicomp.MultiComparison(time_df['time'], time_df['experiment_type'])
multicomp_result.tukeyhsd().summary()

#### nasa-tlx ####
melted_df = pd.melt(nasa_df, id_vars=nasa_df.columns.values[:2], var_name="args", value_name="value")
sns.barplot(x='args', y="value", hue="type", data=melted_df)
