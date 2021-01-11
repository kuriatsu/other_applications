#!/usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from scipy import stats
# from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats import multicomp
from statsmodels.stats.multitest import multipletests
import pickle

def splineXAxis(x_arr, y_arr):
    last_x = x_arr[0]
    y_values = []
    new_data = []
    for index, x in enumerate(x_arr[1:]):
        # print(x)
        if x == last_x:
            y_values.append(y_arr[index+1])
        else:
            y_values.append(y_arr[index+1])
            if not len(y_values):
                y_values = []
                last_x = x
            else:
                # new_data.append([last_x, min(y_values)])
                new_data.append([last_x, sum(y_values)/len(y_values)])
                last_x = x
                y_values = []
    y_values.append(y_arr[index+1])
    if not len(y_values):
        y_values = []
        last_x = x
    else:
        new_data.append([last_x, sum(y_values)/len(y_values)])
        last_x = x

    # print(np.array(new_data))
    return np.array(new_data)

def addAnotation(plt_obj, x1, x2, y, hight, h_offset, text, color):
    plt_obj.plot([x1, x1, x2, x2], [y+h_offset, y+hight+h_offset, y+hight+h_offset, y+h_offset], lw=1.5, c=color)
    plt_obj.text((x1+x2)*0.5, y+hight+h_offset, text, ha='center', va='bottom', color=color)


############################
## visualize motion datas ##
############################
pickle_files = [
    '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/nakatani/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/yoshikawa/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/yasuhara/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/sakashita/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/otake/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/seiya/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/taga/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/miyazaki/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/teranishi/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/ichiki/Town01.pickle',
]

fig = plt.figure()
ax_dict = {'control': fig.add_subplot(2,2,1), 'ui': fig.add_subplot(2,2,2), 'button': fig.add_subplot(2,2,3), 'touch': fig.add_subplot(2,2,4)}
cmap = plt.get_cmap("tab10")
for title, axes in ax_dict.items():
    axes.invert_xaxis()
    axes.set_xlim([50, -20])
    axes.set_ylim([0, 60])
    axes.set_xlabel("Mileage [m]", fontsize=15)
    axes.set_ylabel("Velocity [m/s]", fontsize=15)
    axes.set_title(title, fontsize=15, y=-0.3)


total_vel = []
total_vel_mileage = []

for index, pickle_file in enumerate(pickle_files):
    with open(pickle_file, 'rb') as f:
        extracted_data = pickle.load(f)

    for world_id, profile in extracted_data.items():
        arr_data = np.array(profile.get('data'))[1:, :] # skip first column
        # total_vel += arr_data[:,1]
        total_vel += arr_data[:, 1].tolist()
        # print(total_vel)
        if profile.get('actor_action') in ['static', 'pose']:
            # ax_dict.get(profile.get('experiment_type')).plot(arr_data[:, 4], arr_data[:, 1]*3.6, color=cmap(index%10), alpha=0.5, label='subject_'+str(index))

            splined_data = splineXAxis(arr_data[:, 4], arr_data[:, 1])
            ax_dict.get(profile.get('experiment_type')).plot(splined_data[:,0], splined_data[:,1]*3.6, color=cmap(index%10), alpha=0.5, label='subject_'+str(index))
            total_vel_mileage += splined_data[:,1].tolist()
# for axes in ax_dict.values():
#     axes.legend(loc='lower right', fontsize=12)
plt.show()
print('average_vel' ,sum(total_vel)/ len(total_vel))
print('average_vel_mileage' ,sum(total_vel_mileage)/ len(total_vel_mileage))

###########################################
## visualize and analyze summarized data ##
###########################################

summary_raw_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/results/summary.csv')
summary_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/results/summary_summary.csv')
summary_intervene_count_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/results/summary_intervene_count.csv')
face_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/results/face.csv')
nasa_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/results/nasa-tlx.csv')
accuracy_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/results/accuracy_summary.csv')
accuracy_raw_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/results/accuracy.csv')
time_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/results/time.csv')

fig = plt.figure()
ax = fig.add_subplot(111)

sns.barplot(x='experiment_type', y='time', data=time_df)
multicomp_result = multicomp.MultiComparison(time_df['time'], time_df['experiment_type'])
print('time')
print(multicomp_result.tukeyhsd().summary())
addAnotation(plt, 1, 3, time_df.query('experiment_type == ["button", "touch"]').time.max(), 2, -2, '*', 'k')
plt.show()

sns.barplot(x='experiment_type', y='intervene_time', data=summary_raw_df)
multicomp_result = multicomp.MultiComparison(summary_raw_df['intervene_time'], summary_raw_df['experiment_type'])
print('intervene_time')
print(multicomp_result.tukeyhsd().summary())
plt.show()

sns.barplot(x='experiment_type', y='distance_to_stopline', data=summary_raw_df)
multicomp_result = multicomp.MultiComparison(summary_raw_df['distance_to_stopline'], summary_raw_df['experiment_type'])
print('distance_to_stopline')
print(multicomp_result.tukeyhsd().summary())
plt.show()

sns.barplot(x='experiment_type', y='max_vel', data=summary_raw_df)
multicomp_result = multicomp.MultiComparison(summary_raw_df['max_vel'], summary_raw_df['experiment_type'])
addAnotation(plt, 0, 1, 40, 1, 0, '*', 'k')
addAnotation(plt, 0, 3, 43, 1, 0, '**', 'k')
addAnotation(plt, 1, 2, 46, 1, 0, '***', 'k')
addAnotation(plt, 1, 3, 49, 1, 0, '****', 'k')
print('max_vel')
print(multicomp_result.tukeyhsd().summary())
plt.show()

sns.barplot(x='experiment_type', y='min_vel', data=summary_raw_df)
multicomp_result = multicomp.MultiComparison(summary_raw_df['min_vel'], summary_raw_df['experiment_type'])
addAnotation(plt, 0, 1, 9, 0.5, 0, '*', 'k')
addAnotation(plt, 0, 2, 11, 0.5, 0, '**', 'k')
print('min_vel')
print(multicomp_result.tukeyhsd().summary())
plt.show()

melted_df = pd.melt(accuracy_df, id_vars=accuracy_df.columns.values[:1], var_name='experiment_type', value_name='accuracy')
sns.barplot(x='experiment_type', y='accuracy', data=melted_df)
multicomp_result = multicomp.MultiComparison(melted_df['accuracy'], melted_df['experiment_type'])
print('accuracy')
print(multicomp_result.tukeyhsd().summary())
plt.show()

melted_df = pd.melt(summary_df, id_vars=summary_df.columns.values[:1], var_name='experiment_type', value_name='deccelerated_time_rate')
sns.barplot(x='experiment_type', y='deccelerated_time_rate', data=melted_df)
addAnotation(plt, 0, 1, 0.8, 0.05, 0, '*', 'k')
addAnotation(plt, 0, 2, 0.9, 0.05, 0, '**', 'k')
axes.set_ylim([0, 1.0])
multicomp_result = multicomp.MultiComparison(melted_df['deccelerated_time_rate'], melted_df['experiment_type'])
print('deccelerated_time_rate')
print(multicomp_result.tukeyhsd().summary())
plt.show()

melted_df = pd.melt(summary_intervene_count_df, id_vars=summary_intervene_count_df.columns.values[:1], var_name='experiment_type', value_name='intervene_count_average')
sns.barplot(x='experiment_type', y='intervene_count_average', data=melted_df)
plt.show()


# judge normally distributed or not
_, p = stats.shapiro(accuracy_raw_df.query('is_correct == 1').intervene_time)
print('shapiro-intervene_time-correct', p)
_, p = stats.shapiro(accuracy_raw_df.query('is_correct == 0').intervene_time)
print('shapiro-intervene_time-false', p)
_, p = stats.shapiro(accuracy_raw_df.query('is_correct == 1').intervene_distance)
print('shapiro-intervene_distance-correct', p)
_, p = stats.shapiro(accuracy_raw_df.query('is_correct == 0').intervene_distance)
print('shapiro-intervene_distance-false', p)

# if normal (even if not so...) judge equal variance
_, p = stats.levene(accuracy_raw_df.query('is_correct == 1').intervene_time, accuracy_raw_df.query('is_correct == 0').intervene_time, center='median')
print('levene-intervene_time', p)
_, p = stats.levene(accuracy_raw_df.query('is_correct == 1').intervene_distance, accuracy_raw_df.query('is_correct == 0').intervene_distance, center='median')
print('levene-intervene_distance', p)

# finally do t-test and visualize
result = stats.ttest_ind(accuracy_raw_df.query('is_correct == 1').intervene_time, accuracy_raw_df.query('is_correct == 0').intervene_time, equal_var=True)
print('ttest accuracy-time', result)
sns.barplot(x='is_correct', y='intervene_time', data=accuracy_raw_df)
addAnotation(plt, 0, 1, 16, 1, 0, '*', 'k')
plt.show()
result = stats.ttest_ind(accuracy_raw_df.query('is_correct == 1').intervene_distance, accuracy_raw_df.query('is_correct == 0').intervene_distance, equal_var=True)
print('ttest accuracy-distance', result)
sns.barplot(x='is_correct', y='intervene_distance', data=accuracy_raw_df)
addAnotation(plt, 0, 1, 40, 2, 0, '*', 'k')
plt.show()

#### nasa-tlx ####
multicomp_result = multicomp.MultiComparison(nasa_df['mental'], nasa_df['type'])
print('mental')
print(multicomp_result.tukeyhsd().summary())
multicomp_result = multicomp.MultiComparison(nasa_df['physical'], nasa_df['type'])
print('physical')
print(multicomp_result.tukeyhsd().summary())
multicomp_result = multicomp.MultiComparison(nasa_df['temporal'], nasa_df['type'])
print('temporal')
print(multicomp_result.tukeyhsd().summary())
multicomp_result = multicomp.MultiComparison(nasa_df['performance'], nasa_df['type'])
print('performance')
print(multicomp_result.tukeyhsd().summary())
multicomp_result = multicomp.MultiComparison(nasa_df['effort'], nasa_df['type'])
print('effort')
print(multicomp_result.tukeyhsd().summary())
multicomp_result = multicomp.MultiComparison(nasa_df['frustration'], nasa_df['type'])
print('frustration')
print(multicomp_result.tukeyhsd().summary())
multicomp_result = multicomp.MultiComparison(nasa_df['entire'], nasa_df['type'])
print('entire')
print(multicomp_result.tukeyhsd().summary())
melted_df = pd.melt(nasa_df, id_vars=nasa_df.columns.values[:2], var_name="args", value_name="value")
sns.barplot(x='args', y="value", hue="type", data=melted_df)
addAnotation(plt, -0.3, -0.1, 7, 0.3, 0, '*', 'k')
addAnotation(plt, -0.3, 0.1, 7, 0.3, 0.5, '**', 'k')
addAnotation(plt, 0.7, 1.1, 7, 0.3, 0, '*', 'k')
plt.show()

###########
## anova ##
###########
# model = ols('time ~ C(experiment_type) + C(scenario) + C(subject)', data=time_df).fit()
# print(model.summary())
# anova = sm.stats.anova_lm(model, typ=2)
# print(anova)
