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

# gamesHowellTest
from statsmodels.stats.libqsturng import qsturng, psturng
# from hypothetical.descriptive import var
from itertools import combinations

def gamesHowellTest(df, target, hue):
    alpha = 0.05
    k = len(np.unique(df[target]))

    group_means = dict(df.groupby(hue)[target].mean())
    group_obs = dict(df.groupby(hue)[target].size())
    group_variance = dict(df.groupby(hue)[target].var())

    combs = list(combinations(np.unique(df[hue]), 2))

    group_comps = []
    mean_differences = []
    degrees_freedom = []
    t_values = []
    p_values = []
    std_err = []
    up_conf = []
    low_conf = []

    for comb in combs:
        # Mean differences of each group combination
        diff = group_means[comb[1]] - group_means[comb[0]]

        # t-value of each group combination
        t_val = np.abs(diff) / np.sqrt((group_variance[comb[0]] / group_obs[comb[0]]) +
                                       (group_variance[comb[1]] / group_obs[comb[1]]))

        # Numerator of the Welch-Satterthwaite equation
        df_num = (group_variance[comb[0]] / group_obs[comb[0]] + group_variance[comb[1]] / group_obs[comb[1]]) ** 2

        # Denominator of the Welch-Satterthwaite equation
        df_denom = ((group_variance[comb[0]] / group_obs[comb[0]]) ** 2 / (group_obs[comb[0]] - 1) +
                    (group_variance[comb[1]] / group_obs[comb[1]]) ** 2 / (group_obs[comb[1]] - 1))

        # Degrees of freedom
        df = df_num / df_denom

        # p-value of the group comparison
        p_val = psturng(t_val * np.sqrt(2), k, df)

        # Standard error of each group combination
        se = np.sqrt(0.5 * (group_variance[comb[0]] / group_obs[comb[0]] +
                            group_variance[comb[1]] / group_obs[comb[1]]))

        # Upper and lower confidence intervals
        upper_conf = diff + qsturng(1 - alpha, k, df)
        lower_conf = diff - qsturng(1 - alpha, k, df)

        # Append the computed values to their respective lists.
        mean_differences.append(diff)
        degrees_freedom.append(df)
        t_values.append(t_val)
        p_values.append(p_val)
        std_err.append(se)
        up_conf.append(upper_conf)
        low_conf.append(lower_conf)
        group_comps.append(str(comb[0]) + ' : ' + str(comb[1]))

    result_df = pd.DataFrame({'groups': group_comps,
                          'mean_difference': mean_differences,
                          'std_error': std_err,
                          't_value': t_values,
                          'p_value': p_values,
                          'upper_limit': up_conf,
                          'lower limit': low_conf})

    return result_df


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


def calcMeanAndDev(df, element, list):
    list.append([element,
                        'mean',
                         np.mean(df.query('experiment_type == "baseline"')[element]),
                         np.mean(df.query('experiment_type == "control"')[element]),
                         np.mean(df.query('experiment_type == "button"')[element]),
                         np.mean(df.query('experiment_type == "touch"')[element])
                         ])
    list.append([element,
                        'stddev',
                          np.std(df.query('experiment_type == "baseline"')[element]),
                          np.std(df.query('experiment_type == "control"')[element]),
                          np.std(df.query('experiment_type == "button"')[element]),
                          np.std(df.query('experiment_type == "touch"')[element])
                         ])
    print(list[-2],list[-1])

def saveCsv(data, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

############################
## visualize motion datas ##
############################
pickle_files = [

    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/aso/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/ichiki/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/ienaga/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/matsubara/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/ikai/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/nakakuki/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/yamamoto/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/ando/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/negi/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/kato1/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/kato2/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/sumiya/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/ito/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/isobe/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/yasuhara/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/nakatani/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/otake1/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/otake2/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/taga/Town01.pickle',
    '/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/hikosaka/Town01.pickle',

    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/okawa/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/hayashi/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/teranishi/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/kuroyanagi/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/yokoyama/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/inuzuka/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/kanayama/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/sakashita/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/saji/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/sakai/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment2/yoshioka/Town01.pickle',

    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/ichiki/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/miyazaki/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/otake/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/sakashita/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/seiya/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/taga/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/teranishi/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/yoshikawa/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_study_bag/202012experiment/yasuhara/Town01.pickle',
    # '/media/kuriatsu/SamsungKURI/master_stu2,2,dy_bag/202012experiment/nakatani/Town01.pickle',

]

out_file='/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/results/total.csv'
total_list = [['item', 'name', 'baseline', 'control', 'button', 'touch']]

fig = plt.figure()


# speed
# ax_dict_vel = {'baseline': fig.add_subplot(2,2,1), 'control': fig.add_subplot(2,2,2), 'button': fig.add_subplot(2,2,3), 'touch': fig.add_subplot(2,2,4)}
# sns.set_palette('YlGnBu',4)

# speed and intervene
grid = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[3,1])
# # ax_dict_vel = {'baseline': fig.add_subplot(grid[0,0]), 'control': fig.add_subplot(grid[0,1]), 'button': fig.add_subplot(grid[2,0]), 'touch': fig.add_subplot(grid[2,1])}
# # ax_dict_intervene = {'baseline': fig.add_subplot(grid[1,0]), 'control': fig.add_subplot(grid[1,1]), 'button': fig.add_subplot(grid[3,0]), 'touch': fig.add_subplot(grid[3,1])}
ax_vel = fig.add_subplot(grid[0])
ax_intervene = fig.add_subplot(grid[1])
## colorlize for each subject
# cmap = plt.get_cmap("tab10")
color = {'pose':'#335290ff', 'static':'#335290ff'}

vel_df = None
intervene_df = None
total_vel_mileage = []
dict_intervene = {'baseline': 0, 'control': 0, 'button': 0, 'touch': 0}
label_handle_pose = None
label_handle_static = None

for index, pickle_file in enumerate(pickle_files):
    with open(pickle_file, 'rb') as f:
        extracted_data = pickle.load(f)

    for world_id, profile in extracted_data.items():
        arr_data = np.array(profile.get('data'))[1:, :] # skip first column

        if profile.get('actor_action') in ['static', 'pose']:

            # remove mileage overlap and add them to dataframe
            splined_data = splineXAxis(arr_data[:, 4], arr_data[:, 1] * 3.6)
            if vel_df is None:
                vel_df = pd.DataFrame(splined_data, columns=['mileage', 'velocity'])
                vel_df = pd.concat([vel_df, pd.DataFrame(np.array([[profile.get('experiment_type')]] * len(splined_data)), columns=['experiment_type'])], axis=1)
            else:
                buf_df = pd.DataFrame(splined_data, columns=['mileage', 'velocity'])
                buf_df = pd.concat([buf_df, pd.DataFrame(np.array([[profile.get('experiment_type')]] * len(splined_data)), columns=['experiment_type'])], axis=1)
                vel_df = pd.concat([vel_df, buf_df], axis=0)

            if intervene_df is None:
                buf_arr = np.unique(arr_data[np.where( arr_data[:,5] != None)[0], 4])
                intervene_df = pd.DataFrame(buf_arr , columns=['mileage'], dtype=float)
                intervene_df = pd.concat([intervene_df, pd.DataFrame(np.array([[profile.get('experiment_type')]] * len(buf_arr)), columns=['experiment_type'])], axis=1)
            else:
                buf_arr = np.unique(arr_data[np.where( arr_data[:,5] != None)[0], 4])
                buf_df = pd.DataFrame(buf_arr , columns=['mileage'], dtype=float)
                buf_df = pd.concat([buf_df, pd.DataFrame(np.array([[profile.get('experiment_type')]] * len(buf_arr)), columns=['experiment_type'])], axis=1)
                intervene_df = pd.concat([intervene_df, buf_df], axis=0)

sns.lineplot(data=vel_df, x="mileage", y="velocity", hue="experiment_type", ax=ax_vel)
ax_vel.invert_xaxis()
ax_vel.set_xlim([50, -20])
ax_vel.set_ylim([0, 60])
ax_vel.set_xlabel('distance from obstacle [m]', fontsize=15)
# ax_vel.set_xlabel('', fontsize=15)
ax_vel.set_ylabel('velocity [km/h]', fontsize=15)

print(intervene_df.mileage)
sns.histplot(data=intervene_df, x='mileage', ax=ax_intervene,  binwidth=1.0, hue='experiment_type', alpha=0.2)
ax_intervene.invert_xaxis()
ax_intervene.set_xlim([50, -20])
ax_intervene.set_xlabel('distance from obstacle [m]', fontsize=15)
ax_intervene.set_ylabel('intervene \ncount', fontsize=15)
# for title, axes in ax_dict_vel.items():
#     axes.legend([label_handle_pose, label_handle_static], ['pose', 'static'])
#     axes.invert_xaxis()
#     axes.set_xlim([50, -20])
#     axes.set_ylim([0, 60])
#     axes.set_xticks([])
#     axes.set_ylabel("velocity [m/s]", fontsize=15)
#     # axes.set_xlabel("Mileage [m]", fontsize=15)
#     # axes.set_title(title, fontsize=15, y=-0.3)

# speed and intervene
# for title, axes in ax_dict_intervene.items():
#     sns.histplot(data=dict_intervene.get(title), ax=axes, binwidth=1.0, color='#7dbeb5ff')
#     axes.invert_xaxis()
#     axes.set_xlim([50, -20])
#     axes.set_xlabel("distance from obstacle [m]", fontsize=15)
#     axes.set_ylabel("intervene \ncount", fontsize=15)
#     axes.set_title(title, fontsize=15, y=-1.0)
# for axes in ax_dict_vel.values():
#     axes.legend(loc='lower right', fontsize=12)

plt.show()

exit()

# print('average_vel' ,sum(total_vel)/ len(total_vel))
# print('average_vel_mileage' ,sum(total_vel_mileage)/ len(total_vel_mileage))

###########################################
## visualize and analyze summarized data ##
###########################################

summary_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/results2/summary.csv')
avoid_deceleration_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/results2/deceleration_20km.csv')
# summary_intervene_count_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/results2/intervene_count.csv')
# face_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/results2/face.csv')
nasa_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/results2/nasa-tlx.csv')
accuracy_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/results/accuracy_summary.csv')
# accuracy_raw_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/results2/accuracy.csv')
time_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/results2/time.csv')
rank_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/results2/rank.csv')

fig = plt.figure()
ax = fig.add_subplot(111)
# hatches = ['/', '+', 'x', '.']
sns.set_palette('YlGnBu',4)


print('time')
_, p = stats.levene(time_df.query('experiment_type == "baseline"').time, time_df.query('experiment_type == "control"').time, time_df.query('experiment_type == "button"').time, time_df.query('experiment_type == "touch"').time, center='median')
print('levene-first time', p)
multicomp_result = multicomp.MultiComparison(time_df['time'], time_df['experiment_type'])
calcMeanAndDev(time_df, 'time', total_list)
print(multicomp_result.tukeyhsd().summary())
axes = sns.boxplot(x='experiment_type', y='time', data=time_df,showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
axes.set_ylim([0, 100])
## add hatch
# for i, bar in enumerate(axes.artists):
#     bar.set_hatch(hatches[i])
# addAnotation(plt, 1, 3, time_df.query('experiment_type == ["button", "touch"]').time.max(), 2, -2, '*', 'k')
plt.show()

print('first_intervene_time')
axes = sns.boxplot(x='experiment_type', y='first_intervene_time', data=summary_df, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
_, p = stats.levene(summary_df.query('experiment_type == "baseline"').first_intervene_time, summary_df.query('experiment_type == "control"').first_intervene_time, summary_df.query('experiment_type == "button"').first_intervene_time, summary_df.query('experiment_type == "touch"').first_intervene_time, center='median')
print('levene-first intervenetime', p)
multicomp_result = multicomp.MultiComparison(summary_df['first_intervene_time'], summary_df['experiment_type'])
calcMeanAndDev(summary_df, 'first_intervene_time', total_list)
print(multicomp_result.tukeyhsd().summary())
addAnotation(plt, 2, 3, 20, 0.5, 0, '*', 'k')
# axes.set_ylim([0, 9])
plt.show()

print('last_intervene_time')
axes = sns.boxplot(x='experiment_type', y='last_intervene_time', data=summary_df, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
_, p = stats.levene(summary_df.query('experiment_type == "baseline"').last_intervene_time, summary_df.query('experiment_type == "control"').last_intervene_time, summary_df.query('experiment_type == "button"').last_intervene_time, summary_df.query('experiment_type == "touch"').last_intervene_time, center='median')
print('levene-first last_intervene_time', p)
multicomp_result = multicomp.MultiComparison(summary_df['last_intervene_time'], summary_df['experiment_type'])
calcMeanAndDev(summary_df, 'last_intervene_time', total_list)
print(multicomp_result.tukeyhsd().summary())
# addAnotation(plt, 1, 3, 8, 0.5, 0, '*', 'k')
# axes.set_ylim([0, 14])
plt.show()

print('first_intervene_distance')
axes = sns.boxplot(x='experiment_type', y='first_intervene_distance', data=summary_df, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
_, p = stats.levene(summary_df.query('experiment_type == "baseline"').first_intervene_distance, summary_df.query('experiment_type == "control"').first_intervene_distance, summary_df.query('experiment_type == "button"').first_intervene_distance, summary_df.query('experiment_type == "touch"').first_intervene_distance, center='median')
print('levene-first first_intervene_distance', p)
multicomp_result = multicomp.MultiComparison(summary_df['first_intervene_distance'], summary_df['experiment_type'])
calcMeanAndDev(summary_df, 'first_intervene_distance', total_list)
print(multicomp_result.tukeyhsd().summary())
# addAnotation(plt, 0, 2, 73, 1.0, 0, '*', 'k')
# addAnotation(plt, 0, 3, 73, 1.0, 5, '*', 'k')
# addAnotation(plt, 1, 2, 73, 1.0, 10, '*', 'k')
# addAnotation(plt, 1, 3, 73, 1.0, 15, '*', 'k')
axes.set_ylim([0, 95])
plt.show()

print('last_intervene_distance')
axes = sns.boxplot(x='experiment_type', y='last_intervene_distance', data=summary_df, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
_, p = stats.levene(summary_df.query('experiment_type == "baseline"').last_intervene_distance, summary_df.query('experiment_type == "control"').last_intervene_distance, summary_df.query('experiment_type == "button"').last_intervene_distance, summary_df.query('experiment_type == "touch"').last_intervene_distance, center='median')
print('levene-first last_intervene_distance', p)
multicomp_result = multicomp.MultiComparison(summary_df['last_intervene_distance'], summary_df['experiment_type'])
calcMeanAndDev(summary_df, 'last_intervene_distance', total_list)
print(multicomp_result.tukeyhsd().summary())
# axes.set_ylim([-5, 40])
plt.show()

print('max_vel')
# sns.barplot(x='experiment_type', y='max_vel', data=summary_df)
axes = sns.boxplot(x='experiment_type', y='max_vel', data=summary_df, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
_, p = stats.levene(summary_df.query('experiment_type == "baseline"').max_vel, summary_df.query('experiment_type == "control"').max_vel, summary_df.query('experiment_type == "button"').max_vel, summary_df.query('experiment_type == "touch"').max_vel, center='median')
print('levene-first max_vel', p)
gamesHowellTest(summary_df, 'max_vel', 'experiment_type')
# multicomp_result = multicomp.MultiComparison(summary_df['max_vel'], summary_df['experiment_type'])
addAnotation(plt, 0, 2, 56, 1, 0, '*', 'k')
addAnotation(plt, 0, 3, 56, 1, 3, '*', 'k')
addAnotation(plt, 1, 2, 56, 1, 6, '*', 'k')
addAnotation(plt, 1, 3, 56, 1, 9, '*', 'k')
calcMeanAndDev(summary_df, 'max_vel', total_list)
print(multicomp_result.tukeyhsd().summary())
axes.set_ylim([0, 80])
plt.show()

print('min_vel')
axes = sns.boxplot(x='experiment_type', y='min_vel', data=summary_df, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
_, p = stats.levene(summary_df.query('experiment_type == "baseline"').min_vel, summary_df.query('experiment_type == "control"').min_vel, summary_df.query('experiment_type == "button"').min_vel, summary_df.query('experiment_type == "touch"').min_vel, center='median')
print('levene-first min_vel', p)
# addAnotation(plt, 0, 2, 40, 1.0, 0, '*', 'k')
# addAnotation(plt, 0, 2, 11, 0.5, 0, '**', 'k')
calcMeanAndDev(summary_df, 'min_vel', total_list)
# multicomp_result = multicomp.MultiComparison(summary_df['min_vel'], summary_df['experiment_type'])
# print(multicomp_result.tukeyhsd().summary())
print(gamesHowellTest(summary_df, 'min_vel', 'experiment_type'))
axes.set_ylim([0, 60])
plt.show()

print('std_vel')
axes = sns.boxplot(x='experiment_type', y='std_vel', data=summary_df, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
_, p = stats.levene(summary_df.query('experiment_type == "baseline"').std_vel, summary_df.query('experiment_type == "control"').std_vel, summary_df.query('experiment_type == "button"').std_vel, summary_df.query('experiment_type == "touch"').std_vel, center='median')
print('levene-first std_vel', p)
calcMeanAndDev(summary_df, 'std_vel', total_list)
multicomp_result = multicomp.MultiComparison(summary_df['std_vel'], summary_df['experiment_type'])
print(multicomp_result.tukeyhsd().summary())
# print(gamesHowellTest(summary_df, 'std_vel', 'experiment_type'))
axes.set_ylim([0, 7])
plt.show()

print('avoid_unnecessary_intervention_rate')
melted_df = pd.melt(accuracy_df, id_vars=accuracy_df.columns.values[:1], var_name='experiment_type', value_name='avoid_unnecessary_intervention_rate')
axes = sns.barplot(x='experiment_type', y='avoid_unnecessary_intervention_rate', data=melted_df)
multicomp_result = multicomp.MultiComparison(melted_df['avoid_unnecessary_intervention_rate'], melted_df['experiment_type'])
_, p = stats.levene(melted_df.query('experiment_type == "baseline"').avoid_unnecessary_intervention_rate, melted_df.query('experiment_type == "control"').avoid_unnecessary_intervention_rate, melted_df.query('experiment_type == "button"').avoid_unnecessary_intervention_rate, melted_df.query('experiment_type == "touch"').avoid_unnecessary_intervention_rate, center='median')
print('levene-first avoid_unnecessary_intervention_rate', p)
calcMeanAndDev(melted_df, 'avoid_unnecessary_intervention_rate', total_list)
print(multicomp_result.tukeyhsd().summary())
axes.set_ylim([0, 1])
plt.show()

print('avoid_decceleration_rate')
melted_df = pd.melt(avoid_deceleration_df, id_vars=avoid_deceleration_df.columns.values[:1], var_name='experiment_type', value_name='decceleration_rate')
axes = sns.barplot(x='experiment_type', y='decceleration_rate', data=melted_df)
# addAnotation(plt, 0, 1, 0.9, 0.01, 0, '*', 'k')
# addAnotation(plt, 0, 2, 0.9, 0.01, 0.03, '*', 'k')
# addAnotation(plt, 0, 3, 0.9, 0.01, 0.06, '*', 'k')
_, p = stats.levene(melted_df.query('experiment_type == "baseline"').decceleration_rate, melted_df.query('experiment_type == "control"').decceleration_rate, melted_df.query('experiment_type == "button"').decceleration_rate, melted_df.query('experiment_type == "touch"').decceleration_rate, center='median')
print('levene-first decceleration_rate', p)
multicomp_result = multicomp.MultiComparison(melted_df['decceleration_rate'], melted_df['experiment_type'])
calcMeanAndDev(melted_df, 'decceleration_rate', total_list)
print(multicomp_result.tukeyhsd().summary())
axes.set_ylim([0, 1])
plt.show()

# print('intervene_count')
# melted_df = pd.melt(summary_intervene_count_df, id_vars=summary_intervene_count_df.columns.values[:1], var_name='experiment_type', value_name='intervene_count_average')
# sns.barplot(x='experiment_type', y='intervene_count_average', data=melted_df)
# plt.show()


# judge normally distributed or not
# _, p = stats.shapiro(accuracy_raw_df.query('is_correct == 1').first_intervene_time)
# print('shapiro-first_intervene_time-correct', p)
# _, p = stats.shapiro(accuracy_raw_df.query('is_correct == 0').first_intervene_time)
# print('shapiro-first_intervene_time-false', p)
# _, p = stats.shapiro(accuracy_raw_df.query('is_correct == 1').intervene_distance)
# print('shapiro-intervene_distance-correct', p)
# _, p = stats.shapiro(accuracy_raw_df.query('is_correct == 0').intervene_distance)
# print('shapiro-intervene_distance-false', p)

# if normal (even if not so...) judge equal variance
# _, p = stats.levene(accuracy_raw_df.query('is_correct == 1').first_intervene_time, accuracy_raw_df.query('is_correct == 0').first_intervene_time, center='median')
# print('levene-first_intervene_time', p)
# _, p = stats.levene(accuracy_raw_df.query('is_correct == 1').intervene_distance, accuracy_raw_df.query('is_correct == 0').intervene_distance, center='median')
# print('levene-intervene_distance', p)

# finally do t-test and visualize
# result = stats.ttest_ind(accuracy_raw_df.query('is_correct == 1').first_intervene_time, accuracy_raw_df.query('is_correct == 0').first_intervene_time, equal_var=True)
# print('ttest accuracy-time', result)
# sns.barplot(x='is_correct', y='first_intervene_time', data=accuracy_raw_df)
# addAnotation(plt, 0, 1, 16, 1, 0, '*', 'k')
# plt.show()
# result = stats.ttest_ind(accuracy_raw_df.query('is_correct == 1').intervene_distance, accuracy_raw_df.query('is_correct == 0').intervene_distance, equal_var=True)
# print('ttest accuracy-distance', result)
# sns.barplot(x='is_correct', y='intervene_distance', data=accuracy_raw_df)
# addAnotation(plt, 0, 1, 40, 2, 0, '*', 'k')
# plt.show()

print('ranking')
melted_df = pd.melt(rank_df, id_vars=rank_df.columns.values[:1], var_name='experiment_type', value_name='ranking')
calcMeanAndDev(melted_df, 'ranking', total_list)
_, p = stats.levene(melted_df.query('experiment_type == "baseline"').ranking, melted_df.query('experiment_type == "control"').ranking, melted_df.query('experiment_type == "button"').ranking, melted_df.query('experiment_type == "touch"').ranking, center='median')
print('levene-first rank', p)
multicomp_result = multicomp.MultiComparison(melted_df['ranking'], melted_df['experiment_type'])
print(multicomp_result.tukeyhsd().summary())
axes = sns.boxplot(x='experiment_type', y='ranking', data=melted_df, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
axes.set_ylim([4,1])
addAnotation(plt, 0, 2, 2, -0.1, 0, '*', 'k')
addAnotation(plt, 0, 3, 2, -0.1, 0.3, '*', 'k')
addAnotation(plt, 1, 2, 2, -0.1, 0.6, '*', 'k')
addAnotation(plt, 1, 3, 2, -0.1, 0.9, '*', 'k')
plt.show()

# sns.barplot(x='experiment_type', y='count', data=face_df)
# print('face count')
# print(
# np.mean(face_df.query('experiment_type == "control"')['count']),
# np.mean(face_df.query('experiment_type == "button"')['count']),
# np.mean(face_df.query('experiment_type == "touch"')['count']),
# )
# print(
# np.std(face_df.query('experiment_type == "control"')['count']),
# np.std(face_df.query('experiment_type == "button"')['count']),
# np.std(face_df.query('experiment_type == "touch"')['count']),
# )

# multicomp_result = multicomp.MultiComparison(face_df['count'], face_df['experiment_type'])
# print(multicomp_result.tukeyhsd().summary())
# plt.show()

#### nasa-tlx ####
for item in ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'entire']:
    print(item)
    _, p = stats.levene(nasa_df.query('experiment_type == "baseline"')[item], nasa_df.query('experiment_type == "control"')[item], nasa_df.query('experiment_type == "button"')[item], nasa_df.query('experiment_type == "touch"')[item], center='median')
    print('levene-first' + item, p)
    multicomp_result = multicomp.MultiComparison(nasa_df[item], nasa_df['experiment_type'])
    print(multicomp_result.tukeyhsd().summary())
    calcMeanAndDev(nasa_df, item, total_list)

melted_df = pd.melt(nasa_df, id_vars=nasa_df.columns.values[:2], var_name="args", value_name="value")
# plot = sns.boxplot(x='args', y="value", hue="experiment_type", data=melted_df,showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
plot = sns.barplot(x='args', y="value", hue="experiment_type", data=melted_df)
plot.legend(loc='lower left', bbox_to_anchor=(1.01, 0))
addAnotation(plt, -0.3, 0.1, 8.0, 0.1, 0, '*', 'k')
addAnotation(plt, -0.3, 0.3, 8.0, 0.1, 0.5, '*', 'k')
addAnotation(plt, 0.7, 1.1, 8.0, 0.1, 0, '*', 'k')
addAnotation(plt, 0.7, 1.3, 8.0, 0.1, 0.5, '*', 'k')
addAnotation(plt, 0.9, 1.1, 8.0, 0.1, -0.5, '*', 'k')
addAnotation(plt, 0.9, 1.3, 8.0, 0.1, -1.0, '*', 'k')
# addAnotation(plt, 5.7, 6.1, 8.0, 0.1, 0, '*', 'k')
# addAnotation(plt, 5.7, 6.3, 8.0, 0.1, 0.5, '*', 'k')
# addAnotation(plt, 5.9, 6.1, 8.0, 0.1, 1.0, '*', 'k')
axes.set_ylim([0, 10])

plt.show()

saveCsv(total_list, out_file)
###########
## anova ##
###########
# model = ols('time ~ C(experiment_type) + C(scenario) + C(subject)', data=time_df).fit()
# print(model.summary())
# anova = sm.stats.anova_lm(model, typ=2)
# print(anova)
