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
import statsmodels.api as sm

###############
###acc-type###
###############
df_obj_statistic = pd.read_csv('/home/kuriatsu/DropboxKuri/data/PIE_experiment/obj_statistic.csv')
df_logistic = pd.read_csv('/home/kuriatsu/DropboxKuri/data/PIE_experiment/result_logistic.csv')
df_tlx = pd.read_csv('/home/kuriatsu/DropboxKuri/data/PIE_experiment/tlx.csv')
df_kuriatsu = pd.read_csv('/home/kuriatsu/DropboxKuri/data/PIE_experiment/kuriatsu_try.csv')

acc_df = df_obj_statistic.loc[:, ['id', 'enter_acc','touch_acc']]
acc_df.rename(columns={'enter_acc':'Enter', 'touch_acc': 'Touch'}, inplace=True)
acc_df.dropna(inplace=True)
acc_df = pd.melt(acc_df, id_vars='id', value_vars=['Enter','Touch'], var_name='acc_type', value_name='acc_value')

fig = plt.figure()
ax = sns.boxplot(x='acc_type', y='acc_value', data=acc_df, palette=['orange', 'seagreen'], showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
# ax = sns.barplot(x='acc_type', y='acc_value', data=acc_df, capsize=.2, palette=['orange', 'seagreen'])
sns.swarmplot(data=acc_df, x='acc_type', y='acc_value', color=".25", ax=ax)
ax.set_title(label='Intervene Accuracy for each method')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
fig.set_size_inches(5, 4.5, forward=True)
plt.savefig('pie_acc_inttype.png')
plt.show()

# acc_df = df_obj_statistic.loc[:, ['id', 'enter_acc','touch_acc', 'clarity']]
# fig, ax = plt.subplots(1,1)
# sns.regplot(x=acc_df.clarity, y=acc_df.enter_acc, ax = ax, label='Enter', truncate=False, color='orange')
# sns.regplot(x=acc_df.clarity, y=acc_df.touch_acc, ax = ax, label='Touch', truncate=False, color='seagreen')
# ax.set_xlabel('intention clarity [s]')
# ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy and Intention Clarity')
# ax.legend(loc='lower right')
#
# ax.tick_params(axis='x', labelsize=18)
# ax.tick_params(axis='y', labelsize=18)
#
# plt.subplots_adjust(bottom=0.15, top=0.93, left=0.18, right=0.97)
# fig.set_size_inches(5, 4.5, forward=True)
# # plt.savefig('pie_acc_clarity.png')
# plt.show()

######################
###acc-type t-stats###
######################
result = stats.ttest_ind(df_obj_statistic.enter_acc.dropna(), df_obj_statistic.touch_acc.dropna(), equal_var=True)
print('acc-intervene_type_result: ', result)
print('acc-intervene_mean Enter: ', np.mean(acc_df.query('acc_type == "Enter"').acc_value), 'Touch', np.mean(acc_df.query('acc_type == "Touch"').acc_value))


prob_df = df_obj_statistic.loc[:, ['id', 'prob_anno', 'prob', 'prob_enter', 'prob_touch']]
########################
###intention histbram###
########################
ax = sns.distplot(prob_df.prob_anno, bins=10, kde=False)
ax.set_title(label='Distribution of Pedestrian Intention')
# plt.savefig('pie_dist_int.png')
plt.show()

###############
###id - prob###
###############
# prob_sorted_index = prob_df.sort_values('prob_anno').id
# prob_df = pd.melt(prob_df, id_vars='id', value_vars=['prob_anno', 'prob', 'prob_enter', 'prob_touch'], var_name='prob_type', value_name='prob_value')
# prob_df
# ax = sns.barplot(x='id', y='prob_value', data=prob_df, hue='prob_type', capsize=.2, order=prob_sorted_index)
# ax.set_title(label='calcurated intenrion of each pedestrian in experiment')


#################################
###data_prob - experiment-prob###
#################################
prob_df.dropna(inplace=True)
for index, prob in zip(prob_df.index, prob_df.prob_anno):
    prob_df.at[index, 'Pedestrian Intention in Dataset'] = f"{int(prob * 10) * 0.1:.1f}-{int(prob * 10) * 0.1 + 0.1:.1f}"

prob_df = pd.melt(prob_df, id_vars='Pedestrian Intention in Dataset', value_vars=['prob_enter', 'prob_touch'], var_name='prob_type', value_name='prob_value')

# prob_df = prob_df.append(pd.Series([6, 'prob_enter', 0.0], index=prob_df.columns, name=len(prob_df)))
# prob_df = prob_df.append(pd.Series([6, 'prob_touch', 0.0], index=prob_df.columns, name=len(prob_df)))

ax = sns.barplot(data=prob_df, x='Pedestrian Intention in Dataset', y='prob_value', hue='prob_type', order=['0.0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.7-0.8','0.8-0.9','0.9-1.0'], palette=['orange', 'seagreen'])
ax.set_title(label='Pedestrian Intention Calcurated in Dataset and Experiment')

ax.tick_params(axis='x', labelsize=18, labelrotation=45)
ax.tick_params(axis='y', labelsize=18)

plt.savefig('pie_prob-prob.png')
plt.show()


###############
### pairplot###
###############
# df_intervene_time = df_logistic.dropna()
# df_intervene_time.drop(columns=['clarity', 'is_intervene_correct'], inplace=True)
# sns.pairplot(df_intervene_time, diag_kind='kde', hue='intervene_type')
# plt.show()

################################
### intervene_time-bariables ###
################################
df_intervene_time = df_logistic.dropna()
df_intervene_time.drop(columns=['is_intervene_correct'], inplace=True)

fig, ax = plt.subplots(1, 1)
buf = df_intervene_time.query('intervene_type == 0')
sns.regplot(x='time', y='intervene_time', data=buf, color='orange',ax=ax, label='Enter', truncate=False)
buf = df_intervene_time.query('intervene_type == 1')
sns.regplot(x='time', y='intervene_time', data=buf, color='seagreen', ax = ax, label='Touch', truncate=False)
plt.legend()
ax.set_title(label='Intervene Time and Erapsed Time')
ax.set_xlabel(xlabel='erapsed time [s]')
ax.set_ylabel(ylabel='intervene time [s]')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.subplots_adjust(bottom=0.17, top=0.93, left=0.12, right=0.95)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_inttime_time.png')
plt.show()


fig, ax = plt.subplots(1, 1)
buf = df_intervene_time.query('intervene_type == 0')
sns.regplot(x='prob', y='intervene_time', data=buf, color='orange',ax=ax, label='Enter', truncate=False)
buf = df_intervene_time.query('intervene_type == 1')
sns.regplot(x='prob', y='intervene_time', data=buf, color='seagreen', ax = ax, label='Touch', truncate=False)
plt.legend()
ax.set_title(label='Intervene Time and Pedestrian Intention')
ax.set_xlabel(xlabel='pedestrian intention')
ax.set_ylabel(ylabel='intervene time [s]')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.subplots_adjust(bottom=0.17, top=0.93, left=0.12, right=0.95)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_inttime_int.png')
plt.show()



fig, ax = plt.subplots(1, 1)
buf = df_intervene_time.query('intervene_type == 0')
sns.regplot(x='prob', y='intervene_time', data=buf, color='orange',ax=ax, label='Enter', truncate=False, order=2)
buf = df_intervene_time.query('intervene_type == 1')
sns.regplot(x='prob', y='intervene_time', data=buf, color='seagreen', ax = ax, label='Touch', truncate=False, order=2)
plt.legend()
ax.set_title(label='Intervene Time and Pedestrian Intention (order=2)')
ax.set_xlabel(xlabel='pedestrian intention')
ax.set_ylabel(ylabel='intervene time [s]')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.subplots_adjust(bottom=0.17, top=0.93, left=0.12, right=0.95)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_inttime_int_order2.png')
plt.show()


fig, ax = plt.subplots(1, 1)
buf = df_intervene_time.query('intervene_type == 0')
sns.regplot(x='clarity', y='intervene_time', data=buf, color='orange',ax=ax, label='Enter', truncate=False)
buf = df_intervene_time.query('intervene_type == 1')
sns.regplot(x='clarity', y='intervene_time', data=buf, color='seagreen', ax = ax, label='Touch', truncate=False)
plt.legend()
ax.set_title(label='Intervene Time and Intention Clarity')
ax.set_xlabel(xlabel='intention clarity')
ax.set_ylabel(ylabel='intervene time [s]')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.subplots_adjust(bottom=0.17, top=0.93, left=0.12, right=0.95)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_inttime_clarity.png')
plt.show()


fig, ax = plt.subplots(1, 1)
buf = df_intervene_time.query('intervene_type == 0')
sns.regplot(x='display_distance', y='intervene_time', data=buf, color='orange',ax=ax, label='Enter', truncate=False)
buf = df_intervene_time.query('intervene_type == 1')
sns.regplot(x='display_distance', y='intervene_time', data=buf, color='seagreen', ax = ax, label='Touch', truncate=False)
plt.legend()
ax.set_title(label='Intervene Time and Distance')
ax.set_xlabel(xlabel='distance from ego vehicle to target [m]')
ax.set_ylabel(ylabel='intervene time [s]')

ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

plt.subplots_adjust(bottom=0.17, top=0.93, left=0.12, right=0.95)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_inttime_dist.png')
plt.show()


fig, ax = plt.subplots(1, 1)
buf = df_intervene_time.query('intervene_type == 0')
sns.regplot(x='display_velocity', y='intervene_time', data=buf, color='orange',ax=ax, label='Enter', truncate=False)
buf = df_intervene_time.query('intervene_type == 1')
sns.regplot(x='display_velocity', y='intervene_time', data=buf, color='seagreen', ax = ax, label='Touch', truncate=False)
plt.legend()
ax.set_title(label='Intervene Time and Velocity')
ax.set_xlabel(xlabel='velocity of ego vehicle[m/s]')
ax.set_ylabel(ylabel='intervene time [s]')

ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

plt.subplots_adjust(bottom=0.17, top=0.93, left=0.12, right=0.95)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_inttime_vel.png')
plt.show()


fig, ax = plt.subplots(1, 1)
buf = df_intervene_time.query('intervene_type == 0')
sns.regplot(x='box_size', y='intervene_time', data=buf, color='orange',ax=ax, label='Enter', truncate=False)
buf = df_intervene_time.query('intervene_type == 1')
sns.regplot(x='box_size', y='intervene_time', data=buf, color='seagreen', ax = ax, label='Touch', truncate=False)
plt.legend()
ax.set_title(label='Intervene Time and Target Size')
ax.set_xlabel(xlabel='box size of target [px]')
ax.set_ylabel(ylabel='intervene time [s]')

ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

plt.subplots_adjust(bottom=0.17, top=0.93, left=0.12, right=0.95)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_inttime_size.png')
plt.show()

###################################
###correct-variables biorin plot###
###################################
# df_is_correct = df_logistic.drop(columns=['clarity', 'intervene_distance', 'intervene_ego_speed'])
# df_is_correct = df_is_correct.replace(0, {'intervene_type':'Enter Key', 'is_intervene_correct':'Wrong'})
# df_is_correct = df_is_correct.replace(1, {'intervene_type':'Touch', 'is_intervene_correct':'Correct'})
# fig = plt.figure(linewidth=0.0)
# # plt.xticks([])
# # plt.yticks([])
# plt.axis('off')
# plt.xlabel('Intervention Correctness')
# plt.title('Intervention Correctness vs Each Variables', y=-0.35)
# # plt.xticks('')
# ax.get_legend().remove()
# fig.subplots_adjust(wspace=0.4)
#
# ax = fig.add_subplot(1,6,1)
# violin = sns.violinplot(data=df_is_correct, x='is_intervene_correct', y='time', hue='intervene_type', ax=ax, split=False)
# ax.set_xlabel('')
# ax.set_ylabel('Elapsed Time [s]')
# ax.get_legend().remove()
# violin.set_xticklabels(violin.get_xticklabels(), rotation=45)
#
# ax = fig.add_subplot(1,6,2)
# violin = sns.violinplot(data=df_is_correct, x='is_intervene_correct', y='prob', hue='intervene_type', ax=ax, split=False)
# ax.set_xlabel('')
# ax.set_ylabel('Crossing Intention of the Pedestrian')
# ax.get_legend().remove()
# violin.set_xticklabels(violin.get_xticklabels(), rotation=45)
#
# ax = fig.add_subplot(1,6,3)
# violin = sns.violinplot(data=df_is_correct, x='is_intervene_correct', y='display_distance', hue='intervene_type', ax=ax, split=False)
# ax.set_xlabel('Intervention Correctness')
# ax.set_ylabel('Distance between Ego Vehicle and Pedestrian [m]')
# ax.get_legend().remove()
# violin.set_xticklabels(violin.get_xticklabels(), rotation=45)
#
# ax = fig.add_subplot(1,6,4)
# violin = sns.violinplot(data=df_is_correct, x='is_intervene_correct', y='display_velocity', hue='intervene_type', ax=ax, split=False)
# ax.set_xlabel('')
# ax.set_ylabel('Velocity of Ego Vehicle [m/s]')
# ax.get_legend().remove()
# violin.set_xticklabels(violin.get_xticklabels(), rotation=45)
#
# ax = fig.add_subplot(1,6,5)
# violin = sns.violinplot(data=df_is_correct, x='is_intervene_correct', y='box_size', hue='intervene_type', ax=ax, split=False)
# ax.set_xlabel('')
# ax.set_ylabel('Bounding Box of the Target Pedestrian [px]')
# ax.get_legend().remove()
# violin.set_xticklabels(violin.get_xticklabels(), rotation=45)
#
# ax = fig.add_subplot(1,6,6)
# violin = sns.violinplot(data=df_is_correct.dropna(), x='is_intervene_correct', y='intervene_time', hue='intervene_type', ax=ax, split=False)
# ax.set_xlabel('')
# ax.set_ylabel('Intervene Speed [s]')
# ax.legend(bbox_to_anchor=(1.0, -0.1))
# violin.set_xticklabels(violin.get_xticklabels(), rotation=45)
#
# plt.show()


################################
### acc-bariables ###
################################
df_is_correct = df_logistic.drop(columns=['intervene_distance', 'intervene_ego_speed', 'intervene_time', 'early'])
df_is_correct_with_int_time = df_logistic.drop(columns=['intervene_distance', 'intervene_ego_speed'])
df_is_correct.dropna(inplace=True)
df_is_correct_with_int_time.dropna(inplace=True)
df_is_correct = df_is_correct.replace(0, {'intervene_type':'Enter'})
df_is_correct = df_is_correct.replace(1, {'intervene_type':'Touch'})
df_is_correct_with_int_time = df_is_correct_with_int_time.replace(0, {'intervene_type':'Enter'})
df_is_correct_with_int_time = df_is_correct_with_int_time.replace(1, {'intervene_type':'Touch'})

blank_dict = {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}}
time_acc = {i : {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}} for i in range(int(min(df_is_correct.time) / 60), int(max(df_is_correct.time) / 60) + 1)}
prob_acc = {i : {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}} for i in [i / 10 for i in range(math.floor(min(df_is_correct.prob) * 10), math.floor(max(df_is_correct.prob) * 10) + 1, 1)]}
clarity_acc = {i : {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}} for i in [i / 100 for i in range(math.floor(min(df_is_correct.clarity) * 100), math.floor(max(df_is_correct.clarity) * 100) + 1, 1)]}
display_distance_acc = {i : {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}} for i in range(int(min(df_is_correct.display_distance) / 2) * 2, int(max(df_is_correct.display_distance) / 5  + 1) * 5, 5)}
display_velocity_acc = {i : {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}} for i in range(int(min(df_is_correct.display_velocity) / 5) * 5, int(max(df_is_correct.display_velocity) / 5  + 1) * 5, 5)}
box_size_acc = {i : {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}} for i in range(int(min(df_is_correct.box_size) * 0.1)  * 10, int(max(df_is_correct.box_size) * 0.1 + 1) * 10, 10)}
intervene_time_acc = {i : {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}} for i in [i / 10 for i in range(int(min(df_is_correct_with_int_time.intervene_time) * 10), int(max(df_is_correct_with_int_time.intervene_time) * 10) , 1)]}
early_acc = {i : {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}} for i in [i / 10 for i in range(math.floor(min(df_is_correct_with_int_time.early) * 10), math.floor(max(df_is_correct_with_int_time.early) * 10) + 1)]}

for index in df_is_correct.index:

    time = int(df_is_correct.at[index, 'time'] / 60)
    if time not in time_acc: time_acc[time] = {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}}
    time_acc[time][df_is_correct.at[index, 'intervene_type']]['total']+=1
    time_acc[time][df_is_correct.at[index, 'intervene_type']]['correct']+=df_is_correct.at[index, 'is_intervene_correct']

    prob = int(df_is_correct.at[index, 'prob'] * 10) / 10
    if prob not in prob_acc: prob_acc[prob] = {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}}
    prob_acc[prob][df_is_correct.at[index, 'intervene_type']]['total']+=1
    prob_acc[prob][df_is_correct.at[index, 'intervene_type']]['correct']+=df_is_correct.at[index, 'is_intervene_correct']

    clarity = int(df_is_correct.at[index, 'clarity'] * 100) / 100
    if clarity not in clarity_acc: clarity_acc[clarity] = {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}}
    clarity_acc[clarity][df_is_correct.at[index, 'intervene_type']]['total']+=1
    clarity_acc[clarity][df_is_correct.at[index, 'intervene_type']]['correct']+=df_is_correct.at[index, 'is_intervene_correct']

    display_distance = int(df_is_correct.at[index, 'display_distance'] / 2) * 2
    if display_distance not in display_distance_acc: display_distance_acc[display_distance] = {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}}
    display_distance_acc[display_distance][df_is_correct.at[index, 'intervene_type']]['total']+=1
    display_distance_acc[display_distance][df_is_correct.at[index, 'intervene_type']]['correct']+=df_is_correct.at[index, 'is_intervene_correct']

    display_velocity = int(df_is_correct.at[index, 'display_velocity'] / 5) * 5
    if display_velocity not in display_velocity_acc: display_velocity_acc[display_velocity] = {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}}
    display_velocity_acc[display_velocity][df_is_correct.at[index, 'intervene_type']]['total']+=1
    display_velocity_acc[display_velocity][df_is_correct.at[index, 'intervene_type']]['correct']+=df_is_correct.at[index, 'is_intervene_correct']

    box_size = int(df_is_correct.at[index, 'box_size'] / 5) * 5
    if box_size not in box_size_acc: box_size_acc[box_size] = {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}}
    box_size_acc[box_size][df_is_correct.at[index, 'intervene_type']]['total']+=1
    box_size_acc[box_size][df_is_correct.at[index, 'intervene_type']]['correct']+=df_is_correct.at[index, 'is_intervene_correct']

for index in df_is_correct_with_int_time.index:
    intervene_time = int(df_is_correct_with_int_time.at[index, 'intervene_time'] * 10) / 10 if not math.isnan(df_is_correct_with_int_time.at[index, 'intervene_time']) else math.nan
    if intervene_time not in intervene_time_acc: intervene_time_acc[intervene_time] = {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}}
    intervene_time_acc[intervene_time][df_is_correct_with_int_time.at[index, 'intervene_type']]['total']+=1
    intervene_time_acc[intervene_time][df_is_correct_with_int_time.at[index, 'intervene_type']]['correct']+=df_is_correct_with_int_time.at[index, 'is_intervene_correct']

    early = int(df_is_correct_with_int_time.at[index, 'early'] *10) /  10
    if early not in early_acc: early_acc[early] = {'Enter':{'total':0, 'correct':0}, 'Touch':{'total':0, 'correct':0}}
    early_acc[early][df_is_correct_with_int_time.at[index, 'intervene_type']]['total']+=1
    early_acc[early][df_is_correct_with_int_time.at[index, 'intervene_type']]['correct']+=df_is_correct_with_int_time.at[index, 'is_intervene_correct']

for dict in [time_acc, prob_acc, display_distance_acc, display_velocity_acc, box_size_acc, intervene_time_acc, clarity_acc, early_acc]:
    for i, val in dict.items():

        dict[i]['Enter']['acc'] = val.get('Enter').get('correct') / val.get('Enter').get('total') if val.get('Enter').get('total') != 0 else None
        dict[i]['Touch']['acc'] = val.get('Touch').get('correct') / val.get('Touch').get('total') if val.get('Touch').get('total') != 0 else None



################################################
### double plot of regplot, acc vs variables ###
################################################

fig, ax = plt.subplots(1,1)
sns.regplot(x=[i for i in intervene_time_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in intervene_time_acc.values()], dtype=float), ax = ax, label='Enter', truncate=False, color='orange')
sns.regplot(x=[i for i in intervene_time_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in intervene_time_acc.values()], dtype=float), ax = ax, label='Touch', truncate=False, color='seagreen')
ax.set_xlabel('intervene_time [s]')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy and Intervene Time')
ax.legend(loc='lower right')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_ylim(0.0, 1.0)
plt.subplots_adjust(bottom=0.15, top=0.93, left=0.18, right=0.97)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_acc_inttime.png')
plt.show()

intervene_time_acc_arr = []
for time, val in intervene_time_acc.items():
    if val.get('Enter').get('acc') is not None:
        buf = [0, time, val.get('Enter').get('acc')]
        intervene_time_acc_arr.append(buf)
    if val.get('Touch').get('acc') is not None:
        buf = [1, time, val.get('Touch').get('acc')]
        intervene_time_acc_arr.append(buf)

intervene_time_acc_arr = np.array(intervene_time_acc_arr)

model = sm.OLS(intervene_time_acc_arr[:, 2], intervene_time_acc_arr[:, :2])
result = model.fit()
print(result.summary())


fig, ax = plt.subplots(1,1)
sns.regplot(x=[i for i in early_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in early_acc.values()], dtype=float), ax = ax, label='Enter', truncate=False, color='orange')
sns.regplot(x=[i for i in early_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in early_acc.values()], dtype=float), ax = ax, label='Touch', truncate=False, color='seagreen')
ax.set_xlabel('(action time) - (intervene time) [s]')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy and (Action Time) - (Intervene Time)')
ax.legend(loc='lower right')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_ylim(0.0, 1.0)
plt.subplots_adjust(bottom=0.15, top=0.93, left=0.18, right=0.97)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_acc_inttime-actiontime.png')
plt.show()

fig, ax = plt.subplots(1,1)
# sns.regplot(x=[i for i in prob_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in prob_acc.values()], dtype=float), ax = ax, label='Enter', truncate=False, color='orange')
sns.regplot(x=[i for i in prob_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in prob_acc.values()], dtype=float), ax = ax, label='Enter', truncate=False, color='orange', order=2)
# sns.regplot(x=[i for i in prob_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in prob_acc.values()], dtype=float), ax = ax, label='Touch', truncate=False, color='seagreen')
sns.regplot(x=[i for i in prob_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in prob_acc.values()], dtype=float), ax = ax, label='Touch', truncate=False, color='seagreen', order=2)
ax.set_xlabel('pedestrian intention [s]')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy and Pedestrian Intention')
ax.legend(loc='lower right')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_ylim(0.0, 1.0)
plt.subplots_adjust(bottom=0.15, top=0.93, left=0.18, right=0.97)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_acc_prob.png')
plt.show()

fig, ax = plt.subplots(1,1)
sns.regplot(x=[i for i in clarity_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in clarity_acc.values()], dtype=float), ax = ax, label='Enter', truncate=False, color='orange')
sns.regplot(x=[i for i in clarity_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in clarity_acc.values()], dtype=float), ax = ax, label='Touch', truncate=False, color='seagreen')
ax.set_xlabel('intention clarity')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy and Intention Clarity')
ax.legend(loc='lower right')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_ylim(0.0, 1.0)
plt.subplots_adjust(bottom=0.15, top=0.93, left=0.18, right=0.97)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_acc_clarity.png')
plt.show()

fig, ax = plt.subplots(1,1)
sns.regplot(x=[i for i in display_distance_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in display_distance_acc.values()], dtype=float), ax = ax, label='Enter', truncate=False, color='orange')
sns.regplot(x=[i for i in display_distance_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in display_distance_acc.values()], dtype=float), ax = ax, label='Touch', truncate=False, color='seagreen')
ax.set_xlabel('display distance [m]')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy and Distance to Target')
ax.legend(loc='lower right')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_ylim(0.0, 1.0)
plt.subplots_adjust(bottom=0.15, top=0.93, left=0.18, right=0.97)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_acc_dist.png')
plt.show()

fig, ax = plt.subplots(1,1)
sns.regplot(x=[i for i in display_velocity_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in display_velocity_acc.values()], dtype=float), ax = ax, label='Enter', truncate=False, color='orange')
sns.regplot(x=[i for i in display_velocity_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in display_velocity_acc.values()], dtype=float), ax = ax, label='Touch', truncate=False, color='seagreen')
ax.set_xlabel('velocity of ego vehicle [m/s]')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy and Velocity')
ax.legend(loc='lower right')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_ylim(0.0, 1.0)
plt.subplots_adjust(bottom=0.15, top=0.93, left=0.18, right=0.97)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_acc_vel.png')
plt.show()

fig, ax = plt.subplots(1,1)
sns.regplot(x=[i for i in box_size_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in box_size_acc.values()], dtype=float), ax = ax, label='Enter', truncate=False, color='orange')
sns.regplot(x=[i for i in box_size_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in box_size_acc.values()], dtype=float), ax = ax, label='Touch', truncate=False, color='seagreen')
ax.set_xlabel('target size on interface[px]')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy and Target Size')
ax.legend(loc='lower right')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_ylim(0.0, 1.0)
plt.subplots_adjust(bottom=0.15, top=0.93, left=0.18, right=0.97)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_acc_size.png')
plt.show()

fig, ax = plt.subplots(1,1)
sns.regplot(x=[i for i in time_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in time_acc.values()], dtype=float), ax = ax, label='Enter', truncate=False, color='orange')
sns.regplot(x=[i for i in time_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in time_acc.values()], dtype=float), ax = ax, label='Touch', truncate=False, color='seagreen')
ax.set_xlabel('elapsed time [min]')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy and Eraplsed Time')
ax.legend(loc='lower right')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_ylim(0.0, 1.0)
plt.subplots_adjust(bottom=0.15, top=0.93, left=0.18, right=0.97)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_acc_time.png')
plt.show()


##########################################
### bar and regplot for each intervene ###
##########################################

# fig, ax = plt.subplots(1, 1, sharex=True)
# ax2 = ax.twinx()
# ax.bar(x=[i for i in time_acc.keys()], height=[val.get('Enter').get('total') for val in time_acc.values()], color = 'orange', alpha=0.8, width=0.5, label='total number of pedestrian')
# # ax2.plot([i for i in time_acc.keys()], [val.get('Enter').get('acc') for val in time_acc.values()], color = 'lightseagreen', alpha=0.8, marker='o', linewidth=4, markersize=12)
# sns.regplot(x=[i for i in time_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in time_acc.values()], dtype=float), ax = ax2, label='intervene accuracy', truncate=False)
# ax2.set_yticks([i / 10 for i in range(0, 11, 1)])
#
# ax.set_xlabel('elapsed time [min]')
# ax.set_ylabel('total number of pedestrian')
# ax2.set_ylabel('intervene accuracy')
# ax.set_title('Intervene Accuracy (Enter) and Erapsed Time')
#
# fig.legend(loc='lower right')
# fig.set_size_inches(5, 4.5, forward=True)
# plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# plt.savefig('pie_enter_time_acc.png')
# plt.show()
#
#
#
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax2 = ax.twinx()
# ax.bar(x=[i for i in prob_acc.keys()], height=[val.get('Enter').get('total') for val in prob_acc.values()], color = 'orange', alpha=0.8, width=0.05, label='total number of pedestrian')
# # ax2.plot([i for i in prob_acc.keys()], [val.get('Enter').get('acc') for val in prob_acc.values()], color = 'lightseagreen', alpha=0.8, marker='o', linewidth=4, markersize=12)
# sns.regplot(x=[i for i in prob_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in prob_acc.values()], dtype=float), ax = ax2, label='intervene accuracy', truncate=False)
# ax2.set_yticks([i / 10 for i in range(0, 11, 1)])
#
# ax.set_xlabel('pedestrian intention')
# ax.set_ylabel('total number of pedestrian')
# ax2.set_ylabel('intervene accuracy')
# ax.set_title('Intervene Accuracy (Enter) and Pedestrian Intention')
#
# fig.legend(loc='lower right')
# plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_enter_prob_acc.png')
# plt.show()
#
#
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax2 = ax.twinx()
# ax.bar(x=[i for i in display_distance_acc.keys()], height=[val.get('Enter').get('total') for val in display_distance_acc.values()], color = 'orange', alpha=0.8, width=2.5, label='total number of pedestrian')
# # ax2.plot([i for i in display_distance_acc.keys()], [val.get('Enter').get('acc') for val in display_distance_acc.values()], color = 'lightseagreen', alpha=0.8, marker='o', linewidth=4, markersize=12)
# sns.regplot(x=[i for i in display_distance_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in display_distance_acc.values()], dtype=float), ax = ax2, label='intervene accuracy', truncate=False)
# ax2.set_yticks([i / 10 for i in range(0, 11, 1)])
#
# ax.set_xlabel('distance from ego vehicle to target [m]')
# ax.set_ylabel('total number of pedestrian')
# ax2.set_ylabel('intervene accuracy')
# ax.set_title('Intervene Accuracy (Enter) and Distance')
#
# fig.legend(loc='lower right')
# plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_enter_dist_acc.png')
# plt.show()
#
#
#
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax2 = ax.twinx()
# ax.bar(x=[i for i in display_velocity_acc.keys()], height=[val.get('Enter').get('total') for val in display_velocity_acc.values()], color = 'orange', alpha=0.8, width=2.5, label='total number of pedestrian')
# # ax2.plot([i for i in display_velocity_acc.keys()], [val.get('Enter').get('acc') for val in display_velocity_acc.values()], color = 'lightseagreen', alpha=0.8, marker='o', linewidth=4, markersize=12)
# sns.regplot(x=[i for i in display_velocity_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in display_velocity_acc.values()], dtype=float), ax = ax2, label='intervene accuracy', truncate=False)
# ax2.set_yticks([i / 10 for i in range(0, 11, 1)])
#
# ax.set_xlabel('velocity of ego vehicle [m/s]')
# ax.set_ylabel('total number of pedestrian')
# ax2.set_ylabel('intervene accuracy')
# ax.set_title('Intervene Accuracy (Enter) and Velocity')
#
# fig.legend(loc='lower right')
# plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_enter_vel_acc.png')
# plt.show()
#
#
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax2 = ax.twinx()
# ax.bar(x=[i for i in box_size_acc.keys()], height=[val.get('Enter').get('total') for val in box_size_acc.values()], color = 'orange', alpha=0.8, width=5, label='total number of pedestrian')
# # ax2.plot([i for i in box_size_acc.keys()], [val.get('Enter').get('acc') for val in box_size_acc.values()], color = 'lightseagreen', alpha=0.8, marker='o', linewidth=4, markersize=12)
# sns.regplot(x=[i for i in box_size_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in box_size_acc.values()], dtype=float), ax = ax2, label='intervene accuracy', truncate=False)
# ax2.set_yticks([i / 10 for i in range(0, 11, 1)])
#
# ax.set_xlabel('box size of target [px]')
# ax.set_ylabel('total number of pedestrian')
# ax2.set_ylabel('intervene accuracy')
# ax.set_title('Intervene Accuracy (Enter) and Target Size')
#
# fig.legend(loc='lower right')
# plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_enter_size_acc.png')
# plt.show()
#
#
#
#
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax2 = ax.twinx()
# ax.bar(x=[i for i in intervene_time_acc.keys()], height=[val.get('Enter').get('total') for val in intervene_time_acc.values()], color = 'orange', alpha=0.8, width=0.05, label='total number of pedestrian')
# # ax2.plot([i for i in intervene_time_acc.keys()], [val.get('Enter').get('acc') for val in intervene_time_acc.values()], color = 'lightseagreen', alpha=0.8, marker='o', linewidth=4, markersize=12)
# sns.regplot(x=[i for i in intervene_time_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in intervene_time_acc.values()], dtype=float), ax = ax2, label='intervene accuracy', truncate=False)
# ax2.set_yticks([i / 10 for i in range(0, 11, 1)])
#
# ax.set_xlabel('intervene_time [s]')
# ax.set_ylabel('total number of pedestrian')
# ax2.set_ylabel('intervene accuracy')
# ax.set_title('Intervene Accuracy (Enter) and Intervene Time')
#
# fig.legend(loc='lower right')
# plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_enter_inttime_acc.png')
# plt.show()
#
#
# ##### touch ######
#
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax2 = ax.twinx()
# ax.bar(x=[i for i in time_acc.keys()], height=[val.get('Touch').get('total') for val in time_acc.values()], color = 'tomato', alpha=0.8, width=0.5, label='total number of pedestrian')
# # ax2.plot([i for i in time_acc.keys()], [val.get('Touch').get('acc') for val in time_acc.values()], color = 'lightseagreen', alpha=0.8, marker='o', linewidth=4, markersize=12)
# sns.regplot(x=[i for i in time_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in time_acc.values()], dtype=float), ax = ax2, color='seagreen', label='intervene accuracy', truncate=False)
# ax2.set_yticks([i / 10 for i in range(0, 11, 1)])
#
# ax.set_xlabel('elapsed time [min]')
# ax.set_ylabel('total number of pedestrian')
# ax2.set_ylabel('intervene accuracy')
# ax.set_title('Intervene Accuracy (Touch) and Erapsed Time')
#
# fig.legend(loc='lower right')
# plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_touch_time_acc.png')
# plt.show()
#
#
#
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax2 = ax.twinx()
# ax.bar(x=[i for i in prob_acc.keys()], height=[val.get('Touch').get('total') for val in prob_acc.values()], color = 'tomato', alpha=0.8, width=0.05, label='total number of pedestrian')
# # ax2.plot([i for i in prob_acc.keys()], [val.get('Touch').get('acc') for val in prob_acc.values()], color = 'lightseagreen', alpha=0.8, marker='o', linewidth=4, markersize=12)
# sns.regplot(x=[i for i in prob_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in prob_acc.values()], dtype=float), ax = ax2, color='seagreen', label='intervene accuracy', truncate=False)
# ax2.set_yticks([i / 10 for i in range(0, 11, 1)])
#
# ax.set_xlabel('pedestrian intention')
# ax.set_ylabel('total number of pedestrian')
# ax2.set_ylabel('intervene accuracy')
# ax.set_title('Intervene Accuracy (Touch) and Pedestrian Intention')
#
# fig.legend(loc='lower right')
# plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_touch_prob_acc.png')
# plt.show()
#
#
#
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax2 = ax.twinx()
# ax.bar(x=[i for i in display_distance_acc.keys()], height=[val.get('Touch').get('total') for val in display_distance_acc.values()], color = 'tomato', alpha=0.8, width=2.5, label='total number of pedestrian')
# # ax2.plot([i for i in display_distance_acc.keys()], [val.get('Touch').get('acc') for val in display_distance_acc.values()], color = 'lightseagreen', alpha=0.8, marker='o', linewidth=4, markersize=12)
# sns.regplot(x=[i for i in display_distance_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in display_distance_acc.values()], dtype=float), ax = ax2, color='seagreen', label='intervene accuracy', truncate=False)
# ax2.set_yticks([i / 10 for i in range(0, 11, 1)])
#
# ax.set_xlabel('distance from ego vehicle to target [m]')
# ax.set_ylabel('total number of pedestrian')
# ax2.set_ylabel('intervene accuracy')
# ax.set_title('Intervene Accuracy (Touch) and Distance')
#
# fig.legend(loc='lower right')
# plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_touch_dist_acc.png')
# plt.show()
#
#
#
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax2 = ax.twinx()
# ax.bar(x=[i for i in display_velocity_acc.keys()], height=[val.get('Touch').get('total') for val in display_velocity_acc.values()], color = 'tomato', alpha=0.8, width=2.5, label='total number of pedestrian')
# # ax2.plot([i for i in display_velocity_acc.keys()], [val.get('Touch').get('acc') for val in display_velocity_acc.values()], color = 'lightseagreen', alpha=0.8, marker='o', linewidth=4, markersize=12)
# sns.regplot(x=[i for i in display_velocity_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in display_velocity_acc.values()], dtype=float), ax = ax2, color='seagreen', label='intervene accuracy', truncate=False)
# ax2.set_yticks([i / 10 for i in range(0, 11, 1)])
#
# ax.set_xlabel('velocity of ego vehicle [m/s]')
# ax.set_ylabel('total number of pedestrian')
# ax2.set_ylabel('intervene accuracy')
# ax.set_title('Intervene Accuracy (Touch) and Velocity')
#
# fig.legend(loc='lower right')
# plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_touch_vel_acc.png')
# plt.show()
#
#
#
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax2 = ax.twinx()
# ax.bar(x=[i for i in box_size_acc.keys()], height=[val.get('Touch').get('total') for val in box_size_acc.values()], color = 'tomato', alpha=0.8, width=5, label='total number of pedestrian')
# # ax2.plot([i for i in box_size_acc.keys()], [val.get('Touch').get('acc') for val in box_size_acc.values()], color = 'lightseagreen', alpha=0.8, marker='o', linewidth=4, markersize=12)
# sns.regplot(x=[i for i in box_size_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in box_size_acc.values()], dtype=float), ax = ax2, color='seagreen', label='intervene accuracy', truncate=False)
# ax2.set_yticks([i / 10 for i in range(0, 11, 1)])
#
# ax.set_xlabel('box size of target [px]')
# ax.set_ylabel('total number of pedestrian')
# ax2.set_ylabel('intervene accuracy')
# ax.set_title('Intervene Accuracy (Touch) and Target Size')
#
# fig.legend(loc='lower right')
# plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_touch_size_acc.png')
# plt.show()
#
#
#
#
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax2 = ax.twinx()
# ax.bar(x=[i for i in intervene_time_acc.keys()], height=[val.get('Touch').get('total') for val in intervene_time_acc.values()], color = 'tomato', alpha=0.8, width=0.05, label='total number of pedestrian')
# # ax2.plot([i for i in intervene_time_acc.keys()], [val.get('Touch').get('acc') for val in intervene_time_acc.values()], color = 'lightseagreen', alpha=0.8, marker='o', linewidth=4, markersize=12)
# sns.regplot(x=[i for i in intervene_time_acc.keys()], y=np.array([val.get('Touch').get('acc') for val in intervene_time_acc.values()], dtype=float), ax = ax2, color='seagreen', label='intervene accuracy', truncate=False)
# ax2.set_yticks([i / 10 for i in range(0, 11, 1)])
#
# ax.set_xlabel('intervene_time [s]')
# ax.set_ylabel('total number of pedestrian')
# ax2.set_ylabel('intervene accuracy')
# ax.set_title('Intervene Accuracy (Touch) and Intervene Time')
#
# fig.legend(loc='lower right')
# plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_touch_inttime_acc.png')
# plt.show()

######################
### time-intervene ###
######################
df_time = df_logistic.loc[:, ['intervene_type', 'intervene_time']]
df_time.dropna(inplace=True)
df_time = df_time.replace(0, {'intervene_type':'Enter'})
df_time = df_time.replace(1, {'intervene_type':'Touch'})

fig = plt.figure()
ax = sns.boxplot(data=df_time, x='intervene_type', y='intervene_time', palette=['orange', 'seagreen'], showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
sns.swarmplot(data=df_time, x='intervene_type', y='intervene_time', color=".25", ax=ax)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_title(label='Intervene Time for Intervene Methods')
ax.set_xlabel('intervene method')
ax.set_ylabel('intervene time [s]')
fig.set_size_inches(5, 4.5, forward=True)
plt.subplots_adjust(bottom=0.17, top=0.93, left=0.15, right=0.97)
# plt.savefig('pie_inttime_inttype.png')
plt.show()

result = stats.ttest_ind(df_time.query('intervene_type == "Enter"').intervene_time, df_time.query('intervene_type == "Touch"').intervene_time, equal_var=True)
print('time-intervene_type_result: ', result)
print('time-intervene_mean Enter: ', np.mean(df_time.query('intervene_type == "Enter"').intervene_time), 'Touch', np.mean(df_time.query('intervene_type == "Touch"').intervene_time))



########################
### nasa-tls barplot ###
########################
df_tlx = pd.melt(df_tlx, id_vars=['Intervene', 'WWL'], value_vars=['Mental_Demand', 'Physical_Demand', 'Temporal_Demand', 'Overall_Performance', 'Effort', 'Frustration_Level', 'Total_Demand'], var_name='Scale', value_name='Rating')
fig, ax = plt.subplots(1, 1)
sns.barplot(x='Scale', y='Rating', data=df_tlx,  hue='Intervene',hue_order=['Enter', 'Touch'], ax=ax, palette=['orange','seagreen'])
ax.set_title('Result of NASA-TLX')
ax.set_xlabel('Scale')
ax.set_ylabel('Rating')
ax.set_xticklabels(ax.get_xticklabels(), rotation=65, horizontalalignment='right')
# ax.tick_params(axis='x', labelsize=18, labelrotation=55, labelright=True)
ax.tick_params(axis='y', labelsize=18)
plt.savefig('pie_time_tlx.png')
fig.set_size_inches(5, 4.5, forward=True)
plt.show()

result = stats.ttest_ind(df_tlx.query('Intervene == "Enter"').WWL, df_tlx.query('Intervene == "Touch"').WWL, equal_var=True)
print('nasa-tlx WWL score: ', result)
print('WWL enter mean: ', np.mean(df_tlx.query('Intervene == "Enter"').WWL), ' WWL touch mean: ', np.mean(df_tlx.query('Intervene == "Touch"').WWL))


df_kuriatsu_inttime =df_kuriatsu.dropna()
fig, ax = plt.subplots(1, 1)
buf = df_kuriatsu_inttime.query('intervene_type == "YN"')
sns.regplot(x='prob', y='intervene_time', data=buf.query('is_intervene_correct == 1'), color='orange',ax=ax, label='YN_true', truncate=True, order=2)
sns.regplot(x='prob', y='intervene_time', data=buf.query('is_intervene_correct == 0'), color='blue',ax=ax, label='YN_false', truncate=True, order=2)
plt.legend()
ax.set_title(label='Intervene Time and Pedestrian Intention (order=2)')
ax.set_xlabel(xlabel='pedestrian intention')
ax.set_ylabel(ylabel='intervene time [s]')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.subplots_adjust(bottom=0.17, top=0.93, left=0.12, right=0.95)
fig.set_size_inches(5, 4.5, forward=True)
plt.savefig('pie_inttime_int_order2.png')
plt.show()


prob_acc = {i : {'YN':{'total':0, 'correct':0}} for i in [i / 20 for i in range(math.floor(min(buf.prob) * 20), math.floor(max(buf.prob) * 20) + 1, 1)]}
for index in buf.index:
    prob = int(buf.at[index, 'prob'] * 20) / 20
    if prob not in prob_acc: prob_acc[prob] = {'YN':{'total':0, 'correct':0}}
    prob_acc[prob][buf.at[index, 'intervene_type']]['total']+=1
    prob_acc[prob][buf.at[index, 'intervene_type']]['correct']+=buf.at[index, 'is_intervene_correct']
for i, val in prob_acc.items():
    prob_acc[i]['YN']['acc'] = val.get('YN').get('correct') / val.get('YN').get('total') if val.get('YN').get('total') != 0 else None


fig, ax = plt.subplots(1,1)
# sns.regplot(x=[i for i in prob_acc.keys()], y=np.array([val.get('Enter').get('acc') for val in prob_acc.values()], dtype=float), ax = ax, label='Enter', truncate=False, color='orange')
sns.regplot(x=[i for i in prob_acc.keys()], y=np.array([val.get('YN').get('acc') for val in prob_acc.values()], dtype=float), ax = ax, label='YN', truncate=False, color='orange', order=2)
ax.set_xlabel('pedestrian intention [s]')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy and Pedestrian Intention')
ax.legend(loc='lower right')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_ylim(0.0, 1.0)
plt.subplots_adjust(bottom=0.15, top=0.93, left=0.18, right=0.97)
fig.set_size_inches(5, 4.5, forward=True)
# plt.savefig('pie_acc_prob_kuri.png')
plt.show()
