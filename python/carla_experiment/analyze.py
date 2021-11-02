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
from scipy.stats import f_oneway
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

def addAnotation(plt_obj, x1, x2, y, hight, h_offset, text, color):
    plt_obj.plot([x1, x1, x2, x2], [y+h_offset, y+hight+h_offset, y+hight+h_offset, y+h_offset], lw=1.5, c=color)
    plt_obj.text((x1+x2)*0.5, y+hight+h_offset, text, ha='center', va='bottom', color=color)

def saveCsv(data, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

sns.set_palette('YlGnBu',4)
sns.set(context='paper', style='whitegrid')
color = {'baseline':'#d5e7ba', 'control': '#7dbeb5', 'button': '#388fad', 'touch': '#335290'}
color_list = ['#d5e7ba', '#7dbeb5', '#388fad', '#335290']
sns.set_palette(sns.color_palette(color_list))

summary_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/summary.csv')
nasa_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/nasa-tlx.csv')
rank_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/rank.csv')
subjects = summary_df.subject.drop_duplicates()
experiments = summary_df.experiment_type.drop_duplicates()

################################################################
################################################################

print('counter variance')
summary_df["cv"] = summary_df.std_vel / summary_df.mean_vel
_, p = stats.levene(summary_df[summary_df.experiment_type == 'baseline'].cv.dropna(), summary_df[summary_df.experiment_type == 'control'].cv.dropna(), summary_df[summary_df.experiment_type == 'button'].cv.dropna(), summary_df[summary_df.experiment_type == 'touch'].cv.dropna(), center='median')
axes = sns.boxplot(data=summary_df, x='experiment_type', y='cv', showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
plt.show()
print('levene-first cv', p)
if p > 0.05:
    multicomp_result = multicomp.MultiComparison(np.array(summary_df.dropna(how='any').cv, dtype="float64"), summary_df.dropna(how='any').experiment_type)
    print(multicomp_result.tukeyhsd().summary())


################################################################
################################################################
print('intervene accuracy')
intervene_acc = pd.DataFrame(index=subjects, columns=experiments)
for subject in subjects:
    cross_df = summary_df[(summary_df.subject == subject) & (summary_df.actor_action == "cross")]
    for experiment in experiments:
        buf = cross_df[cross_df.experiment_type==experiment].intervene_vel.isnull()
        rate = 1.0 - buf.sum() / len(cross_df[cross_df.experiment_type==experiment])
        intervene_acc.at[subject, experiment] = rate

axes = sns.boxplot(data=intervene_acc, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
plt.show()
_, p = stats.levene(intervene_acc.baseline, intervene_acc.control, intervene_acc.button, intervene_acc.touch, center='median')
print('levene-first last_intervene_time', p)
if p > 0.05:
    melted_df = pd.melt(intervene_acc, var_name='experiment_type', value_name='intervene_vel')
    multicomp_result = multicomp.MultiComparison(np.array(melted_df['intervene_vel'], dtype="float64"), melted_df['experiment_type'])
    print(multicomp_result.tukeyhsd().summary())

print("avoid decel rate")
intervene_speed = pd.DataFrame(index=subjects, columns=experiments)
for subject in subjects:
    pose_df = summary_df[(summary_df.subject == subject) & (summary_df.actor_action == "pose")]
    for experiment in experiments:
        buf = pose_df[pose_df.experiment_type==experiment].intervene_vel > 40.0
        rate = buf.sum() / len(pose_df[pose_df.experiment_type==experiment])
        intervene_speed.at[subject, experiment] = rate

_, p = stats.levene(intervene_speed.baseline, intervene_speed.control, intervene_speed.button, intervene_speed.touch, center='median')
print('levene-first last_intervene_time', p)
axes = sns.boxplot(data=intervene_speed, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
plt.show()

melted_df = pd.melt(intervene_speed, var_name='experiment_type', value_name='intervene_vel')
multicomp_result = multicomp.MultiComparison(np.array(melted_df['intervene_vel'], dtype="float64"), melted_df['experiment_type'])
print(multicomp_result.tukeyhsd().summary())

print("accuracy vs avoid decel")
_, axes = plt.subplots()
for experiment in experiments:
    axes.errorbar(intervene_acc.mean()[experiment], intervene_speed.mean()[experiment], xerr=intervene_acc.sem()[experiment], yerr=intervene_speed.sem()[experiment], marker='o', capsize=5, label=experiment)

axes.set_xlim(0, 1.0)
axes.set_ylim(0, 1.0)
axes.set_xlabel('Unnecessary intervention rate', fontsize=15)
axes.set_ylabel('Avoid deceleration rate', fontsize=15)
axes.legend(loc='upper right', fontsize=15)
axes.tick_params(axis='x', labelsize=12)
axes.tick_params(axis='y', labelsize=12)
plt.show()
################################################################
print("intervene vel range stacked bar plot")
################################################################

intervene_speed_rate = pd.DataFrame(index=experiments, columns=[10, 20, 30, 40, 50])
intervene_speed_rate.fillna(0, inplace=True)
pose_df = summary_df[summary_df.actor_action == "pose"]
for i, row in pose_df.iterrows():
    for thres in [10, 20, 30, 40, 50]:
        if row.intervene_vel < thres:
            intervene_speed_rate.at[row.experiment_type, thres] += 1

for i, row in intervene_speed_rate.iteritems():
    intervene_speed_rate.at[:, i] = row / intervene_speed_rate[50]

sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[50],color="teal")
sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[40],color="turquoise")
sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[30],color="gold")
sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[20],color="lightsalmon")
sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[10],color="orangered")

################################################################
print("min vel range stacked bar plot")
################################################################

intervene_speed_rate = pd.DataFrame(index=experiments, columns=[10, 20, 30, 40, 50])
intervene_speed_rate.fillna(0, inplace=True)
pose_df = summary_df[summary_df.actor_action == "pose"]
for i, row in pose_df.iterrows():
    for thres in [10, 20, 30, 40, 50]:
        if row.min_vel < thres:
            intervene_speed_rate.at[row.experiment_type, thres] += 1

for i, row in intervene_speed_rate.iteritems():
    intervene_speed_rate.at[:, i] = row / intervene_speed_rate[50]

axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[50],color="teal", label='40-50km/h')
axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[40],color="turquoise", label='30-40km/h')
axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[30],color="gold", label='20-30km/h')
axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[20],color="lightsalmon", label='10-20km/h')
axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[10],color="orangered", label='0-10km/h')
axes.set_ylabel('Rate of driving with minimum velocity while intervenition', fontsize=15)
axes.set_xlabel('Intervention method', fontsize=15)
axes.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()

print('min vel')
_, p = stats.levene(summary_df[summary_df.experiment_type == 'baseline'].min_vel, summary_df[summary_df.experiment_type == 'control'].min_vel, summary_df[summary_df.experiment_type == 'button'].min_vel, summary_df[summary_df.experiment_type == 'touch'].min_vel, center='median')
print('levene-first min vel', p)
if p > 0.05:
    multicomp_result = multicomp.MultiComparison(np.array(summary_df.min_vel, dtype="float64"), melted_df.experiment_type)
    print(multicomp_result.tukeyhsd().summary())

print('max vel')
_, p = stats.levene(summary_df[summary_df.experiment_type == 'baseline'].max_vel, summary_df[summary_df.experiment_type == 'control'].max_vel, summary_df[summary_df.experiment_type == 'button'].max_vel, summary_df[summary_df.experiment_type == 'touch'].max_vel, center='median')
print('levene-first max vel', p)
if p > 0.05:
    multicomp_result = multicomp.MultiComparison(np.array(summary_df.max_vel, dtype="float64"), melted_df.experiment_type)
    print(multicomp_result.tukeyhsd().summary())


################################################################
print('min_vel rerative to baseline plot')
################################################################

intervene_speed_rate = pd.DataFrame(columns=['subjects', 'experiment_type', 'range', 'rate'], index = [])

for subject in subjects:
    for experiment in experiments:
        for thres in [10.0, 20.0, 30.0, 40.0, 50.0]:
            rate = len(summary_df[(summary_df.subject == subject) & (summary_df.experiment_type == experiment) & (summary_df.actor_action == 'pose') & (summary_df.min_vel < thres)]) / len(summary_df[(summary_df.subject == subject) & (summary_df.experiment_type == experiment) & (summary_df.actor_action == 'pose')])
            buf_df = pd.Series([subject, experiment, thres, rate], index=intervene_speed_rate.columns)
            intervene_speed_rate = intervene_speed_rate.append(buf_df, ignore_index=True)

intervene_speed_rate_summary = pd.DataFrame(columns=['experiment_type', 'range', 'mean', 'std_err'], index = [])
for experiment in experiments:
    for thres in [10.0, 20.0, 30.0, 40.0, 50.0]:
        mean = intervene_speed_rate[(intervene_speed_rate.experiment_type == experiment) & (intervene_speed_rate.range == thres)].rate.mean()
        std_err = intervene_speed_rate[(intervene_speed_rate.experiment_type == experiment) & (intervene_speed_rate.range == thres)].rate.sem()
        buf_df = pd.Series([experiment, thres, mean, std_err], index=intervene_speed_rate_summary.columns)
        intervene_speed_rate_summary = intervene_speed_rate_summary.append(buf_df, ignore_index=True)

################################################################
print('stop rate')
################################################################

stop_rate = pd.DataFrame(index=subjects, columns=experiments)
for subject in subjects:
    pose_df = summary_df[(summary_df.subject == subject) & (summary_df.actor_action == "cross")]
    for experiment in experiments:
        buf = pose_df[pose_df.experiment_type==experiment].min_vel < 20.0
        rate = buf.sum() / len(pose_df[pose_df.experiment_type==experiment])
        stop_rate.at[subject, experiment] = rate

axes = sns.boxplot(data=stop_rate, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
plt.show()

_, p = stats.levene(stop_rate.baseline, stop_rate.control, stop_rate.button, stop_rate.touch, center='median')
print('levene-first stop rate', p)
if p > 0.05:
    melted_df = pd.melt(stop_rate, var_name='experiment_type', value_name='intervene_vel')
    multicomp_result = multicomp.MultiComparison(np.array(melted_df['intervene_vel'], dtype="float64"), melted_df['experiment_type'])
    print(multicomp_result.tukeyhsd().summary())

################################################################
print('keep rate')
################################################################

stop_rate = pd.DataFrame(index=subjects, columns=experiments)
for subject in subjects:
    pose_df = summary_df[(summary_df.subject == subject) & (summary_df.actor_action == "pose")]
    for experiment in experiments:
        buf = pose_df[pose_df.experiment_type==experiment].min_vel > 30.0
        rate = buf.sum() / len(pose_df[pose_df.experiment_type==experiment])
        stop_rate.at[subject, experiment] = rate

axes = sns.boxplot(data=stop_rate, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
plt.show()

_, p = stats.levene(stop_rate.baseline, stop_rate.control, stop_rate.button, stop_rate.touch, center='median')
print('levene-first keep rate', p)
if p > 0.05:
    melted_df = pd.melt(stop_rate, var_name='experiment_type', value_name='intervene_vel')
    multicomp_result = multicomp.MultiComparison(np.array(melted_df['intervene_vel'], dtype="float64"), melted_df['experiment_type'])
    print(multicomp_result.tukeyhsd().summary())
