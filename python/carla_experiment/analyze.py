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
from statsmodels.stats.contingency_tables import cochrans_q
from scipy.stats import f_oneway
import scikit_posthocs as sp
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
experiments = ['baseline', 'control', 'button', 'touch']

################################################################
print('counter variance')
################################################################
summary_df["cv"] = summary_df.std_vel / summary_df.mean_vel

_, norm_p = stats.shapiro(summary_df.cv.dropna())
_, var_p = stats.levene(
    summary_df[summary_df.experiment_type == 'baseline'].cv.dropna(),
    summary_df[summary_df.experiment_type == 'control'].cv.dropna(),
    summary_df[summary_df.experiment_type == 'button'].cv.dropna(),
    summary_df[summary_df.experiment_type == 'touch'].cv.dropna(),
    center='median'
    )

if norm_p < 0.05 or var_p < 0.05:
    print('steel-dwass\n', sp.posthoc_dscf(summary_df, val_col='cv', group_col='experiment_type'))
else:
    multicomp_result = multicomp.MultiComparison(np.array(summary_df.dropna(how='any').cv, dtype="float64"), summary_df.dropna(how='any').experiment_type)
    print('levene', multicomp_result.tukeyhsd().summary())

axes = sns.boxplot(data=summary_df, x='experiment_type', y='cv', showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
plt.show()


################################################################
print('intervene accuracy')
################################################################
inttype_accuracy = pd.DataFrame(columns=['experiment', 'result'])
for index, row in summary_df.iterrows():
    if row.actor_action == 'cross':
        buf = pd.DataFrame([(row.experiment_type, np.isnan(row.intervene_vel)) ], columns=['experiment', 'result'])
        inttype_accuracy = inttype_accuracy.append(buf, ignore_index=True)
    elif row.actor_action == 'pose':
        if np.isnan(row.intervene_vel):
            buf = pd.DataFrame([(row.experiment_type, False)], columns=['experiment', 'result'])
            inttype_accuracy = inttype_accuracy.append(buf, ignore_index=True)
        else:
            buf = pd.DataFrame([(row.experiment_type, (row.intervene_vel > 1.0))], columns=['experiment', 'result'])
            inttype_accuracy = inttype_accuracy.append(buf, ignore_index=True)

inttype_accuracy_cross = pd.crosstab(inttype_accuracy.experiment, inttype_accuracy.result)
stats.chi2_contingency(inttype_accuracy_cross)

subject_accuracy = pd.DataFrame(columns=['subject', 'result'])
for index, row in summary_df.iterrows():
    if row.actor_action == 'cross':
        buf = pd.DataFrame([(row.subject, np.isnan(row.intervene_vel)) ], columns=['subject', 'result'])
        subject_accuracy = subject_accuracy.append(buf, ignore_index=True)
    elif row.actor_action == 'pose':
        if np.isnan(row.intervene_vel):
            buf = pd.DataFrame([(row.subject, False)], columns=['subject', 'result'])
            subject_accuracy = subject_accuracy.append(buf, ignore_index=True)
        else:
            buf = pd.DataFrame([(row.subject, (row.intervene_vel > 1.0))], columns=['subject', 'result'])
            subject_accuracy = subject_accuracy.append(buf, ignore_index=True)

subject_accuracy_cross = pd.crosstab(subject_accuracy.subject, subject_accuracy.result)
stats.chi2_contingency(subject_accuracy_cross)


################################################################
print('intervene time')
################################################################
pose_df = summary_df[summary_df.actor_action == 'pose']
_, norm_p = stats.shapiro(pose_df.first_intervene_time.dropna())
_, var_p = stats.levene(pose_df[pose_df.experiment_type == 'baseline'].first_intervene_time.dropna(), pose_df[pose_df.experiment_type == 'control'].first_intervene_time.dropna(), pose_df[pose_df.experiment_type == 'button'].first_intervene_time.dropna(), pose_df[pose_df.experiment_type == 'touch'].first_intervene_time.dropna(), center='median')
if norm_p < 0.05 or var_p < 0.05:
    print('steel-dwass\n', sp.posthoc_dscf(pose_df, val_col='first_intervene_time', group_col='experiment_type'))
else:
    multicomp_result = multicomp.MultiComparison(np.array(pose_df.dropna(how='any').first_intervene_time, dtype="float64"), pose_df.dropna(how='any').experiment_type)
    print('levene', multicomp_result.tukeyhsd().summary())


pose_df = summary_df[summary_df.actor_action == 'pose']
_, norm_p = stats.shapiro(pose_df.first_intervene_time.dropna())
_, var_p = stats.levene(
    pose_df[pose_df.subject == 'ando'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'aso'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'hikosaka'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'ichiki'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'ienaga'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'ikai'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'isobe'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'ito'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'kato'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'matsubara'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'nakakuki'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'nakatani'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'negi'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'otake'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'sumiya'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'taga'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'yamamoto'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'yasuhara'].first_intervene_time.dropna(),
    center='median'
    )

_, anova_p = stats.kruskal(
    pose_df[pose_df.subject == 'ando'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'aso'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'hikosaka'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'ichiki'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'ienaga'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'ikai'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'isobe'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'ito'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'kato'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'matsubara'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'nakakuki'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'nakatani'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'negi'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'otake'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'sumiya'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'taga'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'yamamoto'].first_intervene_time.dropna(),
    pose_df[pose_df.subject == 'yasuhara'].first_intervene_time.dropna(),
)

if anova_p < 0.05:
    if norm_p < 0.05 or var_p < 0.05:
        print('steel-dwass\n', sp.posthoc_dscf(pose_df, val_col='first_intervene_time', group_col='subject'))
    else:
        multicomp_result = multicomp.MultiComparison(np.array(pose_df.dropna(how='any').first_intervene_time, dtype="float64"), pose_df.dropna(how='any').subject)
        print('levene', multicomp_result.tukeyhsd().summary())

################################################################
print("avoid decel rate")
################################################################
avoid_decel_df = pd.DataFrame(columns=['experiment', 'result'])
for index, row in summary_df[summary_df.actor_action == 'pose'].iterrows():
    if row.actor_action == 'pose':
        if np.isnan(row.intervene_vel):
            buf = pd.DataFrame([(row.experiment_type, False)], columns=['experiment', 'result'])
            avoid_decel_df = avoid_decel_df.append(buf, ignore_index=True)
        else:
            buf = pd.DataFrame([(row.experiment_type, (row.intervene_vel > 40.0))], columns=['experiment', 'result'])
            avoid_decel_df = avoid_decel_df.append(buf, ignore_index=True)

avoid_decel_cross = pd.crosstab(avoid_decel_df.experiment, avoid_decel_df.result)
stats.chi2_contingency(avoid_decel_cross)

################################################################
print("accuracy vs intervene_time")
################################################################
accuracy_list = [
    inttype_accuracy_cross.at['baseline', True] / (inttype_accuracy_cross.at['baseline', True] + inttype_accuracy_cross.at['baseline', False]),
    inttype_accuracy_cross.at['control', True] / (inttype_accuracy_cross.at['control', True] + inttype_accuracy_cross.at['control', False]),
    inttype_accuracy_cross.at['button', True] / (inttype_accuracy_cross.at['button', True] + inttype_accuracy_cross.at['button', False]),
    inttype_accuracy_cross.at['touch', True] / (inttype_accuracy_cross.at['touch', True] + inttype_accuracy_cross.at['touch', False])
]
intervene_time_mean_list = [
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'baseline')].first_intervene_time.mean(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'control')].first_intervene_time.mean(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'button')].first_intervene_time.mean(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'touch')].first_intervene_time.mean(),
]
intervene_time_sem_list = [
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'baseline')].first_intervene_time.sem(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'control')].first_intervene_time.sem(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'button')].first_intervene_time.sem(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'touch')].first_intervene_time.sem(),
]
_, axes = plt.subplots()
for i, experiment in enumerate(experiments):
    axes.errorbar(accuracy_list[i], intervene_time_mean_list[i], yerr=intervene_time_sem_list[i], marker='o', capsize=5, label=experiment)

axes.set_xlim(0, 1.0)
# axes.set_ylim(0, 1.0)
axes.set_xlabel('Intervention accuracy', fontsize=15)
axes.set_ylabel('Intervention time [s]', fontsize=15)
axes.legend(loc='lower left', fontsize=15)
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

# intervene_speed_rate = pd.DataFrame(columns=['experiment_type', 'range', 'rate'], index = [])
# # intervene_speed_rate = pd.DataFrame(columns=['subjects', 'experiment_type', 'range', 'rate'], index = [])
#
# # for subject in subjects:
# for experiment in experiments:
#     for thres in [-1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]:
#         rate = len(summary_df[(summary_df.experiment_type == experiment) & (summary_df.actor_action == 'pose') & (summary_df.min_vel > 45.0)].min_vel) / len(summary_df[(summary_df.experiment_type == experiment) & (summary_df.actor_action == 'pose')].min_vel)
#         buf_df = pd.Series([experiment, thres, rate], index=intervene_speed_rate.columns)
#         intervene_speed_rate = intervene_speed_rate.append(buf_df, ignore_index=True)
#
# intervene_speed_rate_summary = pd.DataFrame(columns=['experiment_type', 'range', 'mean', 'std_err'], index = [])
# for experiment in experiments:
#     for thres in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]:
#         mean = intervene_speed_rate[(intervene_speed_rate.experiment_type == experiment) & (intervene_speed_rate.range == thres)].rate.mean()
#         std_err = intervene_speed_rate[(intervene_speed_rate.experiment_type == experiment) & (intervene_speed_rate.range == thres)].rate.sem()
#         buf_df = pd.Series([experiment, thres, mean, std_err], index=intervene_speed_rate_summary.columns)
#         intervene_speed_rate_summary = intervene_speed_rate_summary.append(buf_df, ignore_index=True)
#
# for i, range in enumerate(intervene_speed_rate.range):
#     if range == -1.0:
#         intervene_speed_rate.at[i, "range"] = 0.0
# intervene_speed_rate.at[intervene_speed_rate.range == -1.0].range = 0.0
# axes = sns.pointplot(x='range', y='rate', data=intervene_speed_rate, hue='experiment_type')
#

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
