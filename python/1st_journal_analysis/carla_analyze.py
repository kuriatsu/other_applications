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
import statsmodels.stats.anova as stats_anova
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

sns.set(context='paper', style='whitegrid')
color = {'BASELINE':'#add8e6', 'CONTROL': '#7dbeb5', 'BUTTON': '#388fad', 'TOUCH': '#335290'}
sns.set_palette(sns.color_palette(color.values()))

# summary_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/summary_rm_wrong.csv')
# nasa_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/nasa-tlx.csv')
# rank_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/rank.csv')
summary_df = pd.read_csv('/home/kuriatsu/Documents/experiment/carla_202102_result/summary_rm_wrong.csv')
# summary_df = pd.read_csv('/home/kuriatsu/Documents/experiment/carla_202102_result/summary.csv')
nasa_df = pd.read_csv('/home/kuriatsu/Documents/experiment/carla_202102_result/nasa-tlx.csv')
rank_df = pd.read_csv('/home/kuriatsu/Documents/experiment/carla_202102_result/rank.csv')

subjects = summary_df.subject.drop_duplicates()
experiments = ['BASELINE', 'CONTROL', 'BUTTON', 'TOUCH']

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
print ('----', 'baseline', 'control', 'button', 'touch')
print('mean',
    summary_df[summary_df.experiment_type == 'baseline'].cv.dropna().mean(),
    summary_df[summary_df.experiment_type == 'control'].cv.dropna().mean(),
    summary_df[summary_df.experiment_type == 'button'].cv.dropna().mean(),
    summary_df[summary_df.experiment_type == 'touch'].cv.dropna().mean(),
)
print('var',
    summary_df[summary_df.experiment_type == 'baseline'].cv.dropna().std(),
    summary_df[summary_df.experiment_type == 'control'].cv.dropna().std(),
    summary_df[summary_df.experiment_type == 'button'].cv.dropna().std(),
    summary_df[summary_df.experiment_type == 'touch'].cv.dropna().std(),
    )
axes = sns.boxplot(data=summary_df, x='experiment_type', y='cv', showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
# axes.figure.savefig('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/counter_variance.svg', format="svg")
plt.show()


################################################################
print('intervene accuracy')
################################################################
inttype_accuracy = pd.DataFrame(columns=experiments, index=subjects)
for subject in subjects:
    for experiment in experiments:
        df = summary_df[(summary_df.subject == subject) & (summary_df.experiment_type == experiment)]

        # vehcle speed
        # collect = len(df[((df.actor_action == "cross") & (df.min_vel < 1.0)) | ((df.actor_action == "pose") & (df.min_vel > 1.0))])
        # inttype_accuracy.at[subject, experiment] = collect / len(df)

        # accident count
        # collect = len(df[((df.actor_action == "cross") & (df.min_vel > 1.0))])
        # inttype_accuracy.at[subject, experiment] = collect / len(df[(df.actor_action == "cross")])

        # only cross intervention
        collect = df[(df.actor_action == "cross")].intervene_vel.isnull().sum()
        # inttype_accuracy.at[subject, experiment] = collect / len(df[(df.actor_action == "cross")])


        # cross + pose intervention
        # collect += len(df[(df.actor_action == "pose") & (df.intervene_vel > df.max_vel*0.2)])
        collect += (df[(df.actor_action == "pose")].dropna().intervene_vel > 1.0).sum()
        inttype_accuracy.at[subject, experiment] = collect / len(df)

# for index, row in summary_df.iterrows():
#     if row.actor_action == 'cross':
#         buf = pd.DataFrame([(row.experiment_type, np.isnan(row.intervene_vel)) ], columns=['experiment', 'result'])
#         inttype_accuracy = inttype_accuracy.append(buf, ignore_index=True)
#     elif row.actor_action == 'pose':
#         if np.isnan(row.intervene_vel):
#             buf = pd.DataFrame([(row.experiment_type, False)], columns=['experiment', 'result'])
#             inttype_accuracy = inttype_accuracy.append(buf, ignore_index=True)
#         else:
#             buf = pd.DataFrame([(row.experiment_type, (row.intervene_vel > 1.0))], columns=['experiment', 'result'])
#             inttype_accuracy = inttype_accuracy.append(buf, ignore_index=True)

inttype_accuracy.mean()
inttype_accuracy.std()
# inttype_accuracy_cross = pd.crosstab(inttype_accuracy.experiment, inttype_accuracy.result)
# stats.chi2_contingency(inttype_accuracy_cross)
_, norm_p1 = stats.shapiro(inttype_accuracy.BASELINE)
_, norm_p2 = stats.shapiro(inttype_accuracy.CONTROL)
_, norm_p3 = stats.shapiro(inttype_accuracy.BUTTON)
_, norm_p4 = stats.shapiro(inttype_accuracy.TOUCH)
_, var_p = stats.levene(inttype_accuracy.BASELINE, inttype_accuracy.CONTROL, inttype_accuracy.BUTTON, inttype_accuracy.TOUCH, center='median')

if norm_p1 < 0.05 or norm_p2 < 0.05 or norm_p3 < 0.05 or norm_p4 < 0.05:
    _, anova_p = stats.friedmanchisquare(inttype_accuracy.BASELINE, inttype_accuracy.CONTROL, inttype_accuracy.BUTTON, inttype_accuracy.TOUCH)
    if anova_p < 0.05:
        print("conover test anova_result:", anova_p, sp.posthoc_conover_friedman(inttype_accuracy))
else:
    melted_df = pd.melt(inttype_accuracy.reset_index(), id_vars="subject", var_name="experiment_type", value_name="accuracy")
    anova_result = stats_anova.AnovaRM(melted_df, "accuracy", "subject", ["experiment_type"])
    print("reperted anova: ", anova_result.fit())
    melted_df = pd.melt(inttype_accuracy, var_name="experiment_type", value_name="accuracy")
    print("levene result", var_p)
    # gamesHowellTest(melted_df, "experiment_type", "accuracy")
    multicomp_result = multicomp.MultiComparison(np.array(melted_df.dropna(how='any').accuracy, dtype="float64"), melted_df.dropna(how='any').experiment_type)
    print(multicomp_result.tukeyhsd().summary())

subject_accuracy = inttype_accuracy.T
_, anova_p = stats.friedmanchisquare(
    subject_accuracy.ando,
    subject_accuracy.aso,
    subject_accuracy.hikosaka,
    subject_accuracy.ichiki,
    subject_accuracy.ienaga,
    subject_accuracy.ikai,
    subject_accuracy.isobe,
    subject_accuracy.ito,
    subject_accuracy.kato,
    subject_accuracy.matsubara,
    subject_accuracy.nakakuki,
    subject_accuracy.nakatani,
    subject_accuracy.negi,
    subject_accuracy.otake,
    subject_accuracy.sumiya,
    subject_accuracy.taga,
    subject_accuracy.yamamoto,
    subject_accuracy.yasuhara,
    )
print("subject wise anova = ", anova_p)

################################################################
print('intervene time')
################################################################
intervene_time = pd.DataFrame(columns=experiments, index=subjects)
for subject in subjects:
    for experiment in experiments:
        df = summary_df[(summary_df.subject == subject) & (summary_df.experiment_type == experiment)]
        time = df[(df.actor_action == "pose") & (df.intervene_vel > 1.0)].first_intervene_time.dropna().mean()
        intervene_time.at[subject, experiment] = time

_, norm_p1 = stats.shapiro(intervene_time.BASELINE)
_, norm_p2 = stats.shapiro(intervene_time.CONTROL)
_, norm_p3 = stats.shapiro(intervene_time.BUTTON)
_, norm_p4 = stats.shapiro(intervene_time.TOUCH)
_, var_p = stats.levene(intervene_time.BASELINE, intervene_time.CONTROL, intervene_time.BUTTON, intervene_time.TOUCH, center='median')
if norm_p1 < 0.05 or norm_p2 < 0.05 or norm_p3 < 0.05 or norm_p4 < 0.05:
    _, anova_p = stats.friedmanchisquare(intervene_time.BASELINE, intervene_time.CONTROL, intervene_time.BUTTON, intervene_time.TOUCH)
    print("anova(friedman test)", anova_p)
    if anova_p < 0.05:
        print(sp.posthoc_conover_friedman(intervene_time))
else:
    melted_df = pd.melt(intervene_time.reset_index(), id_vars="subject", var_name="experiment_type", value_name="first_intervene_time")
    anova_result = stats_anova.AnovaRM(melted_df, "first_intervene_time", "subject", ["experiment_type"])
    print("reperted anova: ", anova_result.fit())
    multicomp_result = multicomp.MultiComparison(np.array(melted_df.dropna(how='any').first_intervene_time, dtype="float64"), melted_df.dropna(how='any').experiment_type)
    print(multicomp_result.tukeyhsd().summary())

# pose_df = summary_df[summary_df.actor_action == 'pose']
# _, norm_p = stats.shapiro(pose_df.first_intervene_time.dropna())
# _, var_p = stats.levene(pose_df[pose_df.experiment_type == 'baseline'].first_intervene_time.dropna(), pose_df[pose_df.experiment_type == 'control'].first_intervene_time.dropna(), pose_df[pose_df.experiment_type == 'button'].first_intervene_time.dropna(), pose_df[pose_df.experiment_type == 'touch'].first_intervene_time.dropna(), center='median')
# if norm_p < 0.05 or var_p < 0.05:
#     print('steel-dwass\n', sp.posthoc_dscf(pose_df, val_col='first_intervene_time', group_col='experiment_type'))
# else:
#     multicomp_result = multicomp.MultiComparison(np.array(pose_df.dropna(how='any').first_intervene_time, dtype="float64"), intervene_time.dropna(how='any').experiment_type)
#     print('levene', multicomp_result.tukeyhsd().summary())
#

subject_time = intervene_time.T
_, anova_p = stats.friedmanchisquare(
    subject_time.ando,
    subject_time.aso,
    subject_time.hikosaka,
    subject_time.ichiki,
    subject_time.ienaga,
    subject_time.ikai,
    subject_time.isobe,
    subject_time.ito,
    subject_time.kato,
    subject_time.matsubara,
    subject_time.nakakuki,
    subject_time.nakatani,
    subject_time.negi,
    subject_time.otake,
    subject_time.sumiya,
    subject_time.taga,
    subject_time.yamamoto,
    subject_time.yasuhara,
    )
print("subject wise anova=", anova_p)


################################################################
print("avoid decel rate")
################################################################
# avoid_decel_df = pd.DataFrame(columns=['experiment', 'result'])
# for index, row in summary_df[summary_df.actor_action == 'pose'].iterrows():
#     if row.actor_action == 'pose':
#         if np.isnan(row.intervene_vel):
#             buf = pd.DataFrame([(row.experiment_type, False)], columns=['experiment', 'result'])
#             avoid_decel_df = avoid_decel_df.append(buf, ignore_index=True)
#         else:
#             buf = pd.DataFrame([(row.experiment_type, (row.intervene_vel > 40.0))], columns=['experiment', 'result'])
#             avoid_decel_df = avoid_decel_df.append(buf, ignore_index=True)
#
# avoid_decel_cross = pd.crosstab(avoid_decel_df.experiment, avoid_decel_df.result)
# stats.chi2_contingency(avoid_decel_cross)

################################################################
print("accuracy vs intervene_time")
################################################################
accuracy_mean_list = [
    inttype_accuracy.BASELINE.mean(),
    inttype_accuracy.CONTROL.mean(),
    inttype_accuracy.BUTTON.mean(),
    inttype_accuracy.TOUCH.mean(),
]

accuracy_sem_list = [
    inttype_accuracy.BASELINE.sem(),
    inttype_accuracy.CONTROL.sem(),
    inttype_accuracy.BUTTON.sem(),
    inttype_accuracy.TOUCH.sem(),
]

intervene_time_mean_list = [
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'BASELINE') & (summary_df.intervene_vel > 1.0)].first_intervene_time.dropna().mean(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'CONTROL') & (summary_df.intervene_vel > 1.0)].first_intervene_time.dropna().mean(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'BUTTON') & (summary_df.intervene_vel > 1.0)].first_intervene_time.dropna().mean(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'TOUCH') & (summary_df.intervene_vel > 1.0)].first_intervene_time.dropna().mean(),
]
intervene_time_sem_list = [
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'BASELINE') & (summary_df.intervene_vel > 1.0)].first_intervene_time.dropna().sem(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'CONTROL') & (summary_df.intervene_vel > 1.0)].first_intervene_time.dropna().sem(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'BUTTON') & (summary_df.intervene_vel > 1.0)].first_intervene_time.dropna().sem(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'TOUCH') & (summary_df.intervene_vel > 1.0)].first_intervene_time.dropna().sem(),
]

print ('----', 'baseline', 'control', 'button', 'touch')
print('acc mean', accuracy_mean_list)
print('time mean', intervene_time_mean_list)
print('acc var',
    inttype_accuracy.BASELINE.std(),
    inttype_accuracy.CONTROL.std(),
    inttype_accuracy.BUTTON.std(),
    inttype_accuracy.TOUCH.std(),
    )
print('time sd',
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'BASELINE') & (summary_df.intervene_vel > 1.0)].first_intervene_time.std(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'CONTROL') & (summary_df.intervene_vel > 1.0)].first_intervene_time.std(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'BUTTON') & (summary_df.intervene_vel > 1.0)].first_intervene_time.std(),
    summary_df[(summary_df.actor_action == 'pose') & (summary_df.experiment_type == 'TOUCH') & (summary_df.intervene_vel > 1.0)].first_intervene_time.std(),
)

fig, axes = plt.subplots()
for i, experiment in enumerate(experiments):
    axes.errorbar(accuracy_mean_list[i], intervene_time_mean_list[i], xerr=accuracy_sem_list[i], yerr=intervene_time_sem_list[i], marker='o', capsize=5, label=experiment)

axes.set_xlim(0, 1.0)
axes.set_ylim(0, 6.0)
axes.set_xlabel('Intervention accuracy', fontsize=15)
axes.set_ylabel('Intervention time [s]', fontsize=15)
axes.legend(loc='lower left', fontsize=12)
axes.tick_params(axis='x', labelsize=12)
axes.tick_params(axis='y', labelsize=12)
# axes.figure.savefig('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/int_performance.svg', format="svg")
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
    intervene_speed_rate[i] = row / intervene_speed_rate[50]

axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[50],color="teal")
axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[40],color="turquoise")
axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[30],color="gold")
axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[20],color="lightsalmon")
axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[10],color="orangered")
# axes.figure.savefig('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/int_vel_bar.svg', format="svg")

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
    intervene_speed_rate[i] = row / intervene_speed_rate[50]

axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[50],color="teal", label='40-50km/h')
axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[40],color="turquoise", label='30-40km/h')
axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[30],color="gold", label='20-30km/h')
axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[20],color="lightsalmon", label='10-20km/h')
axes = sns.barplot(x=intervene_speed_rate.index, y=intervene_speed_rate[10],color="orangered", label='0-10km/h')
axes.set_ylabel('Rate of driving with minimum velocity while intervenition', fontsize=15)
axes.set_xlabel('Intervention method', fontsize=15)
axes.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
# axes.figure.savefig('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/min_vel_bar.svg', format="svg")
plt.show()

##################################
print('min vel')
#################################
pose_df = summary_df[summary_df.actor_action == 'pose']
for experiment in experiments:
    print(experiment, pose_df[pose_df.experiment_type==experiment].min_vel.mean())

print ('----', 'baseline', 'control', 'button', 'touch')
print('mean', accuracy_mean_list)
print('acc var',
    summary_df[(summary_df.experiment_type == 'BASELINE') & (summary_df.actor_action == "pose")].min_vel.mean(),
    summary_df[(summary_df.experiment_type == 'CONTROL') & (summary_df.actor_action == "pose")].min_vel.mean(),
    summary_df[(summary_df.experiment_type == "BUTTON") & (summary_df.actor_action == "pose")].min_vel.mean(),
    summary_df[(summary_df.experiment_type == "TOUCH") & (summary_df.actor_action == "pose")].min_vel.mean(),
    )
print('acc sd',
    summary_df[(summary_df.experiment_type == 'BASELINE') & (summary_df.actor_action == "pose")].min_vel.std(),
    summary_df[(summary_df.experiment_type == 'CONTROL') & (summary_df.actor_action == "pose")].min_vel.std(),
    summary_df[(summary_df.experiment_type == "BUTTON") & (summary_df.actor_action == "pose")].min_vel.std(),
    summary_df[(summary_df.experiment_type == "TOUCH") & (summary_df.actor_action == "pose")].min_vel.std(),
    )

axes = sns.boxplot(data=pose_df, x='experiment_type', y='min_vel', showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"}, order=experiments)
axes.set_ylabel('Minimum velocity [km/h]', fontsize=15)
axes.set_xlabel('Intervention method', fontsize=15)
axes.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()

for i, type in enumerate(experiments):
    min_vel_df = pd.DataFrame({
        "Count":[sum(summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel>-1.0) / summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel.count(),
                 sum(summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel>5.0) / summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel.count(),
                 sum(summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel>10.0) / summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel.count(),
                 sum(summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel>15.0) / summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel.count(),
                 sum(summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel>20.0) / summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel.count(),
                 sum(summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel>25.0) / summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel.count(),
                 sum(summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel>30.0) / summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel.count(),
                 sum(summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel>35.0) / summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel.count(),
                 sum(summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel>40.0) / summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel.count(),
                 sum(summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel>45.0) / summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel.count(),
                 sum(summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel>50.0) / summary_df[(summary_df.experiment_type == type) & (summary_df.actor_action == "pose")].min_vel.count()
                 ],
        "Velocity": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    })

    axes = sns.lineplot(data=min_vel_df, x='Velocity', y='Count', color=color.get(type), label=type, marker='o')
    axes.set_ylim(0, 1)
    axes.set_xlim([0,50])
    axes.set_xlabel('Velocity [km/h]', fontsize=15)
    axes.set_ylabel('Rate', fontsize=15)

axes.legend(loc='lower left', fontsize=12)
# axes.figure.savefig('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/min_vel.svg', format="svg")
plt.show()


min_vel_df = pd.DataFrame(columns=experiments, index=subjects)
for subject in subjects:
    for experiment in experiments:
        df = summary_df[(summary_df.subject == subject) & (summary_df.experiment_type == experiment)]
        time = df[(df.actor_action == "pose")].min_vel.dropna().mean()
        min_vel_df.at[subject, experiment] = time

_, norm_p1 = stats.shapiro(min_vel_df.BASELINE)
_, norm_p2 = stats.shapiro(min_vel_df.CONTROL)
_, norm_p3 = stats.shapiro(min_vel_df.BUTTON)
_, norm_p4 = stats.shapiro(min_vel_df.TOUCH)
_, var_p = stats.levene(min_vel_df.BASELINE, min_vel_df.CONTROL, min_vel_df.BUTTON, min_vel_df.TOUCH, center='median')
if norm_p1 < 0.05 or norm_p2 < 0.05 or norm_p3 < 0.05 or norm_p4 < 0.05:
    _, anova_p = stats.friedmanchisquare(min_vel_df.BASELINE, min_vel_df.CONTROL, min_vel_df.BUTTON, min_vel_df.TOUCH)
    print("anova(friedman test)", anova_p)
    if anova_p < 0.05:
        print(sp.posthoc_conover_friedman(min_vel_df))
else:
    melted_df = pd.melt(min_vel_df.reset_index(), id_vars="subject", var_name="experiment_type", value_name="min_vel")
    aov = stats_anova.AnovaRM(melted_df, "min_vel", "subject", ["experiment_type"])
    print("reperted anova: ", aov.fit())
    melted_df = pd.melt(min_vel_df, var_name="experiment_type", value_name="min_vel")
    if var_p < 0.05 and anova_p < 0.05:
        gamesHowellTest(melted_df, "min_vel", "experiment_type")
    elif var_p >= 0.05 and anova_p < 0.05:
        multicomp_result = multicomp.MultiComparison(np.array(melted_df.dropna(how='any').min_vel, dtype="float64"), melted_df.dropna(how='any').experiment_type)
        print(multicomp_result.tukeyhsd().summary())

subject_vel = min_vel_df.T
_, anova_p = stats.friedmanchisquare(
    subject_vel.ando,
    subject_vel.aso,
    subject_vel.hikosaka,
    subject_vel.ichiki,
    subject_vel.ienaga,
    subject_vel.ikai,
    subject_vel.isobe,
    subject_vel.ito,
    subject_vel.kato,
    subject_vel.matsubara,
    subject_vel.nakakuki,
    subject_vel.nakatani,
    subject_vel.negi,
    subject_vel.otake,
    subject_vel.sumiya,
    subject_vel.taga,
    subject_vel.yamamoto,
    subject_vel.yasuhara,
    )

print("subject wise anova = ", anova_p)

################################################################
print('max vel')
################################################################
max_vel_df = pd.DataFrame(columns=experiments, index=subjects)
for subject in subjects:
    for experiment in experiments:
        df = summary_df[(summary_df.subject == subject) & (summary_df.experiment_type == experiment)]
        time = df[(df.actor_action == "pose")].max_vel.dropna().mean()
        max_vel_df.at[subject, experiment] = time

_, norm_p1 = stats.shapiro(max_vel_df.BASELINE)
_, norm_p2 = stats.shapiro(max_vel_df.CONTROL)
_, norm_p3 = stats.shapiro(max_vel_df.BUTTON)
_, norm_p4 = stats.shapiro(max_vel_df.TOUCH)
_, var_p = stats.levene(max_vel_df.BASELINE, max_vel_df.CONTROL, max_vel_df.BUTTON, max_vel_df.TOUCH, center='median')
if norm_p1 < 0.05 or norm_p2 < 0.05 or norm_p3 < 0.05 or norm_p4 < 0.05:
    _, anova_p = stats.friedmanchisquare(max_vel_df.BASELINE, max_vel_df.CONTROL, max_vel_df.BUTTON, max_vel_df.TOUCH)
    print("anova(friedman test)", anova_p)
    if anova_p < 0.05:
        print(sp.posthoc_conover_friedman(max_vel_df))
else:
    _, anova_p = stats_anova.AnovaRM(max_vel_df.BASELINE, max_vel_df.CONTROL, max_vel_df.BUTTON, max_vel_df.TOUCH)
    print("reperted anova: ", anova_p)
    melted_df = pd.melt(max_vel_df, var_name="experiment_type", value_name="first_intervene_time")
    if var_p < 0.05 and anova_p < 0.05:
        gamesHowellTest(melted_df, "first_intervene_time", "experiment_type")
    elif var_p >= 0.05 and anova_p < 0.05:
        multicomp_result = multicomp.MultiComparison(np.array(melted_df.dropna(how='any').first_intervene_time, dtype="float64"), melted_df.dropna(how='any').experiment_type)
        print(multicomp_result.tukeyhsd().summary())

subject_vel = max_vel_df.T
_, anova_p = stats.friedmanchisquare(
    subject_vel.ando,
    subject_vel.aso,
    subject_vel.hikosaka,
    subject_vel.ichiki,
    subject_vel.ienaga,
    subject_vel.ikai,
    subject_vel.isobe,
    subject_vel.ito,
    subject_vel.kato,
    subject_vel.matsubara,
    subject_vel.nakakuki,
    subject_vel.nakatani,
    subject_vel.negi,
    subject_vel.otake,
    subject_vel.sumiya,
    subject_vel.taga,
    subject_vel.yamamoto,
    subject_vel.yasuhara,
    )
print("subject wise anova = ", anova_p)

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

# stop_rate = pd.DataFrame(index=subjects, columns=experiments)
# for subject in subjects:
#     pose_df = summary_df[(summary_df.subject == subject) & (summary_df.actor_action == "cross")]
#     for experiment in experiments:
#         buf = pose_df[pose_df.experiment_type==experiment].min_vel < 20.0
#         rate = buf.sum() / len(pose_df[pose_df.experiment_type==experiment])
#         stop_rate.at[subject, experiment] = rate
#
# axes = sns.boxplot(data=stop_rate, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
# plt.show()
#
# _, p = stats.levene(stop_rate.baseline, stop_rate.control, stop_rate.button, stop_rate.touch, center='median')
# print('levene-first stop rate', p)
# if p > 0.05:
#     melted_df = pd.melt(stop_rate, var_name='experiment_type', value_name='intervene_vel')
#     multicomp_result = multicomp.MultiComparison(np.array(melted_df['intervene_vel'], dtype="float64"), melted_df['experiment_type'])
#     print(multicomp_result.tukeyhsd().summary())

################################################################
print('keep rate')
################################################################

# stop_rate = pd.DataFrame(index=subjects, columns=experiments)
# for subject in subjects:
#     pose_df = summary_df[(summary_df.subject == subject) & (summary_df.actor_action == "pose")]
#     for experiment in experiments:
#         buf = pose_df[pose_df.experiment_type==experiment].min_vel > 30.0
#         rate = buf.sum() / len(pose_df[pose_df.experiment_type==experiment])
#         stop_rate.at[subject, experiment] = rate
#
# axes = sns.boxplot(data=stop_rate, showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
# plt.show()
#
# _, p = stats.levene(stop_rate.baseline, stop_rate.control, stop_rate.button, stop_rate.touch, center='median')
# print('levene-first keep rate', p)
# if p > 0.05:
#     melted_df = pd.melt(stop_rate, var_name='experiment_type', value_name='intervene_vel')
#     multicomp_result = multicomp.MultiComparison(np.array(melted_df['intervene_vel'], dtype="float64"), melted_df['experiment_type'])
#     print(multicomp_result.tukeyhsd().summary())

################################################################
print('nasa-tlx')
################################################################
#### nasa-tlx ####
for item in ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'entire']:
    print(item)
    _, norm_p1 = stats.shapiro(nasa_df[nasa_df.experiment_type == "BASELINE"][item])
    _, norm_p2 = stats.shapiro(nasa_df[nasa_df.experiment_type == "CONTROL"][item])
    _, norm_p3 = stats.shapiro(nasa_df[nasa_df.experiment_type == "BUTTON"][item])
    _, norm_p4 = stats.shapiro(nasa_df[nasa_df.experiment_type == "TOUCH"][item])
    _, var_p = stats.levene(
        nasa_df[nasa_df.experiment_type == "BASELINE"][item],
        nasa_df[nasa_df.experiment_type == "CONTROL"][item],
        nasa_df[nasa_df.experiment_type == "BUTTON"][item],
        nasa_df[nasa_df.experiment_type == "TOUCH"][item],
        center='median'
        )

    if norm_p1 < 0.05 or norm_p2 < 0.05 or norm_p3 < 0.05 or norm_p4 < 0.05:
        _, anova_p = stats.friedmanchisquare(
            nasa_df[nasa_df.experiment_type == "BASELINE"][item],
            nasa_df[nasa_df.experiment_type == "CONTROL"][item],
            nasa_df[nasa_df.experiment_type == "BUTTON"][item],
            nasa_df[nasa_df.experiment_type == "TOUCH"][item],
        )
        print("anova(friedman test)", anova_p)
        if anova_p < 0.05:
            print(sp.posthoc_conover(nasa_df, val_col=item, group_col="experiment_type"))
    else:
        melted_df = pd.melt(nasa_df, id_vars=["name", "experiment_type"],  var_name="type", value_name="rate")
        aov = stats_anova.AnovaRM(melted_df[melted_df.type == item], "rate", "name", ["experiment_type"])
        print("reperted anova: ", aov.fit())
        multicomp_result = multicomp.MultiComparison(nasa_df[item], nasa_df.experiment_type)
        print(multicomp_result.tukeyhsd().summary())



melted_df = pd.melt(nasa_df, id_vars=nasa_df.columns.values[:2], var_name="args", value_name="value")
# plot = sns.boxplot(x='args', y="value", hue="experiment_type", data=melted_df,showmeans=True, meanline=True, meanprops={"linestyle":"--", "color":"Red"})
axes = sns.barplot(x='args', y="value", hue="experiment_type", data=melted_df)
axes.set_ylim([0, 10])
axes.set_ylabel('Workload Rating', fontsize=15)
axes.set_xlabel('Scale', fontsize=15)
axes.tick_params(axis='x', labelsize=12)
axes.tick_params(axis='y', labelsize=12)
axes.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
# axes.figure.savefig('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/nasa-tlx.svg', format="svg")
plt.show()
