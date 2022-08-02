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


def show_mean_var(df, actor_action, col):
    print(f"target:{col}")
    target_df = df[df.actor_action == actor_action]
    mean = [
        target_df[(target_df.experiment_type == 'BASELINE')][col].dropna().mean(),
        target_df[(target_df.experiment_type == 'CONTROL')][col].dropna().mean(),
        target_df[(target_df.experiment_type == 'BUTTON')][col].dropna().mean(),
        target_df[(target_df.experiment_type == 'TOUCH')][col].dropna().mean(),
        ]

    var = [
        target_df[(target_df.experiment_type == 'BASELINE')][col].dropna().std(),
        target_df[(target_df.experiment_type == 'CONTROL')][col].dropna().std(),
        target_df[(target_df.experiment_type == 'BUTTON')][col].dropna().std(),
        target_df[(target_df.experiment_type == 'TOUCH')][col].dropna().std(),
        ]

    sem = [
        target_df[(target_df.experiment_type == 'BASELINE')][col].dropna().sem(),
        target_df[(target_df.experiment_type == 'CONTROL')][col].dropna().sem(),
        target_df[(target_df.experiment_type == 'BUTTON')][col].dropna().sem(),
        target_df[(target_df.experiment_type == 'TOUCH')][col].dropna().sem(),
    ]


    _, norm_p = stats.shapiro(target_df[col].dropna())
    _, var_p = stats.levene(
        target_df[target_df.experiment_type == "BASELINE"][col].dropna(),
        target_df[target_df.experiment_type == "CONTROL"][col].dropna(),
        target_df[target_df.experiment_type == "BUTTON"][col].dropna(),
        target_df[target_df.experiment_type == "TOUCH"][col].dropna(),
        )
    print ('----', 'BASELINE', 'CONTROL', 'BUTTON', 'TOUCH')
    print(mean)
    print(var)

    if norm_p > 0.05 and var_p > 0.05:
        melted_df = pd.melt(inttype_accuracy.reset_index(), id_vars="subject", var_name="experiment_type", value_name=col)
        anova_result = stats_anova.AnovaRM(melted_df, col, "subject", ["experiment_type"])
        print("reperted anova RM: ", anova_result.fit())
    elif norm_p > 0.05 and var_p < 0.05:
        _, anova_p = stats.friedmanchisquare(inttype_accuracy.BASELINE, inttype_accuracy.CONTROL, inttype_accuracy.BUTTON, inttype_accuracy.TOUCH)
        print("freadmanchisquare anova: ", anova_p)
    else:
        {...}

    return mean, sem


sns.set(context='paper', style='whitegrid')
color = {'BASELINE':'#add8e6', 'CONTROL': '#7dbeb5', 'BUTTON': '#388fad', 'TOUCH': '#335290'}
sns.set_palette(sns.color_palette(color.values()))

summary_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/summary_rm_wrong.csv')
nasa_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/nasa-tlx.csv')
rank_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/rank.csv')
with open('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/profile.pickle', 'rb') as f:
    profile_list = pickle.load(f)

# summary_df = pd.read_csv('/home/kuriatsu/Documents/experiment/carla_202102_result/summary.csv')
# summary_df = pd.read_csv('/home/kuriatsu/Documents/experiment/carla_202102_result/summary_rm_wrong.csv')
# nasa_df = pd.read_csv('/home/kuriatsu/Documents/experiment/carla_202102_result/nasa-tlx.csv')
# rank_df = pd.read_csv('/home/kuriatsu/Documents/experiment/carla_202102_result/rank.csv')

subjects = summary_df.subject.drop_duplicates()
experiments = ['BASELINE', 'CONTROL', 'BUTTON', 'TOUCH']



################################################################
print('min_speed ')
################################################################
show_mean_var(summary_df, "pose", "min_vel")

################################################################
print('std_speed ')
################################################################
show_mean_var(summary_df, "pose", "std_vel")


################################################################
print('intervention speed ')
################################################################
int_speed_mean, int_speed_sem = show_mean_var(summary_df, "pose", "first_intervene_time")


################################################################
print('intervention duration ')
################################################################
int_dur_mean, int_dur_sem = show_mean_var(summary_df, "pose", "intervene_duration")

fig, axes = plt.subplots()
for i, experiment in enumerate(experiments):
    axes.errorbar(int_speed_mean[i], int_dur_mean[i], xerr=int_dur_sem[i], yerr=ttc_sem[i], marker='o', capsize=5, label=experiment)

axes.set_xlim(0, 1.0)
axes.set_ylim(0, 6.0)
axes.set_xlabel('Intervention speed [s]', fontsize=15)
axes.set_ylabel('Intervention duration [s]', fontsize=15)
axes.legend(loc='lower left', fontsize=12)
axes.tick_params(axis='x', labelsize=12)
axes.tick_params(axis='y', labelsize=12)
# axes.figure.savefig('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/int_performance.svg', format="svg")
plt.show()

################################################################
print('mean_speed ')
################################################################
speed_mean, speed_sem = show_mean_var(summary_df, "pose", "mean_vel")

################################################################
print('ttc ')
################################################################
ttc_mean, ttc_sem = show_mean_var(summary_df, "cross", "min_ttc")

fig, axes = plt.subplots()
for i, experiment in enumerate(experiments):
    axes.errorbar(speed_mean[i], ttc_mean[i], xerr=speed_sem[i], yerr=ttc_sem[i], marker='o', capsize=5, label=experiment)

axes.set_xlim(0, 1.0)
axes.set_ylim(0, 6.0)
axes.set_xlabel('Average speed [km/h]', fontsize=15)
axes.set_ylabel('TTC to cross pedestrian [s]', fontsize=15)
axes.legend(loc='lower left', fontsize=12)
axes.tick_params(axis='x', labelsize=12)
axes.tick_params(axis='y', labelsize=12)
# axes.figure.savefig('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/int_performance.svg', format="svg")
plt.show()


################################################################
print('profile ')
################################################################
plot_col_list = {"pose":0, "cross":1}
plot_col_list_inv = {0:"STAND", 1:"CROSS"}
plot_list = {}
fig, axes = plt.subplots(4, 2)
for i, experiment_type in enumerate(["BASELINE", "CONTROL", "BUTTON", "TOUCH"]):
    plot_list[experiment_type] = axes[i, :]
    for j in range(0, len(axes[i,:])):
        axes[i, j].set_xlim([50, -10])
        axes[i, j].set_ylim([0, 60])
        axes[i, j].set_xlabel(f"Distance from pedestrian[m]\n{experiment_type}-{plot_col_list_inv[j]}", fontsize=12)
        axes[i, j].set_ylabel("Velocity[km/h]", fontsize=12)


count = 0
lines = [0]*4
for profile in profile_list:
    count+=1
    if count > 12:
        break

    profile["y"] = profile["y"] * 3.6
    ax = plot_list[profile["experiment_type"]][plot_col_list[profile["actor_action"]]]

    if profile["min_vel"] > 10.0:
        plot_start = profile["int_start"] if profile["int_start"] is not None else 0
        plot_end = profile["int_end"] if profile["int_end"] is not None else 0
        lines[0] = sns.lineplot(x=profile["x"][0:plot_start+4], y=profile["y"][0:plot_start+4], ax=ax, color="teal", alpha=0.5, ci=2, label="keep speed - automation")
        lines[1] = sns.lineplot(x=profile["x"][plot_start:plot_end+4], y=profile["y"][plot_start:plot_end+4], ax=ax, color="teal", alpha=0.5, linestyle=":", ci=2, label="keep speed - intervention")
        sns.lineplot(x=profile["x"][plot_end:-1], y=profile["y"][plot_end:-1], ax=ax, color="teal", alpha=0.5, ci=2)
        ax.get_legend().set_visible(False)
        # sns.lineplot(x=profile["x"], y=profile["y"], ax=ax, color="orangered", alpha=0.5)
        # sns.scatterplot(x=profile["x"][plot_start:plot_end], y=profile["y"][plot_start:plot_end], ax=ax, color="black", s=5, marker="x")
    else:
        plot_start = profile["int_start"] if profile["int_start"] is not None else 0
        plot_end = profile["int_end"] if profile["int_end"] is not None else 0
        lines[2] = sns.lineplot(x=profile["x"][0:plot_start+4], y=profile["y"][0:plot_start+4], ax=ax, color="orangered", alpha=0.5, ci=2, label="yield - automation")
        lines[3] = sns.lineplot(x=profile["x"][plot_start:plot_end+4], y=profile["y"][plot_start:plot_end+4], ax=ax, color="orangered", alpha=0.5, linestyle=":", ci=2, label="yield - intervention")
        sns.lineplot(x=profile["x"][plot_end:-1], y=profile["y"][plot_end:-1], ax=ax, color="orangered", alpha=0.5, ci=2)
        ax.get_legend().set_visible(False)
        # sns.lineplot(x=profile["x"], y=profile["y"], ax=ax, color="orangered", alpha=0.5)
        # sns.scatterplot(x=profile["x"][plot_start:plot_end], y=profile["y"][plot_start:plot_end], ax=ax, color="black", s=5, marker="x")

# handler, label = plot_list["TOUCH"][0].get_legend_handles_labels()
handler_list = []
label_list = []
for ax in fig.axes:
    handlers, labels = ax.get_legend_handles_labels()
    for handler, label in zip(handlers, labels):
        if label not in label_list:
            handler_list.append(handler)
            label_list.append(label)


plt.legend(handles=handler_list, labels=label_list, ncol=2, loc="upper right", bbox_to_anchor=(1, -0.4), fontsize=12)
# plot_list["TOUCH"][1].legend(loc="lower center", bbox_to_anchor=(0.0, 1.0),fontsize=12)

plt.show()
