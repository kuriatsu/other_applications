#! /usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


result_df = pd.read_csv('/home/kuriatsu/Documents/experiment_data/PIE_experiment_june/result.csv',
                        dtype={'intervene_speed':'float', 'prob':'float'})
subjects = result_df.subject.drop_duplicates()

sns.set(context='paper', style='whitegrid')
color = {'baseline':'#d5e7ba', 'control': '#7dbeb5', 'button': '#388fad', 'touch': '#335290'}
color_list = ['#388fad', '#335290']
sns.set_palette(sns.color_palette(color_list))

accuracy_df = pd.DataFrame(columns=["BUTTON", "TOUCH"], index=subjects)
for subject in subjects:
    button_df = result_df[(result_df.subject == subject) & (result_df.experiment_type == "BUTTON")]
    push_collect = len(button_df[(button_df.intervene_type == "pushed") & (button_df.prob < 0.5)])
    push_collect += len(button_df[(button_df.intervene_type == "passed") & (button_df.prob >= 0.5)])
    touch_df = result_df[(result_df.subject == subject) & (result_df.experiment_type == "TOUCH")]
    touch_collect = len(touch_df[(touch_df.intervene_type == "touched") & (touch_df.prob < 0.5)])
    touch_collect += len(touch_df[(touch_df.intervene_type == "passed") & (touch_df.prob >= 0.5)])
    accuracy_df.at[subject, "BUTTON"] = push_collect / len(button_df)
    accuracy_df.at[subject, "TOUCH"] = touch_collect / len(touch_df)

_, norm_p1 = stats.shapiro(accuracy_df.BUTTON)
_, norm_p2 = stats.shapiro(accuracy_df.TOUCH)
_, var_p = stats.levene(accuracy_df.BUTTON, accuracy_df.TOUCH, center='median')
if norm_p1 < 0.05 or norm_p2 < 0.05 or var_p < 0.05 :
    print('wilcoxon\n', stats.wilcoxon(x=accuracy_df.BUTTON, y=accuracy_df.TOUCH))
else:
    print('t test\n', stats.ttest_ind(accuracy_df.BUTTON, accuracy_df.TOUCH, equal_var = (var_p < 0.05)))



intervene_time_df = pd.DataFrame(columns=["BUTTON", "TOUCH"], index=subjects)
for subject in subjects:
    button_df = result_df[(result_df.subject == subject) & (result_df.experiment_type == "BUTTON")]
    push_time = button_df.intervene_speed.dropna().mean()
    touch_df = result_df[(result_df.subject == subject) & (result_df.experiment_type == "TOUCH")]
    touch_time = touch_df.intervene_speed.dropna().mean()
    intervene_time_df.at[subject, "BUTTON"] = push_time
    intervene_time_df.at[subject, "TOUCH"] = touch_time

_, norm_p1 = stats.shapiro(intervene_time_df.BUTTON)
_, norm_p2 = stats.shapiro(intervene_time_df.TOUCH)
_, var_p = stats.levene(intervene_time_df.BUTTON, intervene_time_df.TOUCH, center='median')
if norm_p1 < 0.05 or norm_p2 < 0.05 or var_p < 0.05:
    print('wilcoxon\n', stats.wilcoxon(x=intervene_time_df.BUTTON, y=intervene_time_df.TOUCH))
else:
    print('t test \n', stats.ttest_rel(intervene_time_df.BUTTON, intervene_time_df.TOUCH))

accuracy_mean_list = [
    accuracy_df.BUTTON.mean(),
    accuracy_df.TOUCH.mean(),
]

accuracy_sem_list = [
    accuracy_df.BUTTON.sem(),
    accuracy_df.TOUCH.sem(),
]

intervene_time_mean_list = [
    intervene_time_df.BUTTON.mean(),
    intervene_time_df.TOUCH.mean(),
]

intervene_time_sem_list = [
    intervene_time_df.BUTTON.sem(),
    intervene_time_df.TOUCH.sem(),
]

fig, axes = plt.subplots()
for i, experiment in enumerate(["BUTTON", "TOUCH"]):
    axes.errorbar(accuracy_mean_list[i], intervene_time_mean_list[i], xerr=accuracy_sem_list[i], yerr=intervene_time_sem_list[i], marker='o', capsize=5, label=experiment)

x1 = accuracy_mean_list[0]
x2 = accuracy_mean_list[1]
y = min(intervene_time_mean_list[0], intervene_time_mean_list[0]) - 0.2


axes.set_xlim(0.0, 1.0)
axes.set_ylim(0.0, 2.2)
axes.set_xlabel("Intervention accuracy", fontsize=15)
axes.set_ylabel("Intervention time [s]", fontsize=15)
axes.legend(loc="lower left", fontsize=12)
axes.tick_params(axis='x', labelsize=12)
axes.tick_params(axis='y', labelsize=12)
plt.show()
fig.savefig("/home/kuriatsu/Documents/experiment_data/PIE_experiment_june/int_performance.svg", format="svg")

axes = sns.histplot(result_df.prob)

prob_acc_df = pd.DataFrame(columns=["prob","rate", "experiment_type"])
index=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for i in index:
    button_df = result_df[(result_df.experiment_type=="BUTTON") & (result_df.prob < i) & ((i-0.1) <= result_df.prob)]
    touch_df = result_df[(result_df.experiment_type=="TOUCH") & (result_df.prob < i) & ((i-0.1) <= result_df.prob)]
    if i < 0.5:
        if len(button_df) > 0:
            buf_df = pd.DataFrame([(i, len(button_df[button_df.intervene_type == "pushed"]) / len(button_df), "BUTTON")], columns=["prob", "rate", "experiment_type"])
            prob_acc_df = prob_acc_df.append(buf_df, ignore_index=True)
        else:
            buf_df = pd.DataFrame([(i, 0, "BUTTON")], columns=["prob", "rate", "experiment_type"])
            prob_acc_df = prob_acc_df.append(buf_df, ignore_index=True)
        if len(touch_df) > 0:
            buf_df = pd.DataFrame([(i, len(touch_df[touch_df.intervene_type == "touched"]) / len(touch_df), "TOUCH")], columns=["prob", "rate", "experiment_type"])
            prob_acc_df = prob_acc_df.append(buf_df, ignore_index=True)
        else:
            buf_df = pd.DataFrame([(i, 0, "TOUCH")], columns=["prob", "rate", "experiment_type"])
            prob_acc_df = prob_acc_df.append(buf_df, ignore_index=True)
    else:
        if len(button_df) > 0:
            buf_df = pd.DataFrame([(i, len(button_df[button_df.intervene_type == "passed"]) / len(button_df), "BUTTON")], columns=["prob", "rate", "experiment_type"])
            prob_acc_df = prob_acc_df.append(buf_df, ignore_index=True)
        else:
            buf_df = pd.DataFrame([(i, 0, "BUTTON")], columns=["prob", "rate", "experiment_type"])
            prob_acc_df = prob_acc_df.append(buf_df, ignore_index=True)

        if len(touch_df) > 0:
            buf_df = pd.DataFrame([(i, len(touch_df[touch_df.intervene_type == "passed"]) / len(touch_df), "TOUCH")], columns=["prob", "rate", "experiment_type"])
            prob_acc_df = prob_acc_df.append(buf_df, ignore_index=True)
        else:
            buf_df = pd.DataFrame([(i, 0, "TOUCH")], columns=["prob", "rate", "experiment_type"])
            prob_acc_df = prob_acc_df.append(buf_df, ignore_index=True)

# sns.scatterplot(data=prob_acc_df, x="prob", y="rate", hue="experiment_type", ax = axes)
sns.scatterplot(data=prob_acc_df, x="prob", y="rate", hue="experiment_type")
