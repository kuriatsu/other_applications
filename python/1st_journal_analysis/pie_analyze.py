#! /usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


result_df = pd.read_csv('/home/kuriatsu/Dropbox/data/PIE_experiment_june/data/summary.csv',
                        dtype={'intervene_speed':'float', 'prob':'float'})
subjects = result_df.subject.drop_duplicates()

sns.set(context='paper', style='whitegrid')
color = {'baseline':'#d5e7ba', 'control': '#7dbeb5', 'button': '#388fad', 'touch': '#335290'}
color_list = ['#388fad', '#335290']
sns.set_palette(sns.color_palette(color_list))

# intervention accuracy
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
if norm_p1 < 0.05 or norm_p2 < 0.05:
    print('wilcoxon\n', stats.wilcoxon(x=accuracy_df.BUTTON, y=accuracy_df.TOUCH))
else:
    print('t test\n', stats.ttest_ind(accuracy_df.BUTTON, accuracy_df.TOUCH, equal_var = (var_p < 0.05)))

accuracy_df["BUTTON"].mean()

#  how early intervention and acc
correct_int_remain_time = []
false_int_remain_time = []
for row in result_df.itertuples():
    if (row.intervene_type in ["pushed", "touched"]):
        if row.prob <= 0.5:
            correct_int_remain_time.append((row.critical_point - row.intervene_frame)/30)
        else:
            false_int_remain_time.append((row.critical_point - row.intervene_frame)/30)
_, norm_correct = stats.shapiro(correct_int_remain_time)
_, norm_false = stats.shapiro(false_int_remain_time)
_, var_p = stats.levene(correct_int_remain_time, false_int_remain_time, center='median')

print('wilcoxon\n', stats.mannwhitneyu(correct_int_remain_time, false_int_remain_time, alternative="two-sided").pvalue)
print(f"correct_int_remain_time:{sum(correct_int_remain_time)/len(correct_int_remain_time)}, false_int_remain_time:{sum(false_int_remain_time)/len(false_int_remain_time)}")
sns.histplot(correct_int_remain_time)
sns.histplot(false_int_remain_time)

#  intervene able time and acc
correct_int_remain_time = []
false_int_remain_time = []
for row in result_df.itertuples():
    if (row.intervene_type in ["pushed", "touched"]):
        if row.prob <= 0.5:
            correct_int_remain_time.append((row.critical_point - row.display_frame)/30)
        else:
            false_int_remain_time.append((row.critical_point - row.display_frame)/30)
_, norm_correct = stats.shapiro(correct_int_remain_time)
_, norm_false = stats.shapiro(false_int_remain_time)
_, var_p = stats.levene(correct_int_remain_time, false_int_remain_time, center='median')

print('wilcoxon\n', stats.mannwhitneyu(correct_int_remain_time, false_int_remain_time, alternative="two-sided").pvalue)
print(f"correct_int_remain_time:{sum(correct_int_remain_time)/len(correct_int_remain_time)}, false_int_remain_time:{sum(false_int_remain_time)/len(false_int_remain_time)}")
sns.histplot(correct_int_remain_time)
sns.histplot(false_int_remain_time)


# acc vs prob
correct_int_remain_time = []
false_int_remain_time = []
for row in result_df.itertuples():
    if (row.intervene_type in ["pushed", "touched"] and row.prob <= 0.5) or (row.intervene_type == "passed" and row.prob > 0.5):
        correct_int_remain_time.append(row.prob)
    else:
        false_int_remain_time.append(row.prob)
_, norm_correct = stats.shapiro(correct_int_remain_time)
_, norm_false = stats.shapiro(false_int_remain_time)
_, var_p = stats.levene(correct_int_remain_time, false_int_remain_time, center='median')

print('wilcoxon\n', stats.mannwhitneyu(correct_int_remain_time, false_int_remain_time, alternative="two-sided").pvalue)
print(f"correct_int_remain_time:{sum(correct_int_remain_time)/len(correct_int_remain_time)}, false_int_remain_time:{sum(false_int_remain_time)/len(false_int_remain_time)}")
fig, ax = plt.subplots()
sns.histplot(correct_int_remain_time, ax=ax)
sns.histplot(false_int_remain_time, ax=ax)
intervene_time_df = pd.DataFrame(columns=["BUTTON", "TOUCH"], index=subjects)

# acc in prob > 0.5 vs acc in pron <=0.5
database = pd.DataFrame(columns=["cross", "nocross"])
for subject in subjects:
    acc_cross = []
    acc_nocross = []
    for row in result_df[result_df.subject == subject].itertuples():
        if row.prob > 0.5:
            acc_cross.append(row.intervene_type == "passed")
        elif row.prob <= 0.5:
            acc_nocross.append(row.intervene_type in ["pushed", "touched"])
        # elif 0.2 <= row.prob <= 0.5:
        #     acc_vague_nocross.append(row.intervene_type in ["pushed", "touched"])
        # elif 0.5 < row.prob <= 0.8:
        #     acc_vague_cross.append(row.intervene_type == "passed")

    buf = pd.Series([sum(acc_cross)/len(acc_cross), sum(acc_nocross)/len(acc_nocross)], index=database.columns)
    database = database.append(buf, ignore_index=True)

print(f"cross-mean:{database.cross.mean()}, cross-std:{database.cross.std()}, nocross-mean:{database.nocross.mean()}, nocross-std:{database.nocross.std()}")


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

accuracy_sem_list = [
    accuracy_df.BUTTON.std(),
    accuracy_df.TOUCH.std(),
]

intervene_time_mean_list = [
    intervene_time_df.BUTTON.mean(),
    intervene_time_df.TOUCH.mean(),
]

intervene_time_sem_list = [
    intervene_time_df.BUTTON.sem(),
    intervene_time_df.TOUCH.sem(),
]

_, axes = plt.subplots()
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
