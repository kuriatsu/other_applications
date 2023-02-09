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
from statsmodels.formula.api import ols
import statsmodels.stats.anova as stats_anova
from statsmodels.sandbox.stats import multicomp
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import cochrans_q
from scipy.stats import f_oneway
import scikit_posthocs as sp

sns.set(context='paper', style='whitegrid')
color = {'BASELINE':'#add8e6', 'CONTROL': '#7dbeb5', 'BUTTON': '#388fad', 'TOUCH': '#335290'}
sns.set_palette(sns.color_palette(color.values()))

summary_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/summary_rm_wrong.csv')
nasa_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/nasa-tlx.csv')
rank_df = pd.read_csv('/media/kuriatsu/SamsungKURI/master_study_bag/202102experiment/result/rank.csv')
subjects = summary_df.subject.drop_duplicates()
experiments = summary_df.experiment_type.drop_duplicates()


################################################################
print('intervene accuracy')
################################################################
inttype_accuracy = pd.DataFrame(columns=["CONTROL", "BUTTON", "TOUCH"], index=subjects)
for subject in summary_df.subject.drop_duplicates():
    buf_dict = {}
    for experiment in summary_df.experiment_type.drop_duplicates():
        buf = summary_df[(summary_df.subject == subject)&(summary_df.experiment_type==experiment)]
        cross_correct = buf[(buf.actor_action=="cross")].intervene_vel.isnull().sum()
        pose_correct = len(buf[(buf.actor_action=="pose")&(buf.min_vel>1.0)])
        cross_acc = cross_correct/len(buf[buf.actor_action=="cross"])
        total_acc = (cross_correct+pose_correct)/ len(buf)
        buf_dict[experiment] = total_acc
        print(f"{subject}, {experiment}: pose_correct={total_acc}")

    # std = np.std(list(buf_dict.values()))
    # mean = np.mean(list(buf_dict.values()))
    inttype_accuracy.loc[subject] = [
        # (buf_dict["TOUCH"] - buf_dict["BASELINE"])/buf_dict["BASELINE"]
        # (buf_dict["TOUCH"] - mean)/std - (buf_dict["BASELINE"] - mean)/std,
        buf_dict["CONTROL"] - buf_dict["BASELINE"],
        buf_dict["BUTTON"] - buf_dict["BASELINE"],
        buf_dict["TOUCH"] - buf_dict["BASELINE"]
        ]

inttype_accuracy.describe()

_, norm_p2 = stats.shapiro(inttype_accuracy.CONTROL)
_, norm_p3 = stats.shapiro(inttype_accuracy.BUTTON)
_, norm_p4 = stats.shapiro(inttype_accuracy.TOUCH)
_, var_p = stats.levene(inttype_accuracy.CONTROL, inttype_accuracy.BUTTON, inttype_accuracy.TOUCH, center='median')

if norm_p2 < 0.05 or norm_p3 < 0.05 or norm_p4 < 0.05:
    _, anova_p = stats.friedmanchisquare(inttype_accuracy.CONTROL, inttype_accuracy.BUTTON, inttype_accuracy.TOUCH)
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

inttype_accuracy.CONTROL.mean()
inttype_accuracy.BUTTON.mean()
inttype_accuracy.TOUCH.mean()
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
for subject in summary_df.subject.drop_duplicates():
    buf_dict = {}
    for experiment in summary_df.experiment_type.drop_duplicates():
        buf = summary_df[(summary_df.subject == subject)&(summary_df.experiment_type==experiment)]
        buf_dict[experiment] = buf[(buf.actor_action == "pose") & (buf.intervene_vel > 1.0)].first_intervene_time.dropna().mean()

    std = np.std(list(buf_dict.values()))
    mean = np.mean(list(buf_dict.values()))
    intervene_time.loc[subject] = [
        # (buf_dict["TOUCH"] - buf_dict["BASELINE"])/buf_dict["BASELINE"]
        (buf_dict["BASELINE"] - mean)/std,
        (buf_dict["CONTROL"] - mean)/std,
        (buf_dict["BUTTON"] - mean)/std,
        (buf_dict["TOUCH"] - mean)/std,
        ]

_, norm_p1 = stats.shapiro(intervene_time.BASELINE)
_, norm_p2 = stats.shapiro(intervene_time.CONTROL)
_, norm_p3 = stats.shapiro(intervene_time.BUTTON)
_, norm_p4 = stats.shapiro(intervene_time.TOUCH)
_, var_p = stats.levene(intervene_time.CONTROL, intervene_time.CONTROL, intervene_time.BUTTON, intervene_time.TOUCH, center='median')
# _, var_p = stats.levene(intervene_time.BASELINE, intervene_time.CONTROL, intervene_time.BUTTON, intervene_time.TOUCH, center='median')
if norm_p1<0.05 or norm_p2 < 0.05 or norm_p3 < 0.05 or norm_p4 < 0.05:
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

################################################################
print('intervene time')
################################################################
intervene_time = pd.DataFrame(columns=experiments, index=subjects)
for subject in summary_df.subject.drop_duplicates():
    buf_dict = {}
    for experiment in summary_df.experiment_type.drop_duplicates():
        buf = summary_df[(summary_df.subject == subject)&(summary_df.experiment_type==experiment)]
        buf_dict[experiment] = buf[(buf.actor_action == "pose") & (buf.intervene_vel > 1.0)].first_intervene_time.dropna().mean()

    std = np.std(list(buf_dict.values()))
    mean = np.mean(list(buf_dict.values()))
    intervene_time.loc[subject] = [
        # (buf_dict["TOUCH"] - buf_dict["BASELINE"])/buf_dict["BASELINE"]
        (buf_dict["BASELINE"] - mean)/std,
        (buf_dict["CONTROL"] - mean)/std,
        (buf_dict["BUTTON"] - mean)/std,
        (buf_dict["TOUCH"] - mean)/std,
        ]

_, norm_p1 = stats.shapiro(intervene_time.BASELINE)
_, norm_p2 = stats.shapiro(intervene_time.CONTROL)
_, norm_p3 = stats.shapiro(intervene_time.BUTTON)
_, norm_p4 = stats.shapiro(intervene_time.TOUCH)
_, var_p = stats.levene(intervene_time.CONTROL, intervene_time.CONTROL, intervene_time.BUTTON, intervene_time.TOUCH, center='median')
# _, var_p = stats.levene(intervene_time.BASELINE, intervene_time.CONTROL, intervene_time.BUTTON, intervene_time.TOUCH, center='median')
if norm_p1<0.05 or norm_p2 < 0.05 or norm_p3 < 0.05 or norm_p4 < 0.05:
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

################################################################
print('min vel')
################################################################
min_vel_df = pd.DataFrame(columns=experiments, index=subjects)
for subject in summary_df.subject.drop_duplicates():
    buf_dict = {}
    for experiment in summary_df.experiment_type.drop_duplicates():
        buf = summary_df[(summary_df.subject == subject)&(summary_df.experiment_type==experiment)]
        buf_dict[experiment] = buf[(buf.actor_action == "pose")].std_vel.dropna().mean()

    std = np.std(list(buf_dict.values()))
    mean = np.mean(list(buf_dict.values()))
    min_vel_df.loc[subject] = [
        # (buf_dict["TOUCH"] - buf_dict["BASELINE"])/buf_dict["BASELINE"]
        (buf_dict["BASELINE"] - mean)/std,
        (buf_dict["CONTROL"] - mean)/std,
        (buf_dict["BUTTON"] - mean)/std,
        (buf_dict["TOUCH"] - mean)/std,
        ]

_, norm_p1 = stats.shapiro(min_vel_df.BASELINE)
_, norm_p2 = stats.shapiro(min_vel_df.CONTROL)
_, norm_p3 = stats.shapiro(min_vel_df.BUTTON)
_, norm_p4 = stats.shapiro(min_vel_df.TOUCH)
_, var_p = stats.levene(min_vel_df.CONTROL, min_vel_df.CONTROL, min_vel_df.BUTTON, min_vel_df.TOUCH, center='median')
# _, var_p = stats.levene(min_vel.BASELINE, min_vel.CONTROL, min_vel.BUTTON, min_vel.TOUCH, center='median')
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
        multicomp_result = multicomp.MultiComparison(np.array(melted_df.dropna(how='any').std_vel, dtype="float64"), melted_df.dropna(how='any').experiment_type)
        print(multicomp_result.tukeyhsd().summary())
