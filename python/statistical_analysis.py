#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.sandbox.stats import multicomp
import scikit_posthocs as sp
from itertools import combinations



def my_multicomp(df, group_col, val_col):

    flag = 0 # 0:parametric(equal_norm, equal_var) 1:non-parametric(equal_norm) 2:non-parametric

    groups = df[group_col].drop_duplicates()
    norm_list = []
    for group in groups:
        _, norm_p = stats.shapiro(df[val_col].dropna())
        norm_list.append(norm_p)

    if all([i > 0.05 for i in norm_list]):
        _, var_p = stats.levene(
            *[df[df[group_col] == group][val_col].dropna() for group in groups],
            center='median'
            )

        if var_p > 0.05:
            flag = 0
        else:
            flag = 1

    else:
        flag = 2


    if flag == 0:
        melted_df = pd.melt(intervene_time.reset_index(), id_vars="subject", var_name="experiment_type", value_name="first_intervene_time")
        anova_result = stats_anova.AnovaRM(melted_df, "first_intervene_time", "subject", ["experiment_type"])
        print("reperted anova\n", anova_result.fit())

        multicomp_result = multicomp.MultiComparison(np.array(df.dropna(how='any')[val_col], dtype="float64"), df.dropna(how='any')[group_col])
        print('TukeyHSD\n', multicomp_result.tukeyhsd().summary())

    elif flag == 1:
        gamesHowellTest(df, val_col, group_col)

    elif flag == 2:
        print('Steel-Dwass-Critchlow-Fligner test (dscf)\n', sp.posthoc_dscf(df, val_col=val_col, group_col=group_col))
        print("Conover Iman test after anova_result:", anova_p, sp.posthoc_conover_friedman(inttype_accuracy))



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
