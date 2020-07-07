#!/usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import sys
import pandas as pd
import matplotlib.pyplot as plt
# pd.set_option('display.mpl_style', 'default')
# pd.set_option('display.max_width', 5000)
# pd.set_option('display.max_columns', 60)

data = pd.read_csv('/home/kuriatsu/DropboxKuri/data/PIE_experiment/obj_statistic.csv', index_col='id')
print(data['frame'])
data = data.drop(columns=['frame', 'clarity', 'acc'])
data = data.sort_values('prob_anno')
data.plot(kind='bar')
plt.ylabel('crossing intention')
plt.show()
