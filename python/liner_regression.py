#! /usr/bin/python3
# -*- coding:utf-8 -*-

import csv
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

def readCsv(filename):
    with open(filename, 'r') as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader)
        return [row for row in reader], header


def dataScaling(x, y):
    sscaler = preprocessing.StandardScaler()
    sscaler.fit(x)
    xss_sk = sscaler.transform(x)
    sscaler.fit(y)
    yss_sk = sscaler.transform(y)

    return xss_sk, yss_sk


def regression(x, y):
    model_lr_std = LinearRegression()
    model_lr_std.fit(x, y)

    return model_lr_std.coef_, model_lr_std.intercept_, model_lr_std.score(x, y)


def getCoef(x, y ,header):
    data = np.concatenate([x, y], 1)
    corr = np.corrcoef(data.T)
    sns.heatmap(corr, annot=True, cmap='coolwarm', xticklabels=header, yticklabels=header)
    plt.show()


def showData(x, y, header_x, header_y):
    # fig, ax = plt.subplots(x.shape[1], y.shape[1], figsize=(15, 15))
    for i in  range(y.shape[1]):
        for j in range(x.shape[1]):
            plt.scatter(x[:, j], y[:, i], marker='.')
            plt.xlabel(header_x[j])
            plt.ylabel(header_y[i])
            plt.show()

    #         ax[j, i].scatter(x[:, j], y[:, i], marker='.')
    #         ax[j, i].set_xlabel(header_x[j])
    #         ax[j, i].set_ylabel(header_y[i])
    #
    # plt.tight_layout()
    # plt.show()

def main():
    data, header = readCsv('/home/kuriatsu/Documents/others/kuribayashi_try_aligned.csv')
    data = np.array(data)
    x = np.array(data[:, 32:41], dtype=float)
    y = np.array(data[:, 10:12], dtype=float)

    showData(x, y,  header[32:41], header[10:12])
    xss, yss = dataScaling(x, y)

    coef, intercept, score = regression(xss, yss[:,0].reshape(-1, 1))
    print('coef: {}, intercept:{}, score:{}'.format(coef, intercept, score))
    coef, intercept, score = regression(xss, yss[:,0].reshape(-1, 1))
    print('coef: {}, intercept:{}, score:{}'.format(coef, intercept, score))
    getCoef(xss, yss, header[32:41]+header[10:12])

    coef, intercept, score = regression(x, y[:,1].reshape(-1, 1))
    print('coef: {}, intercept:{}, score:{}'.format(coef, intercept, score))
    coef, intercept, score = regression(x, y[:,1].reshape(-1, 1))
    print('coef: {}, intercept:{}, score:{}'.format(coef, intercept, score))
    getCoef(x, y, header[32:41]+header[10:12])


if __name__ == '__main__':
    main()
