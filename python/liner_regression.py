#! /usr/bin/python3
# -*- coding:utf-8 -*-

import csv
import numpy as np
from sklearn import preprocessing
import statsmodels.api as sm
# from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

def readCsv(filename, delete_cols=[]):
    with open(filename, 'r') as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader)
        header = [v for i, v in enumerate(header) if i not in delete_cols]
        print(header)
        out = []

        for row in reader:
            row = [v for i, v in enumerate(row) if i not in delete_cols]
            if '' not in row:
                out.append(row)
            # else:
            #     print("remove row")

        return out, header


def dataScaling(x, y):
    sscaler = preprocessing.StandardScaler()
    sscaler.fit(x)
    xss_sk = sscaler.transform(x)
    sscaler.fit(y)
    yss_sk = sscaler.transform(y)

    return xss_sk, yss_sk


def regression(x, y):
    model = sm.OLS(y, x)
    result = model.fit()
    print(result.summary())

    # model_lr_std = LinearRegression()
    # model_lr_std.fit(x, y)
    # return model_lr_std.coef_, model_lr_std.intercept_, model_lr_std.score(x, y)



def getCoef(x, y ,header):

    data = np.concatenate([x, y], 1)
    corr = np.corrcoef(data.T)
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(corr, annot=True, cmap='coolwarm', xticklabels=header, yticklabels=header, ax=ax)
    ax.set_title(label='Correlation between Each Variables', size=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    fig.set_size_inches(10, 9, forward=True)
    plt.subplots_adjust(bottom=0.15, top=0.93, left=0.14, right=99.0)
    plt.show()


def showData(x, y, header_x, header_y):
    # fig, ax = plt.subplots(x.shape[1], y.shape[1], figsize=(15, 15))
    for i in  range(y.shape[1]):
        for j in range(x.shape[1]):
            sns.jointplot(x[:, j], y[:, i], ax=ax)
            ax.set_axis_labels(header_x[j], header_y[i], fontsize=24)
            # plt.tight_layout()
            plt.show()

            # plt.title('Relation between {} and {}'.format(header_y[i], header_x[j]), fontsize=15)
    #         ax[j, i].scatter(x[:, j], y[:, i], marker='.')
    #         ax[j, i].set_xlabel(header_x[j])
    #         ax[j, i].set_ylabel(header_y[i])
    #
    # plt.tight_layout()
    # plt.show()

def main():
    data, header = readCsv('/home/kuriatsu/DropboxKuri/data/PIE_experiment/result_logistic.csv', delete_cols=[7, 8, 9])
    # data, header = readCsv('/home/kuriatsu/share/PIE_result/june/result_logistic.csv')
    data_time = np.array(data)
    # data_time = np.delete(data_time, 3, 1)
    x_time = np.array(data_time[:, :7], dtype=float)
    y_time = np.array(data_time[:, 7], dtype=float)
    xss_time, yss_time = dataScaling(x_time, y_time.reshape(-1, 1))
    regression(xss_time, yss_time)
    getCoef(xss_time, yss_time, header[:8])

    # showData(x_time, y_time,  header[:5], header[5:7])
    data, header = readCsv('/home/kuriatsu/DropboxKuri/data/PIE_experiment/result_logistic.csv', delete_cols=[7, 8, 10])
    # data, header = readCsv('/home/kuriatsu/share/PIE_result/june/result_logistic.csv', delete_cols=[7, 8, 9])
    data_acc = np.array(data)
    # data_acc = np.delete(data_time, 3, 1)
    x_acc = np.array(data_acc[:, :7], dtype=float)
    y_acc = np.array(data_acc[:, 7], dtype=float)
    xss_acc, yss_acc = dataScaling(x_acc, y_acc.reshape(-1, 1))
    regression(xss_acc, yss_acc)
    getCoef(xss_acc, yss_acc, header[:8])


    # coef, intercept, score = regression(x, y[:,1].reshape(-1, 1))
    # print('coef: {}, intercept:{}, score:{}'.format(coef, intercept, score))
    # coef, intercept, score = regression(x, y[:,1].reshape(-1, 1))
    # print('coef: {}, intercept:{}, score:{}'.format(coef, intercept, score))
    # getCoef(x, y, header[:8]+header[8:10])


if __name__ == '__main__':
    main()
