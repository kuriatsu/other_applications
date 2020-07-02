#! /usr/bin/python3
# -*- coding:utf-8 -*-

import csv
import numpy as np
from sklearn import preprocessing
import statsmodels.api as sm
# from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

def readCsv(filename):
    with open(filename, 'r') as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader)
        del header[5]
        del header[5]
        del header[5]
        out = []
        for row in reader:
            del row[5]
            del row[5]
            del row[5]
            if '' not in row:
                out.append(row)
            else:
                print(out)
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
    plt.title('Correlation between Each Variables', fontsize=20)
    sns.heatmap(corr, annot=True, cmap='coolwarm', xticklabels=header, yticklabels=header)
    plt.tight_layout()
    plt.show()


def showData(x, y, header_x, header_y):
    # fig, ax = plt.subplots(x.shape[1], y.shape[1], figsize=(15, 15))
    for i in  range(y.shape[1]):
        for j in range(x.shape[1]):
            h = sns.jointplot(x[:, j], y[:, i])
            h.set_axis_labels(header_x[j], header_y[i], fontsize=18)
            plt.tight_layout()
            plt.show()

            # plt.title('Relation between {} and {}'.format(header_y[i], header_x[j]), fontsize=15)
    #         ax[j, i].scatter(x[:, j], y[:, i], marker='.')
    #         ax[j, i].set_xlabel(header_x[j])
    #         ax[j, i].set_ylabel(header_y[i])
    #
    # plt.tight_layout()
    # plt.show()

def main():
    data, header = readCsv('/home/kuriatsu/share/PIE_result/june/result_logistic.csv')
    data = np.array(data)
    x = np.array(data[:, :5], dtype=float)
    y = np.array(data[:, 5:7], dtype=float)
    xss, yss = dataScaling(x, y)

    showData(x, y,  header[:5], header[5:7])

    regression(xss, yss[:,0].reshape(-1, 1))
    regression(xss, yss[:,1].reshape(-1, 1))
    getCoef(xss, yss, header[:5]+header[5:7])

    # coef, intercept, score = regression(x, y[:,1].reshape(-1, 1))
    # print('coef: {}, intercept:{}, score:{}'.format(coef, intercept, score))
    # coef, intercept, score = regression(x, y[:,1].reshape(-1, 1))
    # print('coef: {}, intercept:{}, score:{}'.format(coef, intercept, score))
    # getCoef(x, y, header[:8]+header[8:10])


if __name__ == '__main__':
    main()
