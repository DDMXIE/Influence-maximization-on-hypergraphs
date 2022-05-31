# 超网络度两种定义方式的相关性

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transform import Transform
from scipy.stats import pearsonr
from decimal import Decimal
from pylab import *


def getTotalAdj(df_hyper_matrix, N):
    adj_matrix = np.dot(df_hyper_matrix, df_hyper_matrix.T)
    adj_matrix[np.eye(N, dtype=np.bool_)] = 0
    df_adj_matrix = pd.DataFrame(adj_matrix)
    return df_adj_matrix.sum(axis=1)


def getAdjs(df_hyper_matrix, N):
    deg_list = []
    nodes_arr = np.arange(N)
    for node in nodes_arr:
        node_list = []
        edge_set = np.where(df_hyper_matrix.loc[node] == 1)[0]
        for edge in edge_set:
            node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
        node_set = np.unique(np.array(node_list))
        deg_list.append(len(list(node_set)) - 1)
    return np.array(deg_list)


def getDegDistribution(deg_list):
    print(deg_list)
    deg_max = max(deg_list)
    deg_dict = {}
    for i in range(0, int(deg_max) + 1):
        deg_dict[i] = 0
    print(deg_dict)
    for deg in range(0, int(deg_max) + 1):
        total = len(list(np.where(np.array(deg_list) == deg)[0]))
        deg_dict[deg] = deg_dict[deg] + total
    return deg_dict


def main(fileName_list):
    for index in range(len(fileName_list)):
        plt.figure(figsize=(32, 22))
        fileName = fileName_list[index]
        tf = Transform()
        df_hyper_matrix, N = tf.changeEdgeToMatrix('../datasets/' + fileName + '.txt')

        deg_adj = getAdjs(df_hyper_matrix, N)
        deg_hyper = df_hyper_matrix.sum(axis=1)
        pccs = pearsonr(deg_adj, deg_hyper)
        pc = str(Decimal(pccs[0]).quantize(Decimal("0.00")))

        border_size = 3
        ax = plt.subplot(1, 1, 1)
        ax.spines['bottom'].set_linewidth(border_size)
        ax.spines['left'].set_linewidth(border_size)
        ax.spines['right'].set_linewidth(border_size)
        ax.spines['top'].set_linewidth(border_size)

        plt.tick_params(labelsize=150, pad=30)
        plt.scatter(deg_adj, deg_hyper, color='#FFA8A8', edgecolors='#981047', label='PC=' + pc, s=2000, linewidths=5)
        plt.tick_params(width=10, direction='out', length=30)
        plt.subplots_adjust(left=0.18, bottom=0.15)

        plt.savefig('./degree_correlation/' + fileName + '.png')
        plt.show()


if __name__ == '__main__':
    fileName_list = ['Restaurants-Rev', 'Music-Rev', 'Geometry',
                'Algebra', 'Bars-Rev', 'NDC-classes-unique-hyperedges',
                'iAF1260b', 'iJO1366']
    main(fileName_list)

