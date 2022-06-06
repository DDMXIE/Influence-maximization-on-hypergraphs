import numpy as np
import pandas as pd
import random
import powerlaw
from decimal import Decimal
import matplotlib.pyplot as plt


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
    deg_max = max(deg_list)
    deg_dict = {}
    for i in range(0, int(deg_max) + 1):
        deg_dict[i] = 0
    for deg in range(0, int(deg_max) + 1):
        total = len(list(np.where(np.array(deg_list) == deg)[0]))
        deg_dict[deg] = deg_dict[deg] + total
    return deg_dict


def select_node(temp, arr):
    rdm = random.random()
    range_list = []
    for i in range(0, len(arr) + 1):
        range_list.append(sum(arr[:i]))
    for from_index in range(0, len(range_list)):
        to_index = from_index + 1
        if rdm >= range_list[from_index] and rdm < range_list[to_index]:
            if from_index in temp:
                continue
            else:
                temp.append(from_index)
                return from_index
        else:
            continue


def hyperCL(*, hpe_size, hnd_deg):
    edge_aff_dict = {}
    for i in range(0, len(hpe_size)):
        edge_aff_dict[i] = 0
    for edge_index in range(0, len(hpe_size)):
        temp = []
        arr = np.array(hnd_deg) / sum(hnd_deg)
        while len(temp) < hpe_size[edge_index]:
            select_node(temp, arr)
        edge_aff_dict[edge_index] = temp
        print(temp)
    return edge_aff_dict


def get_incidence_matrix(*, exp, nsize, esize, hpe_maxdeg):
    hnd_deg = powerlaw.Power_Law(xmin=1, parameters=[exp]).generate_random(nsize)
    hpe_size = np.random.uniform(0, hpe_maxdeg, esize)
    edge_aff_dict = hyperCL(hpe_size=hpe_size, hnd_deg=hnd_deg)
    matrix = np.zeros((len(hnd_deg), len(hpe_size)))
    for edge in edge_aff_dict:
        for node in edge_aff_dict[edge]:
            matrix[node][edge] = 1
    df_hyper_matrix = pd.DataFrame(matrix)
    return df_hyper_matrix


def get_cv(df_hyper_matrix):
    deg_adj = getAdjs(df_hyper_matrix, len(df_hyper_matrix.index.values))
    deg_hyper = df_hyper_matrix.sum(axis=1)
    deg_adj_avg = np.sum(deg_adj) / len(deg_adj)
    deg_adj_std = np.std(deg_adj)
    deg_hyper_avg = np.sum(deg_hyper) / len(deg_hyper)
    deg_hyper_std = np.std(deg_hyper)
    return deg_adj_std / deg_adj_avg, deg_hyper_std / deg_hyper_avg


def draw(df_hyper_matrix, log_type):
    cv = get_cv(df_hyper_matrix)[1]
    deg_dict = getDegDistribution(list(df_hyper_matrix.sum(axis=1)))
    plt.title('HyperCL Network')
    plt.scatter(deg_dict.keys(), deg_dict.values(), color='#23B8BF', edgecolors='#2C5E5D'
                , s=60, label='COV=' + str(Decimal(cv).quantize(Decimal("0.00"))))
    if log_type == 'single':
        plt.yscale('log')
    elif log_type == 'double':
        plt.xscale('log')
        plt.yscale('log')
    plt.legend()
    plt.show()


def save(df_hyper_matrix, log_type, fileName):

    np.save('./matrix/' + fileName + '.npy', np.array(df_hyper_matrix))
    deg_dict = getDegDistribution(list(df_hyper_matrix.sum(axis=1)))
    matrix = []
    matrix.append(list(deg_dict.values()))
    matrix_df = pd.DataFrame(matrix)
    matrix_df.columns = list(deg_dict.keys())
    matrix_df.to_csv('./nw/' + fileName)




