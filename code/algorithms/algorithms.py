import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Hyperspreading import Hyperspreading
from tqdm import tqdm
import copy
import random
from transform import Transform
import networkx as nx
import matplotlib
matplotlib.use('Agg')
plt.switch_backend('agg')


def getSeeds_sta(degree, i):

    matrix = []
    matrix.append(np.arange(len(degree)))
    matrix.append(degree)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.index = ['node_index', 'node_degree']
    df_sort_matrix = df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)
    degree_list = list(df_sort_matrix.loc['node_degree'])
    nodes_list = list(df_sort_matrix.loc['node_index'])
    chosed_arr = list(df_sort_matrix.loc['node_index'][:i])
    index = np.where(np.array(degree_list) == degree_list[i])[0]
    nodes_set = list(np.array(nodes_list)[index])
    while 1:
        node = random.sample(nodes_set, 1)[0]
        if node not in chosed_arr:
            chosed_arr.append(node)
            break
        else:
            nodes_set.remove(node)
            continue
    return chosed_arr


def degreemax(df_hyper_matrix, K, R):
    """
    Degree algorithm
    """
    degree = getTotalAdj(df_hyper_matrix, N)
    inf_spread_matrix = []
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(0, K):
            seeds = getSeeds_sta(degree, i)
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def HDegree(df_hyper_matrix, K, R):
    """
    HDegree algorithm
    """
    degree = df_hyper_matrix.sum(axis=1)
    inf_spread_matrix = []
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(0, K):
            seeds = getSeeds_sta(degree, i)
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def getDegreeList(degree):

    matrix = []
    matrix.append(np.arange(len(degree)))
    matrix.append(degree)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.index = ['node_index', 'node_degree']
    return df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)


def getMaxDegreeNode(degree, seeds):

    degree_copy = copy.deepcopy(degree)
    global chosedNode
    while 1:
        flag = 0
        degree_matrix = getDegreeList(degree_copy)
        node_index = degree_matrix.loc['node_index']
        for node in node_index:
            if node not in seeds:
                chosedNode = node
                flag = 1
                break
        if flag == 1:
            break
    return [chosedNode]


def updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds):

    edge_set = np.where(df_hyper_matrix.loc[chosenNode] == 1)[0]
    adj_set = []
    for edge in edge_set:
        adj_set.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
    adj_set_unique = np.unique(np.array(adj_set))
    for adj in adj_set_unique:
        adj_edge_set = np.where(df_hyper_matrix.loc[adj] == 1)[0]
        adj_adj_set = []
        for each in adj_edge_set:
            adj_adj_set.extend(list(np.where(df_hyper_matrix[each] == 1)[0]))
        if adj in adj_adj_set:
            adj_adj_set.remove(adj)
        sum = 0
        for adj_adj in adj_adj_set:
            if adj_adj in seeds:
                sum = sum + 1
        degree[adj] = degree[adj] - sum


def updateDeg_hsd(degree, chosenNode, df_hyper_matrix):

    edge_set = np.where(df_hyper_matrix.loc[chosenNode] == 1)[0]

    for edge in edge_set:
        node_set = np.where(df_hyper_matrix[edge] == 1)[0]
        for node in node_set:
            degree[node] = degree[node] - 1


def getDegreeWeighted(df_hyper_matrix, N):

    adj_matrix = np.dot(df_hyper_matrix, df_hyper_matrix.T)
    adj_matrix[np.eye(N, dtype=np.bool_)] = 0
    df_adj_matrix = pd.DataFrame(adj_matrix)
    return df_adj_matrix.sum(axis=1)


def getTotalAdj(df_hyper_matrix, N):
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


def getSeeds_hdd(N, K):

    seeds = []
    degree = getTotalAdj(df_hyper_matrix, N)
    for j in range(1, K+1):
        chosenNode = getMaxDegreeNode(degree, seeds)[0]
        seeds.append(chosenNode)
        updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds)
    return seeds


def getSeeds_hsd(N, K):

    seeds = []
    degree = getTotalAdj(df_hyper_matrix, N)
    for j in range(1, K+1):
        chosenNode = getMaxDegreeNode(degree, seeds)[0]
        seeds.append(chosenNode)
        updateDeg_hsd(degree, chosenNode, df_hyper_matrix)
    return seeds


def hurDisc(df_hyper_matrix, K, R, N):
    """
    HuresticDegreeDiscount algorithm
    """
    inf_spread_matrix = []
    seeds_list = getSeeds_hdd(N, K)
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, K+1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def sglDisc(df_hyper_matrix, K, R, N):
    """
    HuresticSingleDiscount algorithm
    """
    inf_spread_matrix = []
    seeds_list = getSeeds_hsd(N, K)
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, K+1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list



def generalGreedy(df_hyper_matrix, K, R):
    """
    GeneralGreedy algorithm
    """
    degree = df_hyper_matrix.sum(axis=1)
    inf_spread_matrix = []
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        seeds = []
        for i in range(0, K):
            scale_list_temp = []
            maxNode = 0
            maxScale = 1
            for inode in range(0, len(degree)):
                if inode not in seeds:
                    seeds.append(inode)
                    scale, I_list = hs.hyperSI(df_hyper_matrix, seeds)
                    seeds.remove(inode)
                    scale_list_temp.append(scale)
                    if scale > maxScale:
                        maxNode = inode
                        maxScale = scale
            seeds.append(maxNode)
            scale_list.append(max(scale_list_temp))
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def computeCI(l, N, df_hyper_matrix):

    CI_list = []
    degree = df_hyper_matrix.sum(axis=1)
    M = len(df_hyper_matrix.columns.values)
    for i in range(0, N):
        # 找到它的l阶邻居
        edge_set = np.where(df_hyper_matrix.loc[i] == 1)[0]
        if l == 1:
            node_list = []
            for edge in edge_set:
                node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
            if i in node_list:
                node_list.remove(i)
            node_set = np.unique(np.array(node_list))
        elif l == 2:
            node_list = []
            for edge in edge_set:
                node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
            if i in node_list:
                node_list.remove(i)
            node_set1 = np.unique(np.array(node_list))
            node_list2 = []
            edge_matrix = np.dot(df_hyper_matrix.T, df_hyper_matrix)
            edge_matrix[np.eye(M, dtype=np.bool_)] = 0
            df_edge_matrix = pd.DataFrame(edge_matrix)
            adj_edge_list = []
            for edge in edge_set:
                adj_edge_list.extend(list(np.where(df_edge_matrix[edge] != 0)[0]))
            adj_edge_set = np.unique(np.array(adj_edge_list))
            for each in adj_edge_set:
                node_list2.extend(list(np.where(df_hyper_matrix[each] == 1)[0]))
            node_set2 = list(np.unique(np.array(node_list2)))
            for node in node_set2:
                if node in list(node_set1):
                    # print(node_set2)
                    node_set2.remove(node)
            node_set = np.array(node_set2)
        ki = degree[i]
        sum = 0
        for u in node_set:
            sum = sum + (degree[u] - 1)
        CI_i = (ki - 1) * sum
        CI_list.append(CI_i)
    return CI_list


def getSeeds_ci(l, N, K, df_hyper_matrix):
    seeds = []
    n = np.ones(N)
    CI_list = computeCI(l, N, df_hyper_matrix)
    CI_arr = np.array(CI_list)
    for j in range(0, K):
        CI_chosed_val = CI_arr[np.where(n == 1)[0]]
        CI_chosed_index = np.where(n == 1)[0]
        index = np.where(CI_chosed_val == np.max(CI_chosed_val))[0][0]
        node = CI_chosed_index[index]
        n[node] = 0
        seeds.append(node)
    return seeds


def CIAgr(df_hyper_matrix, K, R, N, l):
    """
    H-CI algorithm
    """
    inf_spread_matrix = []
    seeds_list = getSeeds_ci(l, N, K, df_hyper_matrix)
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, K + 1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def getSeeds_ris(N, K, lamda, theta, df_hyper_matrix):
    S = []
    U = []
    # 迭代θ次
    for theta_iter in range(0, theta):
        df_matrix = copy.deepcopy(df_hyper_matrix)
        # 随机选择节点
        selected_node = random.sample(list(np.arange(len(df_hyper_matrix.index.values))),1)[0]
        # 以1-λ的比例删边，构成子超图
        all_edges = np.arange(len(df_hyper_matrix.columns.values))
        prob = np.random.random(len(all_edges))
        index = np.where(prob > lamda)[0]
        for edge in index:
            df_matrix[edge] = 0
        # 将子超图映射到普通图
        adj_matrix = np.dot(df_matrix, df_matrix.T)
        adj_matrix[np.eye(N, dtype=np.bool_)] = 0
        df_adj_matrix = pd.DataFrame(adj_matrix)
        df_adj_matrix[df_adj_matrix > 0] = 1
        G = nx.from_numpy_matrix(df_adj_matrix.values)
        shortest_path = nx.shortest_path(G, target=selected_node)
        RR = []
        for each in shortest_path:
            RR.append(each)
        U.append(list(np.unique(np.array(RR))))
    # 重复k次
    for k in range(0, K):
        U_list = []
        for each in U:
            U_list.extend(each)
        dict = {}
        for each in U_list:
            if each in dict.keys():
                dict[each] = dict[each] + 1
            else:
                dict[each] = 1
        candidate_list = sorted(dict.items(), key=lambda item: item[1], reverse=True)
        chosed_node = candidate_list[0][0]
        S.append(chosed_node)
        for each in U:
            if chosed_node in each:
                U.remove(each)
    return S


def RISAgr(df_hyper_matrix, K, R, N, lamda, theta):
    """
    H-RIS algorithm
    """
    inf_spread_matrix = []
    seeds_list = getSeeds_ris(N, K, lamda, theta, df_hyper_matrix)
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, K + 1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


if __name__ == '__main__':

    hs = Hyperspreading()
    tf = Transform()
    fileName = 'Algebra'
    # fileName = 'Restaurants-Rev'
    # fileName = 'Music-Rev'
    # fileName = 'Geometry'
    # fileName = 'Bars-Rev'
    # fileName = 'NDC-classes-unique-hyperedges'
    # fileName = 'iAF1260b'
    # fileName = 'iJO1366'

    df_hyper_matrix, N = tf.changeEdgeToMatrix('../datasets/' + fileName + '.txt')


    K = 25
    R = 500
    sta_scale_list = HDegree(df_hyper_matrix, K, R)
    ggd_scale_list = generalGreedy(df_hyper_matrix, K, 50)
    dmax_scale_list = degreemax(df_hyper_matrix, K, R)
    hurd_scale_list = hurDisc(df_hyper_matrix, K, R, N)
    sgd_scale_list = sglDisc(df_hyper_matrix, K, R, N)
    ci_scale_list = CIAgr(df_hyper_matrix, K, R, N, 1)
    ci_scale_list2 = CIAgr(df_hyper_matrix, K, R, N, 2)
    ris_scale_list = RISAgr(df_hyper_matrix, K, R, N, 0.01, 200)


    final_matrix = []
    final_matrix.append(sta_scale_list)
    final_matrix.append(ris_scale_list)
    final_matrix.append(dmax_scale_list)
    final_matrix.append(ci_scale_list)
    final_matrix.append(ci_scale_list2)
    final_matrix.append(sgd_scale_list)
    final_matrix.append(hurd_scale_list)
    final_matrix.append(ggd_scale_list)

    final_df = pd.DataFrame(final_matrix).T
    final_df.columns = [['H-Degree', 'RIS', 'Degree', 'CI (l=1)', 'CI (l=2)',
                         'HeuristicSingleDiscount', 'HeuristicDegreeDiscount', 'greedy']]
    print(final_df)
    final_df.to_csv('./csv/' + fileName + '.csv')
