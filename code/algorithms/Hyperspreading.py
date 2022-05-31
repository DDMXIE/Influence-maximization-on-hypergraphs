import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class Hyperspreading:
    def constructMatrix(self):
        """
        构造超图的点边矩阵
        :return: 超图的点边矩阵 matrix
        """
        matrix = np.random.randint(0, 2, size=(100, 100))
        # matrix = np.random.randint(0, 2, size=(100, 500))
        # for i in range(100):
        #     if sum(matrix[i]) == 0:
        #         j = np.random.randint(0, 10)
        #         matrix[i, j] = 1
        return matrix

    def getHpe(inode, matrix):
        """
        获取节点inode所在的超边集
        :param inode: 所选节点
        :param matrix: 超图的点边矩阵
        :return: 节点inode所涉及到的超边集合
        """
        return np.where(matrix[inode, :] == 1)[0]

    def chooseHpe(hpe_set):
        """
        从涉及超边中选择一条超边
        :param hpe_set: 涉及的超边集合
        :return: 所选超边
        """
        # print(hpe_set)
        if len(hpe_set) > 0:
            return random.sample(list(hpe_set), 1)[0]
        else:
            return []
        # return random.sample(list(hpe_set), 1)[0]
    def getNodesofHpe(hpe, matrix):
        """
        获取特定超边所包含的节点
        :param hpe: 特定超边
        :return: 特定超边所包含的节点
        """
        return np.where(matrix[:, hpe] == 1)[0]

    def getNodesofHpeSet(hpe_set, matrix):
        """
        获取超边集合中的所有节点
        :param hpe_set: 超边集合
        :param matrix: 超图的点边矩阵
        :return: 超边集合中的所有节点
        """
        adj_nodes = []
        for hpe in hpe_set:
            adj_nodes.extend(Hyperspreading.getNodesofHpe(hpe, matrix))
        return np.array(adj_nodes)

    def findAdjNode_RP(inode, df_hyper_matrix):
        """
        找到邻居节点集合
        :param I_list: 感染节点集
        :param df_hyper_matrix: 超图的点边矩阵
        :return: 不重复的邻居节点集 np.unique(nodes_in_edges)
        """
        # 找到该点所属的超边集合
        hpe_set = Hyperspreading.getHpe(inode, df_hyper_matrix.values)
        # 找到可能传播到超边中的顶点集合
        adj_nodes = Hyperspreading.getNodesofHpeSet(hpe_set, df_hyper_matrix.values)
        return np.array(adj_nodes)

    def findAdjNode_CP(inode, df_hyper_matrix):
        """
        找到邻居节点集合
        :param I_list: 感染节点集
        :param df_hyper_matrix: 超图的点边矩阵
        :return: 不重复的邻居节点集 np.unique(nodes_in_edges)
        """
        # 找到该点所属的超边集合
        edges_set = Hyperspreading.getHpe(inode, df_hyper_matrix.values)
        # 选择一条超边
        edge = Hyperspreading.chooseHpe(edges_set)
        # 找到可能传播到超边中的顶点集合
        adj_nodes = np.array(Hyperspreading.getNodesofHpe(edge, df_hyper_matrix.values))
        return adj_nodes

    def formatInfectedList(I_list, infected_list, infected_T):
        """
        筛选出不在I_list和infected_T当中的节点
        :param I_list: 感染节点集
        :param infected_list: 本次受感染的节点（未筛选）
        :return: 本次受感染的节点（筛选后）format_list
        """
        return (x for x in infected_list if x not in I_list and x not in infected_T)

    def getTrueStateNode(self, adj_nodes, I_list, R_list):
        """
        从所有可能感染节点中排查筛选只是S态的节点
        :param adj_nodes: 所有可能感染节点
        :param I_list: 截至上一时刻全部感染节点
        :param R_list: 截至上一时刻全部恢复节点
        :return:
        """
        adj_list = list(adj_nodes)
        for i in range(0, len(adj_nodes)):
            if adj_nodes[i] in I_list or adj_nodes[i] in R_list:
                adj_list.remove(adj_nodes[i])
        return np.array(adj_list)

    def spreadAdj(adj_nodes, I_list, infected_T, beta):
        """
        对可能感染的这部分节点进行传播
        :param adj_nodes: 邻居节点
        :param I_list: 感染节点集
        :param infected_T: T时刻下感染节点集
        :param beta: 传播概率
        :return: 本次最终被感染的节点集
        """
        random_list = np.random.random(size=len(adj_nodes))
        infected_list = adj_nodes[np.where(random_list < beta)[0]]
        infected_list_unique = Hyperspreading.formatInfectedList(I_list, infected_list, infected_T)
        return infected_list_unique

    def hyperSI(self, df_hyper_matrix, seeds):
        I_list = list(seeds)

        # 开始传播
        beta = 0.01
        iters = 25
        I_total_list = [1]

        for t in range(0, iters):
            infected_T = []
            for inode in I_list:
                # 找到邻居节点集
                adj_nodes = Hyperspreading.findAdjNode_CP(inode, df_hyper_matrix)
                # 开始对邻节点传播
                infected_list_unique = Hyperspreading.spreadAdj(adj_nodes, I_list, infected_T, beta)
                # 加入本次所感染的节点
                infected_T.extend(infected_list_unique)
            I_list.extend(infected_T)
            I_total_list.append(len(I_list))
        # plt.plot(np.arange(np.array(len(I_total_list))),I_total_list,color='orange')
        # plt.show()
        return I_total_list[-1:][0], I_list

