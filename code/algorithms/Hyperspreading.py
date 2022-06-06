import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class Hyperspreading:

    def getHpe(inode, matrix):

        return np.where(matrix[inode, :] == 1)[0]

    def chooseHpe(hpe_set):

        if len(hpe_set) > 0:
            return random.sample(list(hpe_set), 1)[0]
        else:
            return []

    def getNodesofHpe(hpe, matrix):

        return np.where(matrix[:, hpe] == 1)[0]

    def getNodesofHpeSet(hpe_set, matrix):

        adj_nodes = []
        for hpe in hpe_set:
            adj_nodes.extend(Hyperspreading.getNodesofHpe(hpe, matrix))
        return np.array(adj_nodes)

    def findAdjNode_CP(inode, df_hyper_matrix):

        edges_set = Hyperspreading.getHpe(inode, df_hyper_matrix.values)

        edge = Hyperspreading.chooseHpe(edges_set)

        adj_nodes = np.array(Hyperspreading.getNodesofHpe(edge, df_hyper_matrix.values))
        return adj_nodes

    def formatInfectedList(I_list, infected_list, infected_T):

        return (x for x in infected_list if x not in I_list and x not in infected_T)

    def getTrueStateNode(self, adj_nodes, I_list, R_list):

        adj_list = list(adj_nodes)
        for i in range(0, len(adj_nodes)):
            if adj_nodes[i] in I_list or adj_nodes[i] in R_list:
                adj_list.remove(adj_nodes[i])
        return np.array(adj_list)

    def spreadAdj(adj_nodes, I_list, infected_T, beta):

        random_list = np.random.random(size=len(adj_nodes))
        infected_list = adj_nodes[np.where(random_list < beta)[0]]
        infected_list_unique = Hyperspreading.formatInfectedList(I_list, infected_list, infected_T)
        return infected_list_unique

    def hyperSI(self, df_hyper_matrix, seeds):

        I_list = list(seeds)
        beta = 0.01
        iters = 25
        I_total_list = [1]

        for t in range(0, iters):
            infected_T = []
            for inode in I_list:
                adj_nodes = Hyperspreading.findAdjNode_CP(inode, df_hyper_matrix)
                infected_list_unique = Hyperspreading.spreadAdj(adj_nodes, I_list, infected_T, beta)
                infected_T.extend(infected_list_unique)
            I_list.extend(infected_T)
            I_total_list.append(len(I_list))
        # plt.plot(np.arange(np.array(len(I_total_list))),I_total_list,color='orange')
        # plt.show()
        return I_total_list[-1:][0], I_list

