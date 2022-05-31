import pandas as pd
import numpy as np
import networkx as nx

def loadData(path):
    """
    超网络数据处理
    :param path: 读取文件路径
    :return: node_dict, node_arr, node_list, arr
    """
    node_dict = {}
    df = pd.read_csv(path, index_col=False, header=None)
    arr = df.values
    node_list = []
    for each in arr:
        node_list.extend(list(map(int, each[0].split(" "))))
    node_arr = np.unique(np.array(node_list))
    for i in range(0, len(node_arr)):
        node_dict[node_arr[i]] = i
    return node_dict, node_arr, node_list, arr

class HyperG:

    # HyperG 属性定义
    N = 0
    M = 0
    node_dict = {}
    node_arr = []
    arr = []

    # HyperG 方法定义
    def setN(self, N):
        """
        setter方法 ：N
        :param N: 超点个数
        """
        self.N = N

    def setM(self, M):
        """
        setter方法 ：M
        :param M: 超边个数
        """
        self.M = M

    def getN(self):
        """
        getter方法： N
        :return: N 超点个数
        """
        return self.N

    def getM(self):
        """
        getter方法： M
        :return: M 超边个数
        """
        return self.M

    def initData(self, node_dict, node_arr, node_list, arr):
        """
        超图初始化 init方法
        :param node_dict:
        :param node_arr: 节点列表（去重）
        :param node_list: 全部节点列表（不去重）
        :param arr: 超边列表
        """
        self.node_dict = node_dict
        self.node_arr = node_arr
        self.arr = arr

    def cptSize(self):
        """
        计算超网络大小方法
        :return: （N: 超点个数,  M：超边条数）
        """
        node_dict = self.node_dict
        node_arr = self.node_arr
        arr = self.arr
        # print(node_dict, len(list(node_arr)), len(arr))
        self.N = len(list(node_arr))
        self.M = len(arr)
        return (self.N, self.M)

    def get_edge_dict(self, path):
        """
        每条超边包含着哪些节点
        """
        node_dict, node_arr, node_list, arr = loadData(path)
        hpe_dict = {}
        df = pd.read_csv(path, index_col=False, header=None)
        arr = df.values
        i = 0
        for each in arr:
            new_list = []
            nodes_index_list = list(map(int, each[0].split(" ")))
            for index in nodes_index_list:
                new_list.append(node_dict[index])
            hpe_dict[i] = new_list
            i = i + 1
        return hpe_dict

    def get_nodes_dict(self, path):
        """
        每个节点隶属于哪些超边
        """
        nodes_dict = {}
        node_dict, node_arr, node_list, arr = loadData(path)
        total = len(node_dict.values())
        for i in range(0, total):
            nodes_dict[i] = []
        df = pd.read_csv(path, index_col=False, header=None)
        arr = df.values
        i = 0
        for each in arr:
            nodes_index_list = list(map(int, each[0].split(" ")))
            for index in nodes_index_list:
                nodes_dict[node_dict[index]].append(i)
            i = i + 1
        return nodes_dict

    def get_hyper_degree(self, path, type):
        """
        超图中的超度（隶属边数）
        type：node/edge 节点的度/超边的度
        """
        if type == 'node':
            dict = HyperG.get_nodes_dict(self, path)
        elif type == 'edge':
            dict = HyperG.get_edge_dict(self, path)
        for each in dict:
            dict[each] = len(dict[each])
        return dict

    def get_average_degree(self, path, type):
        """
        超图中的平均度
        :param type: node/edge 节点的度/超边的度
        """
        deg_dict = HG.get_hyper_degree(path, type)
        deg_list = deg_dict.values()
        avg_deg = sum(deg_list) / len(deg_list)
        return avg_deg

    def get_adjacent_node(self, path):
        """
        每个节点的邻居（相邻）节点
        """
        dict_node = HyperG.get_nodes_dict(self, path)
        dict_edge = HyperG.get_edge_dict(self, path)

        adj_dict = {}
        for j in range(0, len(dict_node)):
            adj_dict[j] = []
        for i in dict_node:
            edge_list = dict_node[i]
            for edge in edge_list:
                adj_dict[i].extend(dict_edge[edge])
        for k in range(0, len(adj_dict)):
            adj_dict[k] = list(np.unique(np.array(adj_dict[k])))
            # 去掉自环，重边
            if k in adj_dict[k]:
                adj_dict[k].remove(k)
        return adj_dict

    def get_projected_network(self, path):
        """
        将超图映射成投影网络
        """
        adj_dict = HyperG.get_adjacent_node(self, path)
        G = nx.Graph()
        G.add_nodes_from(list(adj_dict.keys()))
        for from_node in adj_dict:
            node_list = adj_dict[from_node]
            for to_node in node_list:
                G.add_edge(from_node, to_node)
        return G

    def get_clustering_coefficient(self, path):
        """
        超图对应投影网络的聚类系数
        """
        G = HyperG.get_projected_network(self, path)
        return nx.average_clustering(G)

    def get_average_neighbor_degree(self, path):
        """
        超图对应投影网络的每个节点的邻居节点平均度
        """
        G = HyperG.get_projected_network(self, path)
        return nx.average_neighbor_degree(G)

    def get_density(self, path):
        """
        超图对应投影网络的网络密度
        """
        G = HyperG.get_projected_network(self, path)
        return nx.density(G)

    def get_average_shortest_path_length(self, path):
        """
        超图对应投影网络的最短路径长度
        """
        path_lengths = []
        G = HyperG.get_projected_network(self, path)
        node_list = G.nodes
        for node in range(0, len(node_list)):
            path_value_list = list(nx.shortest_path_length(G, target=node).values())
            path_lengths.extend(path_value_list)
        return sum(path_lengths) / len(path_lengths)

    def get_diameter(self, path):
        """
        超图对应投影网络的网络直径
        """
        path_lengths = []
        G = HyperG.get_projected_network(self, path)
        node_list = G.nodes
        for node in range(0, len(node_list)):
            path_value_list = list(nx.shortest_path_length(G, target=node).values())
            path_lengths.extend(path_value_list)
        return max(path_lengths)

    def get_average_adj_degree(self, path):
        adj_dict = HyperG.get_adjacent_node(self, path)
        sum = 0
        for i in adj_dict:
            sum = sum + len(adj_dict[i])
        return sum / len(adj_dict.keys())


if __name__ == '__main__':

    print('-------------------------------------')
    print('------------- 超图拓扑结构 -------------')
    print('-------------------------------------')

    # 初始化
    HG = HyperG()
    path = '../datasets/Restaurants-Rev.txt'
    # path = '../datasets/Algebra.txt'
    # path = '../datasets/Geometry.txt'
    # path = '../datasets/Music-Rev.txt'
    # path = '../datasets/Bars-Rev.txt'
    # path = '../datasets/NDC-classes-unique-hyperedges.txt'
    # path = '../datasets/NDC-substances-unique-hyperedges.txt'
    # path = '../datasets/DAWN-unique-hyperedges.txt'
    # path = '../datasets/iAF1260b.txt'
    # path = '../datasets/iJO1366.txt'
    node_dict, node_arr, node_list, arr = loadData(path)
    HG.initData(node_dict, node_arr, node_list, arr)

    # # 计算超图大小
    size = HG.cptSize()
    print("超图节点数：", size[0])
    print("超图超边数：", size[1])

    nodes_belong_edges = HG.get_nodes_dict(path)
    print("每个超点隶属的超边：", nodes_belong_edges)

    edges_include_nodes = HG.get_edge_dict(path)
    print("每个超边包含的节点：", edges_include_nodes)

    nodes_degree = HG.get_hyper_degree(path, 'node')
    print("每个节点的度：", nodes_degree)

    edges_degree = HG.get_hyper_degree(path, 'edge')
    print("每条超边的度：", edges_degree)

    avg_node_deg = HG.get_average_adj_degree(path)
    print("节点平均度：", avg_node_deg)

    avg_node_degree = HG.get_average_degree(path, 'node')
    print("超点平均超度：", avg_node_degree)

    avg_edge_degree = HG.get_average_degree(path, 'edge')
    print("超边平均度：", avg_edge_degree)

    clustering_coefficient = HG.get_clustering_coefficient(path)
    print("聚类系数：", clustering_coefficient)

    density = HG.get_density(path)
    print("网络密度：", density)

    average_shortest_path_length = HG.get_average_shortest_path_length(path)
    print("平均最短路径长度：", average_shortest_path_length)

    diameter = HG.get_diameter(path)
    print("网络直径：", diameter)

    average_neighbor_degree = HG.get_average_neighbor_degree(path)
    print("邻居的平均度：", average_neighbor_degree)

    # adjacent_nodes = HG.get_adjacent_node                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                (path)
    # print("节点的邻居节点：", adjacent_nodes)







