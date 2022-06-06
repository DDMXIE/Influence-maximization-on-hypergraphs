import pandas as pd
import numpy as np
import networkx as nx

def loadData(path):

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

    N = 0
    M = 0
    node_dict = {}
    node_arr = []
    arr = []

    def setN(self, N):

        self.N = N

    def setM(self, M):

        self.M = M

    def getN(self):

        return self.N

    def getM(self):

        return self.M

    def initData(self, node_dict, node_arr, node_list, arr):

        self.node_dict = node_dict
        self.node_arr = node_arr
        self.arr = arr

    def cptSize(self):

        node_dict = self.node_dict
        node_arr = self.node_arr
        arr = self.arr
        self.N = len(list(node_arr))
        self.M = len(arr)
        return (self.N, self.M)

    def get_edge_dict(self, path):

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

        if type == 'node':
            dict = HyperG.get_nodes_dict(self, path)
        elif type == 'edge':
            dict = HyperG.get_edge_dict(self, path)
        for each in dict:
            dict[each] = len(dict[each])
        return dict

    def get_average_degree(self, path, type):

        deg_dict = HG.get_hyper_degree(path, type)
        deg_list = deg_dict.values()
        avg_deg = sum(deg_list) / len(deg_list)
        return avg_deg

    def get_adjacent_node(self, path):

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

        adj_dict = HyperG.get_adjacent_node(self, path)
        G = nx.Graph()
        G.add_nodes_from(list(adj_dict.keys()))
        for from_node in adj_dict:
            node_list = adj_dict[from_node]
            for to_node in node_list:
                G.add_edge(from_node, to_node)
        return G

    def get_clustering_coefficient(self, path):

        G = HyperG.get_projected_network(self, path)
        return nx.average_clustering(G)

    def get_average_neighbor_degree(self, path):

        G = HyperG.get_projected_network(self, path)
        return nx.average_neighbor_degree(G)

    def get_density(self, path):

        G = HyperG.get_projected_network(self, path)
        return nx.density(G)

    def get_average_shortest_path_length(self, path):

        path_lengths = []
        G = HyperG.get_projected_network(self, path)
        node_list = G.nodes
        for node in range(0, len(node_list)):
            path_value_list = list(nx.shortest_path_length(G, target=node).values())
            path_lengths.extend(path_value_list)
        return sum(path_lengths) / len(path_lengths)

    def get_diameter(self, path):

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

    print('----------------------------------------------------')
    print('------------- Topology of a hypergraph -------------')
    print('----------------------------------------------------')

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

    size = HG.cptSize()
    print("n：", size[0])
    print("m：", size[1])

    nodes_belong_edges = HG.get_nodes_dict(path)
    print("nodes_belong_edges", nodes_belong_edges)

    edges_include_nodes = HG.get_edge_dict(path)
    print("edges_include_nodes", edges_include_nodes)

    nodes_degree = HG.get_hyper_degree(path, 'node')
    print("nodes_degree", nodes_degree)

    edges_degree = HG.get_hyper_degree(path, 'edge')
    print("edges_degree", edges_degree)

    avg_node_deg = HG.get_average_adj_degree(path)
    print("average_node_degree", avg_node_deg)

    avg_node_degree = HG.get_average_degree(path, 'node')
    print("average_node_hyperdegree：", avg_node_degree)

    avg_edge_degree = HG.get_average_degree(path, 'edge')
    print("average_hyperedge_size：", avg_edge_degree)

    clustering_coefficient = HG.get_clustering_coefficient(path)
    print("clustering_coefficient", clustering_coefficient)

    density = HG.get_density(path)
    print("density", density)

    average_shortest_path_length = HG.get_average_shortest_path_length(path)
    print("average_shortest_path_length", average_shortest_path_length)

    diameter = HG.get_diameter(path)
    print("diameter", diameter)

    average_neighbor_degree = HG.get_average_neighbor_degree(path)
    print("average_neighbor_degree", average_neighbor_degree)








