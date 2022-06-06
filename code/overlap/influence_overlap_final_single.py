# matplotlib.use('Agg')
# plt.switch_backend('agg')
from collections import Counter
import pandas as pd
from pylab import *
from Hyperspreading import Hyperspreading
from transform import Transform


def each_node_influence(N):
    node_influence_dict = {}
    for inode in range(N):
        seeds = [inode]
        scale, I_list = hs.hyperSI(df_hyper_matrix, seeds)
        node_influence_dict[inode] = I_list
    return node_influence_dict


def each_pair_nodes_influence_overlap(N, node_influence_dict):
    matrix = np.zeros((N, N))
    for from_node in range(N):
        for to_node in range(N):
            if from_node == to_node:
                continue
            else:
                arr1 = np.array(node_influence_dict[from_node])
                arr2 = np.array(node_influence_dict[to_node])
                inters = np.intersect1d(arr1, arr2)
                overlap = len(inters)
                matrix[from_node][to_node] = overlap
    df_matrix = pd.DataFrame(matrix)
    return df_matrix


def get_distribution_from_arr(arr):
    count_arr = Counter(arr).most_common(len(arr))
    id_list = []
    val_list = []
    for (idx, val) in count_arr:
        id_list.append(idx)
        val_list.append(val)
    val_number_list = val_list
    val_list = np.array(val_list) / np.sum(np.array(val_list))
    return id_list, val_list, val_number_list


def overlap_with_nodes_neighbors(df_hyper_matrix, N, overlap_df):
    overlap_matrix = overlap_df.values
    matrix = np.dot(df_hyper_matrix, df_hyper_matrix.T)
    overlap_nb_list = []
    overlap_total_list = []
    for i in range(N):
        for j in range(i+1, N):
            if matrix[i][j] > 0:
                overlap_nb_list.append(overlap_matrix[i][j])
            overlap_total_list.append(overlap_matrix[i][j])

    overlap_nb_list = np.array(overlap_nb_list) / N
    overlap_total_list = np.array(overlap_total_list) / N

    overlap_random = np.random.choice(overlap_total_list, size=len(overlap_nb_list))

    id_total_list, val_total_list, val_total_number_list = get_distribution_from_arr(overlap_total_list)
    id_nb_list, val_nb_list, val_nb_number_list = get_distribution_from_arr(overlap_nb_list)
    id_rd_list, val_rd_list, val_rd_number_list = get_distribution_from_arr(overlap_random)

    return id_total_list, val_total_list, val_total_number_list, id_nb_list, val_nb_list, val_nb_number_list, id_rd_list, val_rd_list, val_rd_number_list


def draw_distribution(id_total_list, val_total_list, id_nb_list, val_nb_list, id_rd_list, val_rd_list, fileName, ylim_top):
    markersize = 2000
    border_size = 3
    linewidth = 2
    ax = plt.subplot(1, 1, 1)
    ax.spines['bottom'].set_linewidth(border_size)
    ax.spines['left'].set_linewidth(border_size)
    ax.spines['right'].set_linewidth(border_size)
    ax.spines['top'].set_linewidth(border_size)
    plt.scatter(id_total_list, val_total_list, color="#00a799", edgecolors='#007578', label='Random', alpha=0.9, s=markersize, linewidths=5)
    plt.scatter(id_nb_list, val_nb_list, color="#F79A9E", edgecolors='#d90044', label='Neighbors', alpha=0.9, s=markersize, linewidths=5)

    if fileName == 'Geometry' or fileName == 'Music-Rev':
        y_lim_bottom = -0.002
    elif fileName == 'Bars-Rev':
        y_lim_bottom = -0.004
    else:
        y_lim_bottom = -0.007
    plt.ylim(y_lim_bottom, ylim_top)

    plt.xscale('log')

    plt.tick_params(labelsize=150, pad=20)
    plt.tick_params(width=5, direction='out', length=20)
    plt.subplots_adjust(left=0.2, bottom=0.15)

    if fileName == 'Algebra':
        plt.legend(prop={'size': 90})

    plt.savefig('./overlap/' + fileName + '.png')


if __name__ == '__main__':
    hs = Hyperspreading()

    tf = Transform()

    file_name_list = ['Restaurants-Rev', 'Music-Rev', 'Geometry', 'NDC-classes-unique-hyperedges', 'Algebra', 'Bars-Rev'
        , 'iAF1260b', 'iJO1366']
    ylim_list = [0.2, 0.05, 0.05, 0.3, 0.25, 0.1, 0.3, 0.3]

    for index in range(len(file_name_list)):
        plt.figure(figsize=(28, 20))
        ylim_top = ylim_list[index]
        fileName = file_name_list[index]
        df_hyper_matrix, N = tf.changeEdgeToMatrix('../datasets/' + fileName + '.txt')

        node_influence_dict = each_node_influence(N)
        overlap_df = each_pair_nodes_influence_overlap(N, node_influence_dict)
        id_total_list, val_total_list, val_total_number_list, id_nb_list, val_nb_list, val_nb_number_list, id_rd_list, val_rd_list, val_rd_number_list = overlap_with_nodes_neighbors(df_hyper_matrix, N, overlap_df)

        draw_distribution(id_total_list, val_total_list, id_nb_list, val_nb_list, id_rd_list, val_rd_list, fileName, ylim_top)
        plt.show()


