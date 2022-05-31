import pandas as pd
import numpy as np
from datamanage import DataManage

class Transform:

    def changeEdgeToMatrix(self, path):
        dm = DataManage()
        node_dict, N, M = dm.generateMap(path)
        print(node_dict)
        matrix = np.random.randint(0, 1, size=(N, M))
        # print(matrix)
        df = pd.read_csv(path, index_col=False, header=None, engine='python')
        arr = df.values
        index = 0
        for each in arr:
            print(list(map(int, each[0].split(" "))))
            edge_list = list(map(int, each[0].split(" ")))
            for edge in edge_list:
                # print(edge, node_dict[edge])
                matrix[node_dict[edge]][index] = 1
            index = index + 1
        print(pd.DataFrame(matrix))
        return pd.DataFrame(matrix), N
