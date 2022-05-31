import pandas as pd
import numpy as np

class DataManage:
    def generateMap(self, path):
        node_dict = {}
        df = pd.read_csv(path, index_col=False, header = None)
        arr = df.values
        node_list = []
        for each in arr:
            # print(list(map(int, each[0].split(" "))))
            node_list.extend(list(map(int, each[0].split(" "))))
        node_arr = np.unique(np.array(node_list))
        for i in range(0, len(node_arr)):
            node_dict[node_arr[i]] = i
        return node_dict, len(list(node_arr)), len(arr)

# if __name__ == '__main__':
#     dataMap = DataManage()
#     dataMap.generateMap('../datasets/Restaurants-Rev.txt')