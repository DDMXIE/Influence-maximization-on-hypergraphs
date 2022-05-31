import os
import re
from typing import (
    Iterable,
    List,
    Tuple,
    Set,
    Dict,
    Union
)
import pandas as pd
import numpy as np
import random
from enum import Enum
from scipy.io import loadmat

dataset_list = [
    "chuancai.csv",
    "yuecai.csv",
    "iAB_RBC_283.mat",
    "iAF692.mat",
    "iAF1260b.mat",
    "iHN637.mat",
    "iIT341.mat",
    "iJO1366.mat"
]
'iAB_RBC_283.mat'

# 读取dataframe格式的数据

####################
##这是一个牛X的分隔符##
####################


'''
对于超链路预测来说，分割数据集分为以下几步：
    输入超图H的超边集合 和 生成超边集合
    1. 从H的超边集合中随机数量的超边，与生成超边集合组成集合B，剩余超边为集合A
    2. 在训练时，需要用到集合A与集合B，其中集合A为正样本，集合B为负样本
    3. 在测试时，只需要用到集合B，此时集合B按来源分为正样本或者负样本
    输出：集合A ， (集合B ， 集合B的标签)
'''


def splitDataset(raw_incidence: pd.DataFrame, simulation_edges: pd.DataFrame, missnumber: int = 25) -> Dict[
    str, Union[pd.DataFrame, pd.Series]]:
    '''数据集分离'''
    columns_list = list(raw_incidence.columns)
    random.shuffle(columns_list)
    train_set, miss_set = raw_incidence[columns_list[missnumber:]], raw_incidence[columns_list[:missnumber]]

    '''生成标签'''
    test_set = pd.concat([miss_set, simulation_edges], axis=1)
    test_label = pd.concat(
        [
            pd.Series(np.ones(miss_set.shape[1]), index=miss_set.columns),
            pd.Series(np.zeros(simulation_edges.shape[1]), index=simulation_edges.columns)
        ]
    )

    return {
        "train_set": train_set,
        "test_set": test_set,
        "test_label": test_label
    }


def reader(*, dataset: str = "chuancai.csv", datadir: str = "./input_data") -> pd.DataFrame:
    assert dataset in dataset_list, "所给dataset未预先设置"
    "在给定路径下搜索指定的数据集"
    for father_dir, _, filename_list in os.walk(datadir):  # 生成三元组：(<根路径>,<文件夹名>,<文件名>)
        if dataset in filename_list:
            file_path = os.path.join(father_dir, dataset)
            break
    else:
        raise FileNotFoundError(f"在\"{os.path.abspath(datadir)}\"中没有找到文件：{dataset}！！！")

    suffix, df_U = dataset.split(".")[-1], None

    if suffix == "csv":
        df_S = pd.read_csv(file_path, index_col=0)
        ...  # 这里以后补上生成的超边集合
    elif suffix == "mat":
        Model = loadmat(file_path)["Model"]
        S, U = Model["S"][0, 0], Model["US"][0, 0]
        S[S != 0] = 1
        U[U != 0] = 1

        df_S = pd.DataFrame(S.todense())
        reaction_names = map(lambda x: str(x[0]), Model["rxns"][0, 0].squeeze())
        metabolite_names = tuple(map(lambda x: str(x[0]), Model["mets"][0, 0].squeeze()))
        df_S.index, df_S.columns = metabolite_names, reaction_names
        df_S = df_S.T.drop_duplicates().T

        df_U = pd.DataFrame(U)
        U_reaction_names = map(lambda x: str(x[0]), Model["unrnames"][0, 0].squeeze())
        df_U.columns, df_U.index = U_reaction_names, metabolite_names
    else:
        ...

    return df_S, df_U


def changeEdgeToMatrix(datadir, dataset):
    raw_incidence, complete_incidence = reader(datadir=datadir, dataset=dataset)
    # raw_incidence, complete_incidence = reader(datadir="../datasets/", dataset='chuancai.csv')
    raw_incidence.columns = np.arange(len(raw_incidence.columns.values))
    raw_incidence.index = np.arange(len(raw_incidence.index.values))

    raw_incidence = pd.DataFrame(raw_incidence, dtype=int)
    return raw_incidence, len(raw_incidence.index.values)

# if __name__ == "__main__":

# simulation_edges = complete_incidence[set(complete_incidence.columns) - set(raw_incidence.columns)]

# print(splitDataset(raw_incidence,simulation_edges)["test_label"])
# splitDataset(H.incidence_matrix())