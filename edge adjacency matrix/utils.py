import numpy as np
import itertools
import os
import copy
from random import shuffle

class DATA_PROCESS(object):
    def __init__(self, node_adjacency_matrix):
        nodes_num, _ = np.shape(node_adjacency_matrix)
        self.node_adjacency_matrix = np.ones([nodes_num, nodes_num], dtype=np.int) - np.identity(nodes_num, dtype=np.int32)
        self.lables_adjacency_matrix = node_adjacency_matrix

    def __tranferMatrixToDict(self, node_adjacency_matrix):
        nodes_num, _ = np.shape(node_adjacency_matrix)
        assert nodes_num > 2
        count = 0
        node_adjacency_matrix_dict = {}
        for i in range(nodes_num):
            for j in range(i,nodes_num,1):
                if node_adjacency_matrix[i,j] == 1:
                    node_adjacency_matrix_dict[count] = (i, j)
                    count += 1
        return node_adjacency_matrix_dict

    def __transferToEdgeAdjacencyMatrix(self, node_adjacency_matrix):
        nodes_num, _ = np.shape(node_adjacency_matrix)
        assert nodes_num > 2
        edges_num = np.sum(node_adjacency_matrix) // 2
        edge_adjacency_matrix = np.zeros([edges_num, edges_num])
        for i in range(nodes_num):
            common_nodes_list = []
            for key, value in self.__tranferMatrixToDict(node_adjacency_matrix).items():
                if i in value:
                    common_nodes_list.append(key)
            coordinates = list(itertools.product(common_nodes_list, common_nodes_list))
            for coordinate in coordinates:
                edge_adjacency_matrix[coordinate[0],coordinate[1]] = 1
        return edge_adjacency_matrix

    def __edgeLabels(self, lables_adjacency_matrix):
        node_num, _ = np.shape(lables_adjacency_matrix)
        all_edges_num = node_num * (node_num-1) // 2
        labels = np.zeros([all_edges_num, ], dtype=np.int32)
        all_edges_dict = self.__tranferMatrixToDict(self.node_adjacency_matrix)
        labelled_edges_dict = self.__tranferMatrixToDict(lables_adjacency_matrix)
        labels_id = [key for key, value1 in all_edges_dict.items() for _, value2 in labelled_edges_dict.items() if value1 == value2]
        return np.array(labels_id)

    def get_edge_adjacency_matrix(self):
        return self.__transferToEdgeAdjacencyMatrix(self.node_adjacency_matrix)

    def get_label_id(self):
        return self.__edgeLabels(self.lables_adjacency_matrix)

class LAPLACIAN(object):
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix

    def __laplacian(self, W):
        d = np.sum(W, axis=0)
        d = 1 / np.sqrt(d)
        d = np.diag(d)
        L = np.matmul(np.matmul(d, W), d)
        return np.float32(L)

    def get_laplician(self):
        return self.__laplacian(self.adjacency_matrix)

class INDICES(object):
    def __init__(self, dir, percent=0.4, skiprows = 0, node_num = 62):
        self.dir = dir
        self.precent = percent
        self.skiprows = skiprows
        self.node_num = node_num


    def __convert(self, l):
        ret = copy.deepcopy(l)
        buf = ret[0]-1
        ret[0] = ret[1]-1
        ret[1] = buf
        return ret

    def __read_data(self):
        coordinates = []
        dir = os.path.dirname(os.path.realpath(__file__))
        dir = os.path.join(dir, self.dir)
        with open(dir) as f:
            for i, x in enumerate(f):
                if i > self.skiprows:
                    x = x.strip()
                    x = list(map(int,x.split('\t')))
                    coordinates.append(x)
        positive_indices = list(map(self.__convert, coordinates))
        return positive_indices

    def label_indices(self):
        positive_indices = self.__read_data()
        all_indices = list(itertools.product(list(range(self.node_num)),list(range(self.node_num))))
        all_indices = [value for value in list(map(list, all_indices)) if value[0] < value[1]]
        negative_indices = [value for value in all_indices if not value in positive_indices]
        probe_num = int(np.floor(len(positive_indices)*self.precent))
        shuffle(positive_indices)
        shuffle(negative_indices)
        probe_indices = positive_indices[0:probe_num]
        train_positive_indices = positive_indices[probe_num:]
        train_negative_indices = negative_indices[0:len(train_positive_indices)]
        return train_positive_indices, train_negative_indices, probe_indices


if __name__ == '__main__':
    dir = 'dolphins\out.dolphins'
    data = INDICES(dir).label_indices()
    print(data)

