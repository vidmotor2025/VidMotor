import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[i, j] = 1
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_undigraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


class Graph():
    def __init__(self, layout='human36M', strategy='uniform'):
        self.get_edge(layout)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'human36M':
            self.num_node = 17
            self.self_link = [(i, i) for i in range(self.num_node)]
            self.neighbor_link = [(1, 0), (4, 0), (2, 1), (5, 4), (3, 2), (6, 5),
                                  (7, 0), (8, 7), (9, 8), (10, 9), (11, 8), (14, 8),
                                  (12, 11), (13, 12), (15, 14), (16, 15)]  # inward
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):  # Generate the adjacency matrix
        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = get_uniform_graph(self.num_node, self.self_link, self.neighbor_link)
            self.A = A
        elif strategy == 'spatial':
            outward = [(j, i) for (i, j) in self.neighbor_link]
            A = get_spatial_graph(self.num_node, self.self_link, self.neighbor_link, outward)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy.")


