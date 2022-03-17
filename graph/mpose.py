import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_nodes = 13 # reduced keypts


self_link = [(i, i) for i in range(num_nodes)]

inward_ori_index = [ # Kinetics-skeleton (OpenPose)
    (3, 4),
    (3, 2),
    (2, 1),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (1, 11),
    (8, 9),
    (9, 10),
    (11, 12),
    (12, 13),
]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_nodes = num_nodes
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        
        # for msg3d
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.neighbor, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.neighbor + self.self_loops, self.num_nodes)


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_nodes, self_link, inward, outward) # num_node -> num_nodes
        else:
            raise ValueError()
        return A
