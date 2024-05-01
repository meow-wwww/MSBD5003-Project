import numpy as np

def check_cluster_number(labels, min_pts):
    '''
    labels: np.array, shape=(N,)
    '''
    for i in set(labels).difference({0}):
        if np.sum(labels == i) < min_pts:
            assert False, f'Cluster {i} has less than min_pts points.'