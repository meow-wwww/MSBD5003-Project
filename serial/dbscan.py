import numpy as np



def dbscan(data, eps, min_pts):
    '''
    data: np.array, shape=(N, 2)
    eps: float
    min_pts: int
    
    Returns:
    labels: np.array, shape=(N,)
    '''
    def find_neighbors(data, point, eps):
        neighbors_id = []
        for i, p in enumerate(data):
            if np.linalg.norm(p - point) < eps:
                neighbors_id.append(i)
        return neighbors_id
    
    N = data.shape[0]
    labels = np.zeros(N, dtype=int) - 1 # -1: unclassified, -2: noise, 1, 2, 3, ...: cluster id
    c = 0
    for i in range(N):
        if labels[i] != -1:
            continue
        neighbors_id = find_neighbors(data, data[i], eps)
        if len(neighbors_id) < min_pts:
            labels[i] = -2
            continue
        c += 1
        labels[i] = c
        neighbors_id_prime = neighbors_id.copy()
        assert i in neighbors_id_prime
        neighbors_id_prime.remove(i)
        for j in neighbors_id_prime:
            if labels[j] == -2:
                labels[j] = c
            if labels[j] != -1:
                continue
            labels[j] = c
            neighbors_j = find_neighbors(data, data[j], eps)
            if len(neighbors_j) >= min_pts:
                neighbors_id_prime += neighbors_j
    return labels
    
    
