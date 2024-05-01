import typing

def transform(i: int, S: int, k, noise_label=-2) -> int:
    '''
    i: split id
    S: total number of splits
    k: the cluster id in split #i
    '''
    if k == noise_label:
        return noise_label
    else:
        return (k-1)*S + i