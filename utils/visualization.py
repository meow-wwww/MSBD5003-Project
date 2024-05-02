import numpy as np
import matplotlib.pyplot as plt

# a color list of 50 colors
# color_list = ['#']

def visualize_dataset_with_label(data, label, noise_label=-2,
                                 split_list=None):
    assert data.shape[0] == label.shape[0]
    
    for label_value in set(label).difference({noise_label}):
        data_of_this_cluster = data[label == label_value]
        plt.scatter(data_of_this_cluster[:,0], data_of_this_cluster[:,1], s=1)
    plt.scatter(data[label == noise_label][:,0], data[label == noise_label][:,1], s=15, marker='x', c='red')
    
    if split_list is not None:
        for _, x_start, x_end, y_start, y_end in split_list:
            plt.plot([x_start, x_end, x_end, x_start, x_start], [y_start, y_start, y_end, y_end, y_start])
    
    plt.show()
    return


def visualize_dataset_with_split(data, split_list):
    '''
    split_list: a list of [id, x_start, x_end, y_start, y_end]
    '''
    plt.scatter(data[:,0], data[:,1], s=1)
    for _, x_start, x_end, y_start, y_end in split_list:
        plt.plot([x_start, x_end, x_end, x_start, x_start], [y_start, y_start, y_end, y_end, y_start])
    return