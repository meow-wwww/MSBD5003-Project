#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:50:44 2024

@author: a86183
"""

def compute_area_list(dataset, epsilon):
    '''
    Input:
        dataset(list of tuples): A list of 2D points represented as tuples (x, y)
        epsilon: The max distance allowed between points in the same box
    Output:
        area_list(list of lists): A list of boxes, where each box is represented as a list of 2D points
    '''
    # dataset = [(data_x[i], data_y[i]) for i in range(len(data_x))]
    sorted_dataset = sorted(dataset, key=lambda p: p[0])

    # build strips
    strips = []
    strip = [sorted_dataset[0]]
    for i in range(1, len(sorted_dataset)):
        point = sorted_dataset[i]
        if point[0] - strip[0][0] <= epsilon / (2 ** 0.5):
            strip.append(point)
        else:
            strips.append(strip)
            strip = [point]
    strips.append(strip)

    # build boxes
    area_list = []
    for strip in strips:
        strip = sorted(strip, key = lambda p: p[1])
        current_box = [strip[0]]
        for i in range(1, len(strip)):
            point = strip[i]
            if point[1] - current_box[0][1] <= epsilon / (2 ** 0.5):
                current_box.append(point)
            else:
                area_list.append(current_box)
                current_box = [point]
        area_list.append(current_box)

    return area_list



def point_list_to_bounding_box(point_list_list, expand=0):
    area_list = []
    for area_id, point_list in enumerate(point_list_list):
        x_list = [p[0] for p in point_list]
        y_list = [p[1] for p in point_list]
        assert len(x_list) == len(y_list)
        assert len(x_list) == len(point_list)
        area_list.append([area_id, min(x_list)-expand, max(x_list)+expand, min(y_list)-expand, max(y_list)+expand])
    return area_list