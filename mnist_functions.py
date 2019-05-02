import numpy as np
import pandas as pd


valid_vectors = []
for i in range(28):
    for j in range(28):
        valid_vectors.append([i, j])

def get_white_group(btmap, pos_vec, group):
    group.append(pos_vec)
    neighbors = get_white_neighbors(btmap, pos_vec)
    for neighbor in neighbors:
        if neighbor not in group:
            group.append(neighbor)
            get_white_group(btmap, neighbor, group)
            
def get_white_neighbors(btmap, pos_vec):
    neighbors = []
    if pos_vec[0] >= 0 and pos_vec[0] < 28:
        if pos_vec[0] != 0:
            if btmap[pos_vec[0] - 1, pos_vec[1]] == 0:
                neighbors.append([pos_vec[0] - 1, pos_vec[1]])
        if pos_vec[0] != 27:
            if btmap[pos_vec[0] + 1, pos_vec[1]] == 0:
                neighbors.append([pos_vec[0] + 1, pos_vec[1]])
            
    if pos_vec[1] >= 0 and pos_vec[1] < 28:
        if pos_vec[1] != 0:
            if btmap[pos_vec[0], pos_vec[1] - 1] == 0:
                neighbors.append([pos_vec[0], pos_vec[1] - 1])
        if pos_vec[1] != 27:
            if btmap[pos_vec[0], pos_vec[1] + 1] == 0:
                neighbors.append([pos_vec[0], pos_vec[1] + 1])
    return neighbors
def get_btmap_from_vectors(vectors):
    btmap = np.zeros(shape=(28, 28))
    for v in vectors:
        btmap[v[0], v[1]] = 1
    return btmap

def get_white_groups(btmap):
    groups = []
    num_of_groups = 0
    for v in valid_vectors:
        if v not in groups and btmap[v[0], v[1]] == 0:
            group = []
            get_white_group(btmap, v, group)
            if group:
                if len(group) > 2:
                    num_of_groups = num_of_groups + 1
                groups.extend(group)
    return num_of_groups, groups

def color_changes_cnt(btarray):
    changes_cnt = 0
    color = np.ceil(btarray[0]/255)
    for bit in btarray:
        if np.ceil(bit/255) != color:
            changes_cnt = changes_cnt + 1
            color = np.ceil(bit/255)
    return changes_cnt

def color_changes(btmap):
    row_color_changes = []
    for row in btmap:
        row_color_changes.append(color_changes_cnt(row))
    column_color_changes = []
    for column in btmap.T:
        column_color_changes.append(color_changes_cnt(column))
    
    return row_color_changes, column_color_changes