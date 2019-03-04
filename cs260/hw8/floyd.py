#!/usr/bin/env python3

import math


inf = math.inf

expr = ""
vtx_array = []
weight_array = []
# edge_array = []
distance_matrix = []
predecessor_matrix = []

def get_expr():
    global expr
    
    try:
        expr = input()
    except:
        return False

    if len(expr) == 0:
        return False
    return True

def parse_input():
    global expr
    global vtx_array
    global weight_array
    global edge_array
    tokens = []
    while get_expr():
        tokens = expr.split()
        parent_vtx = int(tokens[0])
        vtx_array.append(parent_vtx)
        weight_array.append(((parent_vtx, parent_vtx), 0))
        for i in range(1, len(tokens)):
            vtx = int(tokens[i].split(',')[0])
            weight = int(tokens[i].split(',')[1])
            edge = (parent_vtx, vtx)
            # Undirected graph
            rev_edge = (vtx, parent_vtx)
            
            weight_array.append((edge, weight))
            weight_array.append((rev_edge, weight))

def init_dist_mat():
    global distance_matrix
    for i in range (0, len(vtx_array)):
        col = []
        for j in range (0, len(vtx_array)):
            col.append(inf)
        distance_matrix.append(col)
    for (edge, w) in weight_array:
        distance_matrix[edge[0]][edge[1]] = w

def init_pred_mat():
    global predecessor_matrix
    for i in range (0, len(vtx_array)):
        col = []
        for j in range (0, len(vtx_array)):
            col.append(None)
        predecessor_matrix.append(col)
    for (edge, w) in weight_array:
        predecessor_matrix[edge[0]][edge[1]] = edge[0]

def floyd():
    global distance_matrix
    init_dist_mat()
    init_pred_mat()
    
    for k in range (0, len(vtx_array)):
        for i in range (0, len(vtx_array)):
            for j in range (0, len(vtx_array)):
                if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                    predecessor_matrix[i][j] = k
    
def print_dist_mat():
    print("Distance matrix")
    for i in distance_matrix:
        for j in i:
            print("{:3}".format(j), end=" ")
        print()

def print_pred_mat():
    print("Predecessor matrix")
    for i in predecessor_matrix:
        for j in i:
            print("{:3}".format(j), end=" ")
        print()


def main():
    global distance_matrix
    parse_input()
    floyd()
    print_dist_mat()
    print()
    print_pred_mat()

if __name__ == "__main__":
    main()
