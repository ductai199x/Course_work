#!/usr/bin/env python3

from set import Set
import random
import time
import copy

def bfs(tree):
    is_visited = [0 for i in range(0, len(tree.parents)) ]
    queue = []
    
    root_idx = tree.parents.index(-1)

    queue.append(tree.values[root_idx])
    is_visited[root_idx] = 1

    print("BFS Traversal: ")
    while sum(is_visited) < len(is_visited):
        if queue:
            i = queue.pop(0)
            print(i, end=" ")
            # print("popping: ", i)

        for j in range(0, len(tree.parents)):
            if is_visited[j]:
                continue
            if tree.parents[j] == root_idx:
                queue.append(tree.values[j])
                is_visited[j] = 1
        # print("queue: ", queue)
        root_idx = tree.values.index(i)
        # print("root_idx: ", root_idx)
    while queue:
        print(queue.pop(0), end=" ")
    print()

def dfs(tree):
# In-order traversal
    is_visited = [0 for i in range(0, len(tree.parents)) ]
    i = tree.parents.index(-1)
    
    print("DFS Traversal: ")
    dfs_helper(tree, is_visited, i)
    print()

def dfs_helper(tree, is_visited, i):
    is_leaf = True
    is_print = False
    for j in range(0, len(tree.parents)):
        if is_visited[j]:
            continue
        if tree.parents[j] == i:
            is_leaf = False
            is_visited[i] = 1
            if not is_print:
                print(tree.values[i], end=" ")
                is_print=True
            dfs_helper(tree, is_visited, j)
    if is_leaf:
        print(tree.values[i], end=" ")

def build_tree():
    random.seed(time.time())
    rand_list = []
    passed=True
    # First make a list of stuff for the set system
    for i in range(10):
        val = random.randrange(-1000,1000)
        if(val not in rand_list):
            rand_list.append(val)

    cpy_list = copy.deepcopy(rand_list)
    ret_set = Set(rand_list)

    print("Constructing a tree by doing lots of unions")
    while(cpy_list.count(cpy_list[0]) != len(rand_list)):
        length = len(rand_list)
        idx1 = random.randrange(0, length)
        val1 = rand_list[idx1]
        idx2 = random.randrange(0, length)
        val2 = rand_list[idx2]
        set1=cpy_list[idx1]
        set2=cpy_list[idx2]
        for i in range(len(cpy_list)):
            if ( cpy_list[i] == set1 ):
                cpy_list[i] = set2
        ret_set.Merge2(ret_set, val1, val2)
    
    print(ret_set.values)
    print(ret_set.parents)
    return ret_set

def main():
    tree = build_tree()
    bfs(tree)
    dfs(tree)

if __name__ == "__main__":
    main()
