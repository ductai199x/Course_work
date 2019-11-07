#!/usr/bin/env python3

import random
import timeit

number_of_trees = 20
size_diff = 100
min_num = 0
max_num = 100


def down_heap(heap, x, heap_size):
    largest = x
    left_child = 2 * largest + 1
    right_child = 2 * largest + 2
    
    if left_child < heap_size and heap[left_child] < heap[x]:
        largest = left_child
    if right_child < heap_size and heap[right_child] < heap[largest]:
        largest = right_child
    
    if largest != x:
        swap(largest, x, heap)
        down_heap(heap, largest, heap_size)   


def make_heap(array):
    arr_length = len(array)
    for i in reversed(range(0, arr_length // 2)):
        down_heap(array, i, arr_length)


def swap(n, m, heap):
    t = heap[n]
    heap[n] = heap[m]
    heap[m] = t

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def main():
    sizes_array = [i for i in range (10, size_diff*(number_of_trees + 1), size_diff)]
    trees_array = []

    for size in sizes_array:
        trees_array.clear()
        for i in range (0, size):
            trees_array.append(random.randint(min_num, max_num))
        
        wrapped = wrapper(make_heap, trees_array)
        print(size, timeit.timeit(wrapped, number=1000))    

if __name__ == "__main__":
    main()


