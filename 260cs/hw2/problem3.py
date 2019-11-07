#!/usr/bin/env python3

import random
import time
from node import Node
from problem1 import list_concat
from problem2 import list_concat_copy

min_num_el = 1000
max_num_el = 15000
step = 1000


def generate_list(length):
    lst = None
    for i in range(0, length):
        if i == 0:
            lst = Node(random.randint(1, 101))
        else:
            lst = Node(random.randint(1, 101), lst)

    return lst


def main():
    lists_of_length = [i for i in range(min_num_el, max_num_el + step, step)]
    listA = None
    listB = None
    for length in lists_of_length:
        listA = generate_list(length)
        listB = generate_list(length)

        t1 = time.clock()
        list_concat(listA, listB)
        print("%d %f" % (length, time.clock() - t1), end="")

        t2 = time.clock()
        list_concat_copy(listA, listB)
        print(" %f" % (time.clock() - t2))
        print("")


if __name__ == "__main__":
    main()

