#!/usr/bin/env python3

from node import Node
from problem1 import print_list

def list_concat_copy(listA, listB):
    new_list1 = listA
    new_list2 = listB
    first_node = new_list1
    while new_list1.next:
        new_list1 = new_list1.next

    new_list1.next = new_list2
    return first_node


def main():
    a = Node(3)
    a = Node(2, a)
    a = Node(1, a)
    print_list(a)

    b = Node(6)
    b = Node(5, b)
    b = Node(4, b)
    print_list(b)

    print_list(list_concat_copy(a, b))


if __name__ == "__main__":
    main()

