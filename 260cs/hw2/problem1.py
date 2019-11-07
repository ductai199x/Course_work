#!/usr/bin/env python3

from node import Node


def list_concat(listA, listB):
    first_node = listA
    while listA.next is not None:
        listA = listA.next

    listA.next = listB
    return first_node


def print_list(lst):
    while 1:
        print(str(lst.data), end="")
        if lst.next is None:
            print("")
            break
        else:
            lst = lst.next


def main():
    a = Node(3)
    a = Node(2, a)
    a = Node(1, a)
    print_list(a)

    b = Node(6)
    b = Node(5, b)
    b = Node(4, b)
    print_list(b)

    print_list(list_concat(a, b))


if __name__ == "__main__":
    main()

