#!/usr/bin/env python3

class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __str__(self):
        # pre-order printing of the tree.
        result = ''
        result += str(self.val)
        if self.left:
            result += str(self.left)
        if self.right:
            result += str(self.right)
        return result


def serialize(root):
    n = root
    if n == None: 
        return '#'
    return '{} {} {}'.format(n.val, serialize(n.left), serialize(n.right))

def deserialize(data):
    def helper():
        val = next(vals)
        if val == '#':
            return None
        node = Node(int(val))
        node.left = helper()
        node.right = helper()
        return node
    vals = iter(data.split())
    return helper()


    #     1
    #    / \
    #   3   4
    #  / \   \
    # 2   5   7
tree = Node(1)
tree.left = Node(3)
tree.left.left = Node(2)
tree.left.right = Node(5)
tree.right = Node(4)
tree.right.right = Node(7)

print (serialize(tree))
# 1 3 2 # # 5 # # 4 # 7 # #
print ((deserialize('1 3 2 # # 5 # # 4 # 7 # #')))
# 132547
