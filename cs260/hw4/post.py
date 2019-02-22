#!/usr/bin/env python3
#
# Create a file, one expression per line
#	 redirect from standard input:
#		test.py < input
#
# Notes:  We are not making our input bullet-proof.  If it looks like a #,
# then it is
#
#		Operands must be integers
#
#		The parser doesn't handle negative operands
#

from lexer import *


node_stack = []

class Node:
    def __init__(self, label, left_child, right_child):
        self.label = label
        self.left_child = left_child
        self.right_child = right_child

    def __str__(self):
        if self.left_child == None:
            left = "None"
        else:
            left = self.left_child.label
        if self.right_child == None:
            right = "None"
        else:
            right = self.right_child.label
        return "label: " + self.label + " left: " + left + " right: " + right
        # return self.label

def build_stack():
    global node_stack

    while get_expression():
        t = get_next_token()
        node_stack = []
        while t:
            node_stack.append(str(t))
            t = get_next_token()
        admin_fn()
            
def build_tree(stack, n, stack_len, root):
    global tree

    temp_node = Node(stack.pop(), None, None)

    if n == 0:
        tree = temp_node
        root = temp_node
        stack_len = len(stack)
    if n > 0:
        if n%2 == 1:
            tree.left_child = temp_node
        else:
            tree.right_child = temp_node
            tree = temp_node
    if n >= stack_len:
        tree = root
        return

    n += 1
    build_tree(stack, n, stack_len, root)

def print_pre_tree(tree):
    if tree.left_child is None and tree.right_child is None:
        print(tree.label, end =" ")
    else:
        print(tree.label, end =" ") 
        if tree.left_child != None:
            print_pre_tree(tree.left_child)
        if tree.right_child != None:
            print_pre_tree(tree.right_child)
        
def print_in_tree(tree):
    if tree.left_child is None and tree.right_child is None:
        print(tree.label, end =" ")
    else:
        if tree.left_child != None:
            print_in_tree(tree.left_child)
        print(tree.label, end =" ") 
        if tree.right_child != None:
            print_in_tree(tree.right_child)
        
def print_post_tree(tree):
    if tree.left_child is None and tree.right_child is None:
        print(tree.label, end =" ")
    else:
        if tree.left_child != None:
            print_post_tree(tree.left_child)
        if tree.right_child != None:
            print_post_tree(tree.right_child)
        print(tree.label, end =" ") 

def evaluate(tree):
    if tree.left_child is None and tree.right_child is None:
        return str(tree.label)
    else:
        if tree.left_child != None and tree.left_child != None:
            return str(eval(evaluate(tree.left_child) + tree.label + evaluate(tree.right_child)))

def admin_fn():
    build_tree(node_stack, 0, 0, None)
    print("pre:", end=" ")
    print_pre_tree(tree)
    print()
    print("in:", end=" ")
    print_in_tree(tree)
    print()
    print("post:", end=" ")
    print_post_tree(tree)
    print()
    print("eval:", end=" ")
    print(evaluate(tree))
    print()


def main():
    build_stack()
        
if __name__ == '__main__':
    main()
