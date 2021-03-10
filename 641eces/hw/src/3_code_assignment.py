# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
class TreeNode(object):
    def __init__(self, left=None, right=None, value=None):
        self.left = left
        self.right = right
        self.value = value


# %%
class Tree(object):
    def __init__(self, root=None):
        self.root = root

    def inorder(self):
        def traverse(node):
            travel = []
            if node.left != None:
                travel = traverse(node.left)
            travel.append(node.value)
            if node.right != None:
                travel += traverse(node.right)
            return travel

        return traverse(self.root)

    def preorder(self):
        def traverse(node):
            travel = []
            travel.append(node.value)
            if node.left != None:
                travel += traverse(node.left)
            if node.right != None:
                travel += traverse(node.right)
            return travel

        return traverse(self.root)

    def postorder(self):
        def traverse(node):
            travel = []
            travel.append(node.value)
            if node.left != None:
                travel += traverse(node.left)
            if node.right != None:
                travel += traverse(node.right)
            return travel

        return traverse(self.root)


# %%
class TreeGenerator(Tree):
    def __init__(self, list, type="btree"):
        self.list = list

        if type == "btree":
            self.build_btree_from_list(list)

    def build_btree_from_list(self, arr):
        def insert(node, arr):
            if len(arr) == 0:
                return
            mid = len(arr) // 2

            if node.value == None:
                node.value = arr[mid]
            else:
                node = TreeNode(value=arr[mid])

            node.left = insert(node, arr[:mid])
            node.right = insert(node, arr[mid + 1 :])

            return node

        arr = sorted(arr)
        self.root = TreeNode()
        processed = []
        # insert(self.root, arr, processed, 0, len(arr))
        insert(self.root, arr)
        return self.root


# %%
n4 = TreeNode(value=4)
n5 = TreeNode(value=5)
n2 = TreeNode(left=n4, right=n5, value=2)
n3 = TreeNode(value=3)
n1 = TreeNode(left=n2, right=n3, value=1)

tree = Tree(n1)


# %%
# (a) Inorder (Left, Root, Right) : 4 2 5 1 3
# (b) Preorder (Root, Left, Right) : 1 2 4 5 3
# (c) Postorder (Left, Right, Root) : 4 5 2 3 1
# print(tree.inorder())
# print(tree.preorder())
# print(tree.postorder())


# %%
print("\n## Problem 1 ##\n")
arr = [1, 2, 3, 4, 4, 5, 6, 7, 8, 9]
print(f"array is: {arr}")
btree = TreeGenerator(arr)
print(f"Inorder from btree: {btree.inorder()}")


# %%
import math
import numpy as np

from Bio import SeqIO, AlignIO
from Bio.Align import MultipleSeqAlignment


# %%
with open("Coronavirus_Data.fasta", "r") as handle:
    records = list(SeqIO.parse(handle, "fasta"))

print("\n## Problem 2 ##\n")
# %%
epsilon = 2.2204e-16


def jukes_cantor(sequence1, sequence2):
    """Calculate the Jukes-Cantor distance between the two provided aligned
    sequences.
    """

    # Initialization
    difference_counter = 0
    length_counter = 0

    # Step 1: Count differences between sequences, ignoring gaps
    for i in range(min(len(sequence1), len(sequence2))):
        if sequence1[i] != "-" and sequence2[i] != "-":
            length_counter += 1
            if sequence1[i] != sequence2[i]:
                difference_counter += 1

    # Step 2: Calculate and return results
    p = difference_counter / length_counter
    jukes = -3.0 / 4.0 * math.log(max(epsilon, 1 - 4.0 / 3.0 * p))
    return jukes


# %%
align = MultipleSeqAlignment([])
for r in records:
    align.add_sequence(r.id, r.seq.__str__())


# %%
# dist_mat = np.zeros((len(align), len(align)))
# for i in range(len(align)):
#     for j in range(i+1, len(align)):
#         dist_mat[i][j] = jukes_cantor(align[i], align[j])

# dist_mat += dist_mat.T


# %%
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceCalculator

constructor = DistanceTreeConstructor()
calculator = DistanceCalculator("identity")
dm = calculator.get_distance(align)
upgmatree = constructor.upgma(dm)

print(upgmatree)

# %%
dist_mat = dm.matrix.copy()
labels = dm.names.copy()
l = len(dist_mat[-1])
for i in range(l):
    dist_mat[i] += [0] * (l - i - 1)
dist_mat = np.array(dist_mat)
dist_mat += dist_mat.T


# %%
HAS_PANDAS = True
try:
    import pandas as pd
except ImportError:
    HAS_PANDAS = False

print("NORMALIZED DISTANCE MATRIX:")
if HAS_PANDAS:
    labels = [r.id for r in align]
    df = pd.DataFrame(dist_mat, columns=labels, index=labels)
    pd.options.display.float_format = "{:,.2f}".format
    print(df)
else:
    print(dm)


# %%
target = "Human_Sars_CoV"
t = dist_mat[labels.index(target), :]
closest = labels[np.argpartition(t, 1)[1]]
print(f"{closest} is closest to {target}")


# %%
target = "2019_nCoV_CDS1"
t = dist_mat[labels.index(target), :]
closest = labels[np.argpartition(t, 1)[1]]
print(f"{closest} is closest to {target}")
