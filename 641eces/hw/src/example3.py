class TreeNode:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val


def TreeGenerator(A):
    if A is None or len(A) == 0:
        return None
    mid = len(A) // 2
    root = TreeNode(A[mid])
    root.left = TreeGenerator(A[:mid])
    root.right = TreeGenerator(A[mid + 1 :])
    return root