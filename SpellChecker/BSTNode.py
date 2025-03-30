# DICTIONARY
# binary search tree to store all words in dictionary
# only needs insert and exist functions, print functions are helpers when testing
class BSTNode:
    def __init__(self, val=None):
        self.left = None
        self.middle = None
        self.right = None
        self.val = val


    def insert(self, val):
        # if the tree is empty, insert root node
        if not self.val: 
            self.val = val
            return

        # if given value is equal to current nodes val
        if self.val == val:
            return
        
        # insert into left node
        if val < self.val: # if given val is less than current nodes value
            if self.left:  # if left node is a not a leaf node, continue down left side
                self.left.insert(val)
                return
            self.left = BSTNode(val)  # if left node IS a leaf node, insert new value
            return

        # insert into right node
                       # if given val is greater than current nodes value
        if self.right: # if right node is not a leaf node, continue down right side
            self.right.insert(val)
            return
        self.right = BSTNode(val) # if right node IS a leaf node, insert new value


    def exists(self, val):
        if val == self.val:
            return True

        if val < self.val:
            if self.left == None:
                return False
            return self.left.exists(val)

        if self.right == None:
            return False
        return self.right.exists(val)

    # values of the tree inorder
    def inorder(self, vals):
        if self.left is not None:
            self.left.inorder(vals)
        if self.val is not None:
            vals.append(self.val)
        if self.right is not None:
            self.right.inorder(vals)
        return vals



def create_dict(fname):
    dictionary = BSTNode()
    with open(fname) as f:
        for line in f:
            words = line.split()
            for word in words:
                dictionary.insert(word)
    return dictionary
    # create a tree storing all words
    # traverse using disjkstras using minimum distance as the hueristic function