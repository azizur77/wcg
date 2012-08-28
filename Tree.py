#'''
#Created on 2011-01-11
#
#@author: kazem
#'''
class Node(object):
    def __init__(self, id_):
        self.id = id_
        self.left = None
        self.right = None
        self.parent = None
        
    def preorder(self):
        print self.id
        if self.left != None:
            self.left.preorder() 
        if self.right != None:
            self.right.preorder()
class binary_tree:
    def __init__(self):
        self.root = None
        self.__count = 0

    def add_node(self, id_):
        Q = []
        visited = {}
        if self.root is None:
            self.root = Node(id_)
            self.__count += 1
            return
        else:
            Q.append(self.root)
            visited[self.root] = 1
            while len(Q) > 0:
                v = Q.pop(0)
                if v.left == None:
                    v.left = Node(id_)
                    v.left.parent = v
                    self.__count += 1
                    return
                elif v.left not in visited:
                    visited[v.left] = 1
                    Q.append(v.left)
                if v.right == None:
                    v.right = Node(id_)
                    v.right.parent = v
                    self.__count += 1
                    return
                elif v.right not in visited:
                    visited[v.right] = 1
                    Q.append(v.right)                  
    #print all leaves of the tree in BFS order
    def BFS_leaves(self, object):
        out = []
        Q = []
        visited = {}
        if object is None:
            print "no element in tree!"
            return
        else:
            Q.append(object)
            visited[object] = 1
            while len(Q) > 0:
                v = Q.pop(0)
                if v.left == None and v.right == None:
                    #print v.id
                    out.append(v)
                if v.left != None and v.left not in visited:
                    visited[v.left] = 1
                    Q.append(v.left)
                if v.right != None and v.right not in visited:
                    visited[v.right] = 1
                    Q.append(v.right)                  
        return out
    #print all leaves of T that are at distance d from v 
    def distance(self, v, d):
        k = d
        visited = []
        if k == 1:
            out = [v.id]
            return out
        else:
            while k - 1 > 0:
                #add every node that we visits to ignore duplicates
                visited.append(v)
                parent = v.parent
                #print "parent of %d is %d" % (v.id, parent.id)
                v = parent
                k -= 1
        #now let's traverse the other side of the tree rooted at v
        if v.left not in visited:
            out_temp = self.BFS_leaves(v.left)
        else:
            out_temp = self.BFS_leaves(v.right)
        out = []
        for n_ in out_temp:
            out.append(n_.id)
        return out
    def preorder(self):
        print self.root.id
        if self.root.left != None:
            self.root.left.preorder()
        if self.root.right != None:
            self.root.right.preorder()