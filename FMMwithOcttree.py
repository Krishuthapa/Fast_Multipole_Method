import numpy as np
import pandas as pd

import sys

import matplotlib.pyplot as plt

import time

import matplotlib.pyplot as plt


def generatePoints(dimension = 512 , count = 2500):
    points = np.random.randn(count, dimension)
    
    return points

points = generatePoints(4,1000)

scale = np.array([63 - 0] * points.shape[1])/(np.max(points,axis = 0) - np.min(points,axis = 0))
shift = np.min(points,axis =0) * scale - 0

new_points = np.round(points*scale - shift).astype(int)
all_points_binary = [] 

for point in new_points:
    binary_representations = np.array([np.binary_repr(x, width=6) for x in point])

    one_dim_binary_representation = ''.join([''.join(column) for column in zip(*binary_representations)])

    all_points_binary.append(one_dim_binary_representation)

zipped_points_info = list(zip(all_points_binary,[value for value in range(len(all_points_binary))]))
sorted_zipped_points_info = sorted(zipped_points_info,key = lambda x: x[0])

sorted_points,sorted_indices = zip(*sorted_zipped_points_info)

sorted_points = np.array(sorted_points)
sorted_indices = np.array(sorted_indices)


class Node:
    def __init__(self,identifier = '', isLeaf = False, isRoot = False):
        super(Node,self).__init__()

        self.identifier = identifier
        self.isLeaf = isLeaf
        self.isRoot = isRoot

        self.index = None

        self.children = None
        self.parent = None

        self.multipole_val = None
        
        self.proximity_node_indices = []
        self.far_node_indices = []

    def setChildren(self,children):
        self.children = children
    
    def setParent(self,parent):
        self.parent = parent
        
class Octtree:
    def getLCA(self,traversed_node,point):
        checking_node = traversed_node.parent
        
        if checking_node.identifier == point[0:len(checking_node.identifier)]:
            return checking_node, traversed_node,point
        
        if checking_node.isRoot and len(checking_node.identifier) == 0:
            return checking_node, traversed_node, checking_node.identifier
        
        return self.getLCA(checking_node, point[:-2])
    
    def getCommonPattern(self,comparing,actual):
        commonPattern = ''

        for index in range(0,len(comparing),3):            
            if comparing[index:index+3] == actual[index:index+3]:
                commonPattern+=comparing[index:index+3]
            else:
                break

        return commonPattern
    
    def getNearOrFarFieldDistinction(self, current_node, node_id, candidate_node_classes):
        candidate_node = self.node_id_map[node_id]

        euclidean_distance = np.linalg.norm(candidate_node.multipole_val - current_node.multipole_val)

        if euclidean_distance > 2.5:
            candidate_node_classes.append([False,candidate_node.index])
            return candidate_node_classes
        
        if len(candidate_node.identifier) >= len(current_node.identifier):
            candidate_node_classes.append([True,candidate_node.index])
            return candidate_node_classes
        
        for child in candidate_node.children:
            candidate_node_classes = self.getNearOrFarFieldDistinction(current_node,child.index,candidate_node_classes)

        return candidate_node_classes

    def getNearAndFarField(self,tree):
        queue = [tree]

        while len(queue)>0:
            node = queue.pop(0)
                
            candidate_node_classes = []

            if node.isRoot:
                node.proximity_node_indices = [node.index]

                if len(node.children) ==0:
                    continue 
                   
                for child in node.children:
                    queue.append(child)
                
                continue

            candidate_node_indices= node.parent.proximity_node_indices
                
            for candidate_node_id in candidate_node_indices:
                candidate_node_classes =  self.getNearOrFarFieldDistinction(node,candidate_node_id, candidate_node_classes) 
                
            if len(candidate_node_classes) == 0:
                node.proximity_node_indices = node.parent.proximity_node_indices
                node.far_node_indices = []
                
                                   
            for candidate_node_class in candidate_node_classes:
                isNear, candidate_node_id = candidate_node_class

                if isNear:
                    node.proximity_node_indices.append(candidate_node_id)
                    continue
                    
                node.far_node_indices.append(candidate_node_id)
            
            #print("Candidates", node.index, node.identifier, node.proximity_node_indices, node.far_node_indices)

            if node.isLeaf:
                continue
            
            if len(node.children)>0:    
                for child in node.children:
                    queue.append(child)

        
    def addPoint(self,latest_node, points, leaf_index, node_index):

        if len(points) <= leaf_index:
            return
        
        point = points[leaf_index]
        
        LCA, LCA_child, identifier= self.getLCA(latest_node,point[:-3])
        commonPattern = self.getCommonPattern(latest_node.identifier,points[leaf_index])
        
        if (LCA.isRoot and len(commonPattern) > 0) or (LCA !=None and len(LCA.identifier) < len(commonPattern)):
            intermediate_node = Node(identifier = commonPattern)
            intermediate_node.parent = LCA
            intermediate_node.index = node_index

            node_index+=1

            LCA.children.pop()
            LCA.children.append(intermediate_node)

            LCA_child.parent  = intermediate_node

            leaf_node = Node(identifier= points[leaf_index], isLeaf= True)
            leaf_node.index = self.points_indices[leaf_index]
            leaf_node.multipole_val = self.actual_points[leaf_index]
            leaf_index += 1

            leaf_node.parent = intermediate_node
            intermediate_node.children = [LCA_child,leaf_node]

            return (leaf_node,leaf_index,node_index)   
        
        if LCA.isRoot and len(commonPattern) == 0:
            leaf_node = Node(identifier= points[leaf_index],isLeaf=True)
            leaf_node.index = self.points_indices[leaf_index]
            leaf_node.multipole_val = self.actual_points[leaf_index]
            leaf_index += 1

            LCA.children.append(leaf_node)
            leaf_node.parent = LCA

            return (leaf_node,leaf_index,node_index)
                
        leaf_node = Node(identifier= points[leaf_index],isLeaf=True)
        leaf_node.index = self.points_indices[leaf_index]
        leaf_node.multipole_val = self.actual_points[leaf_index]
        leaf_index += 1

        leaf_node.parent = LCA

        LCA.children.append(leaf_node)

        return (leaf_node,leaf_index,node_index)
     
    def performMultipoleExpansion(self,node):
        if node.isLeaf:
            return

        node_multipole = []

        for child in node.children:
            self.performMultipoleExpansion(child)
            node_multipole.append(child.multipole_val)
        
        avg_value = np.sum(node_multipole,0)/len(node.children)
        node.multipole_val = avg_value

    
    def generateLeafIdentifierMap(self,node):
        self.node_id_map[node.index] = node

        if node.isLeaf:
            return
        
        for child in node.children:
            self.generateLeafIdentifierMap(child)

        return self.node_id_map
        

    def buildTree(self):
        index = len(self.actual_points)
        point_index = 0

        rootNode = Node(identifier = "",isRoot = True)
        rootNode.index = index
        index+=1
        
        firstLeaf = Node(identifier=self.points[point_index],isLeaf= True)
        firstLeaf.index = self.points_indices[point_index]
        firstLeaf.multipole_val = self.actual_points[point_index]

        rootNode.children = [firstLeaf]
        firstLeaf.parent = rootNode

        latest_leaf = firstLeaf
        point_index+=1

        while point_index < len(self.points):
            latest_leaf, point_index, index= self.addPoint(latest_leaf,self.points,point_index,index)
        
        return rootNode

    def __init__(self,binary_points,indices,actual_points):
        super(Octtree,self).__init__()
        
        self.points = binary_points
        self.points_indices = indices
        self.actual_points = actual_points

        self.node_id_map = {}

octtree = Octtree(binary_points=sorted_points,indices = sorted_indices, actual_points = points[sorted_indices])

start_time = time.time()
tree = octtree.buildTree()
end_time = time.time()

print("Runtime for tree construction", end_time - start_time)

mp_start_time = time.time()
octtree.performMultipoleExpansion(tree)
mp_end_time = time.time()

print("Runtime for multipole expansion", mp_end_time - mp_start_time)

node_id_map = octtree.generateLeafIdentifierMap(tree)
octtree.getNearAndFarField(tree)




    
