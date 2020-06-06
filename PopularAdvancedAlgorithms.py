# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:50:58 2020

@author: Edgar
"""

"""Dijkstra's Algorithm """
"""Dijkstra's algorithm is an algorithm for finding the shortest paths between nodes in a graph, which may represent, for example, road networks."""
"""The difference between Dijkstra and BFS is that with BFS we have a simple FIFO queue, 
and the next node to visit is the first node that was added in the queue. But, using Dijkstra, 
we need to pull the node with the lowest cost so far."""
"""https://www.geeksforgeeks.org/python-program-for-dijkstras-shortest-path-algorithm-greedy-algo-7/"""
import sys 
  
class Graph(): 
  
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [[0 for column in range(vertices)]  
                    for row in range(vertices)] 
  
    def printSolution(self, dist): 
        print "Vertex tDistance from Source"
        for node in range(self.V): 
            print node, "t", dist[node] 
  
    # A utility function to find the vertex with  
    # minimum distance value, from the set of vertices  
    # not yet included in shortest path tree 
    def minDistance(self, dist, sptSet): 
  
        # Initilaize minimum distance for next node 
        min = sys.maxint 
  
        # Search not nearest vertex not in the  
        # shortest path tree 
        for v in range(self.V): 
            if dist[v] < min and sptSet[v] == False: 
                min = dist[v] 
                min_index = v 
  
        return min_index 
  
    # Funtion that implements Dijkstra's single source  
    # shortest path algorithm for a graph represented  
    # using adjacency matrix representation 
    def dijkstra(self, src): 
  
        dist = [sys.maxint] * self.V 
        dist[src] = 0
        sptSet = [False] * self.V 
  
        for cout in range(self.V): 
  
            # Pick the minimum distance vertex from  
            # the set of vertices not yet processed.  
            # u is always equal to src in first iteration 
            u = self.minDistance(dist, sptSet) 
  
            # Put the minimum distance vertex in the  
            # shotest path tree 
            sptSet[u] = True
  
            # Update dist value of the adjacent vertices  
            # of the picked vertex only if the current  
            # distance is greater than new distance and 
            # the vertex in not in the shotest path tree 
            for v in range(self.V): 
                if self.graph[u][v] > 0 and \ 
                     sptSet[v] == False and \ 
                     dist[v] > dist[u] + self.graph[u][v]: 
                    dist[v] = dist[u] + self.graph[u][v] 
  
        self.printSolution(dist) 
        
"""Bellman-ford algorithm"""
"""Does the same thing as Dijkstra's algorithm, but it is more versatile."""
"""Dijkstra’s algorithm is a Greedy algorithm and time complexity is O(VLogV) (with the use of Fibonacci heap). 
Dijkstra doesn’t work for Graphs with negative weight edges, Bellman-Ford works for such graphs. 
Bellman-Ford is also simpler than Dijkstra and suites well for distributed systems. 
But time complexity of Bellman-Ford is O(VE), which is more than Dijkstra."""
def BellmanFord(self, src):  
  
        # Step 1: Initialize distances from src to all other vertices  
        # as INFINITE  
        dist = [float("Inf")] * self.V  
        dist[src] = 0
  
  
        # Step 2: Relax all edges |V| - 1 times. A simple shortest  
        # path from src to any other vertex can have at-most |V| - 1  
        # edges  
        for _ in range(self.V - 1):  
            # Update dist value and parent index of the adjacent vertices of  
            # the picked vertex. Consider only those vertices which are still in  
            # queue  
            for u, v, w in self.graph:  
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:  
                        dist[v] = dist[u] + w  
  
        # Step 3: check for negative-weight cycles. The above step  
        # guarantees shortest distances if graph doesn't contain  
        # negative weight cycle. If we get a shorter path, then there  
        # is a cycle.  
  
        for u, v, w in self.graph:  
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:  
                        print("Graph contains negative weight cycle") 
                        return
                          
        # print all distance  
        self.printArr(dist)  
"""Floyd's cyccle detection algorithm
Algorithm for detecting loops in linked list and graphs using 2 pointers
This also detects the node at which the loop starts so the loop can be removed"""
#https://leetcode.com/problems/linked-list-cycle-ii/
def detectCycle(head):
    if head is None:
        return None
    
    tortoise = head
    rabbit = head
    
    flag = False
    first_meeting = None
    while tortoise and rabbit and rabbit.next:
        rabbit = rabbit.next.next
        tortoise = tortoise.next
        if tortoise == rabbit:
            flag = True
            first_meeting = tortoise
            break
        
    if not flag:
        return None
    else:
        pt1 = head
        pt2 = first_meeting
        while pt1 != pt2:
            pt1 = pt1.next
            pt2 = pt2.next
            
        return pt1

"""Lowest common subsequence
    LCS is a classic problem in computer science that finds the length of the 
    longest subsequence in a string

    A subsequence is a sequence of characers in a string that come one after another but not necesarily 
    exactly in space after. Ex: in 'edgar', a subsequence is e,a,r."""

"""Solution:
https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/
https://www.youtube.com/watch?v=NnD96abizww"
This is the solution, using Dynamic Programming, which involves essentially creating a matrix with both words as memory
second row and second column (which are the first row and first column that are not headers) are filled with 0's, or in the code below, None values
n each intersection, we essentially create substrings for both words up to the characters at that point, and calculate the maximum subsequence length up to that point using values at previous character combination cells.
For better explanation, watch the video
his matrix is very similar to the 01 knapsack dynamic programming matrix, but not exactly the same"""
def lcs(X, Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
  
    # declaring the matrix for storing the dp values 
    L = [[None]*(n + 1) for i in range(m + 1)] 
  
    subsequence = ""
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: #note we dont do X[i] == Y[j] because the cell locations are offset by 1 to their respective character locations in the words due to the 0 rows. This means L[i][j] corresponds to X[i-1] and Y[j-1].
                subsequence += X[i-1] 
                L[i][j] = L[i-1][j-1]+1 #if the characters match, the value will be 1 plus the value at the left-top cell of this cell. This is because we matched a new character, which is adds + 1 to the length to whatever longest subsequence we had up to that point using the previous characters, which is the value of the top-left cell. 
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1])  #if the characters do not match, the value at that matrix coordinate is the max of either the top cell or the left cell. This is essentially the max subsequence using the characters up to that point WITHOUT INCLUDING THE CURRENT CHARACTER.
  
    # finnally, we return the very last, or bottom right value, since this will contain the max subsequence up to the ending of both strings.
    return L[m][n], subsequence

#print(lcs("edgar", "elapr"))

"""0/1 Knapsack Problem
https://www.youtube.com/watch?v=8LusJS5-AGo&t=232s
You want to fill a bag with the most value with the least weight
The 01 version of the knapsack problem means you can either pick the item or not pick it, but not split the item
If you can split the item, you just solve it using greedy, which means you sort by val/wt and pick until you reach a point where an item does not fit, so you split that last item
If you do not use dynamic programming and you just sort by val/wt, and not split the last item, it will give you a local optima solution but not a global optimal
In order to get the local optima solution, we use dynamic programming
We create a value vs weight matrix, similar to the LCS one but not exactly the same, comparing max weight and value combinations up to that point"""
"""Constraints: """
"""the weights must be sorted"""
def knapSack(W, wt, val, n): 
    K = [[0 for x in range(W + 1)] for x in range(n + 1)] 
  
    # Build table K[][] in bottom up manner 
    for i in range(n + 1): 
        for w in range(W + 1): 
            if i == 0 or w == 0: 
                K[i][w] = 0 #the first column will be 0 because w will be 0 and nothing can be put inside the back. We also make the first row to 0s for symmetry and easier handling.
            elif wt[i-1] <= w: #note since the first column and row will be 0s, the values of the matrix weights and value locations will be offset by 1 to the weight and value arrays. Thus matrix K[i][j] corresponds to wt[i-1] and val[i-1]
                K[i][w] = max(val[i-1]  + K[i-1][w-wt[i-1]],  K[i-1][w]) #here we evaluate the maximum value we can get at this specific i,j cell. We get max of the value we selected plus the value of the cell of the matrix at the remaining weight WITHOUT SELECTING THIS ITEM (previous row), vs the best value we can get without including this item (the val of this weight column at the previous value row, which is K[i-1][w])
            else: 
                K[i][w] = K[i-1][w] #if the value weighs too much, we select the best value we can get without including this item, which is the value of the previous value row at this column weight, K[i-1][w]
  
    return K[n][W] #we return the bottom-right value of the matrix, which is the maximum value we can get after going through the val and wt arrays.
  
# Driver program to test above function 
val = [60, 100, 120] 
wt = [10, 20, 30] 
W = 50
n = len(val) 
print(knapSack(W, wt, val, n)) 

"""Hamming distance between two Integers
https://www.geeksforgeeks.org/hamming-distance-between-two-integers/
Given two integers, the task is to find the hamming distance between two integers. Hamming Distance between two integers is the number of bits which are different at same position in both numbers."""
# Function to calculate hamming distance  
def hammingDistance(n1, n2) : 
    x = n1 ^ n2  
    setBits = 0
  
    while (x > 0) : 
        setBits += x & 1
        x >>= 1
      
    return setBits  

n1 = 9
n2 = 14
print(hammingDistance(9, 14)) 

"""Disjoint Set (Union Find)
https://www.geeksforgeeks.org/union-find/
https://www.youtube.com/watch?v=ID00PMy0-vE
Union-Find Algorithm can be used to check whether an undirected graph contains cycle or not.
This method assumes that the graph doesn’t contain any self-loops.
We can keep track of the subsets in a 1D array, in this case named parent[i]
NOTE: the implementation of union() and find() is naive and takes O(n) time in worst case. """

from collections import defaultdict 
   
#This class represents a undirected graph using adjacency list representation 
class Graph: 
   
    def __init__(self,vertices): 
        self.V= vertices #No. of vertices 
        self.graph = defaultdict(list) # default dictionary to store graph 
   
  
    # function to add an edge to graph 
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
   
    # A utility function to find the subset of an element i 
    def find_parent(self, parent,i): 
        if parent[i] == -1: 
            return i 
        if parent[i]!= -1: 
             return self.find_parent(parent,parent[i]) 
  
    # A utility function to do union of two subsets 
    def union(self,parent,x,y): 
        x_set = self.find_parent(parent, x) 
        y_set = self.find_parent(parent, y) 
        parent[x_set] = y_set 
  
    # The main function to check whether a given graph 
    # contains cycle or not 
    def isCyclic(self): 
          
        # Allocate memory for creating V subsets and 
        # Initialize all subsets as single element sets 
        parent = [-1]*(self.V) 
  
        # Iterate through all edges of graph, find subset of both 
        # vertices of every edge, if both subsets are same, then 
        # there is cycle in graph. 
        for i in self.graph: 
            for j in self.graph[i]: 
                x = self.find_parent(parent, i)  
                y = self.find_parent(parent, j) 
                if x == y: 
                    return True
                self.union(parent,x,y) 




