#!/usr/bin/env python3

from collections import defaultdict

class Graph():
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.num_vertices = vertices

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def dfs(self, v, visited):
        visited[v] = True
        print(v, end=" ")

        for i in self.graph[v]:
            if visited[i] == False:
                self.dfs(i, visited)

    # Do DFS traversal of adjacent to v. Then put v on stack
    def fill_order(self, v, visited, stack):
        visited[v] = True

        for i in self.graph[v]:
            if visited[i] == False:
                self.fill_order(i, visited, stack)
        stack.append(v)

    # Tranpose (reverse direction of graph)
    def get_transpose(self):
        g = Graph(self.num_vertices)

        for i in self.graph:
            for j in self.graph[i]:
                g.add_edge(j, i)
        return g

    def print_scc(self):
        stack = []

        visited = [False] * self.num_vertices

        for i in range(self.num_vertices):
            if visited[i] == False:
                self.fill_order(i, visited, stack)

        transpose = self.get_transpose()
        
        visited = [False] * self.num_vertices
        while(stack):
            v = stack.pop(0)
            if visited[v] == False:
                transpose.dfs(v, visited)
                print()

def main():
    g = Graph(5) 
    g.add_edge(1, 0) 
    g.add_edge(0, 2) 
    g.add_edge(2, 1) 
    g.add_edge(0, 3) 
    g.add_edge(3, 4) 

    print ("Following are strongly connected components in given graph") 
    g.print_scc() 

if __name__ == "__main__":
    main()