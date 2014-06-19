import igraph as ig
import numpy
import sys
from copy import deepcopy



def gn(origGraph):
	"""
	Parameters:
		origGraph: a graph in igraph format

	Return value:
		A dendrogram (VertexDendrogram) created by running Girvan-Newman

	Notes: 
		Runs the Girvan-Newman (edge-betweenness) algorithm on the graph provided.
		Iteratively removes the edge with the highest edge-betweenness, then recalculates.
	"""
	
	# initialize a list of removed edges that result in a split of the graph
	splits = []

	G = deepcopy(origGraph)

	while G.es:

		# Calculate all edge betweennesses
		# TODO: only recalculate on relevant portions
		edge_betweennesses = G.edge_betweenness()

		# returns an arbitrary index if there is a tie at max.
		# TODO: find which index argmax actually returns.
		max_index = numpy.argmax(edge_betweennesses)

		# edge with the max betweenness
		edge = G.get_edgelist()[max_index]

		G.delete_edges(edge)

		if splitGraph(G, edge):

			# edge is a tuple, but we want a list of lists.
			splits += [list(edge)]

	return createDendrogram(origGraph, splits)


def getOptimalClustering(dendro):
	"""
	Given a VertexDendrogram, returns the optimal VertexClustering
	(calculated from modularity). 
	"""
	# calculates the modularities of all clusters 
	# and chooses the optimal one. Don't be fooled
	# by the lack of parentheses, it calls a method
	# if unassigned!
	dendro.optimal_count

	return dendro.as_clustering()

def splitGraph(G, edge):
	""" 
	Parameters:
		G: an igraph graph
		edge: an edge of the form (v1, v2) where v1 and v2 are vertices in G.
	
	Return value:
		A boolean value. True if removing the edge splits the graph.
	
	Notes:
		Checks to see if removing edge from G splits the graph into 2 disjoint
	communities. If so, returns True, otherwise False.
	"""
	return not G.edge_disjoint_paths(source=edge[0], target=edge[1])


def createDendrogram(G, splits):
	"""
	Given a historical list of split edges, creates a dendrogram 
	by calculating the merges. 

	Unfortunately, runs in O(n^2). TODO: think about another algorithm
	(perhaps a tree approach?) that does better. This is a useful function
	for any divisive algorithm for which splits can be saved more easily
	than merges.
	"""

	# To create a dendrogram, new merges have id of max id + 1
	n = len(splits) + 1
	merges = []
	while splits:
		# most recent split popped off
		edge = splits.pop()

		merges += [edge]
		
		# since we have merged 2 vertices, we have to replace
		# all occurences of those vertices with the new 
		# "merged" index n.
		splits = replaceOccurences(splits, n, edge)
		
		n += 1

	return ig.VertexDendrogram(G, merges)



def replaceOccurences(splits, n, edge):
	"""
	Given a 2d list `splits`, replaces all occurences of elements in
	`edge` with n.
	"""
	for i in range(len(splits)):
		for j in range(2):
			if splits[i][j] in edge:
				splits[i][j] = n
	return splits



if __name__ == "__main__":
	G = ig.load(sys.argv[1])
	gn(G)