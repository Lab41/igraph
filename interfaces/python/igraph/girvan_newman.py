import igraph as ig
import numpy
import sys


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

	G = origGraph.copy() 

	while G.es:

		# Calculate all edge betweennesses
		# TODO: only recalculate on relevant portions
		edge_betweennesses = G.edge_betweenness()

		# returns an arbitrary index if there is a tie at max.
		# TODO: find which index argmax actually returns.
		max_index = numpy.argmax(edge_betweennesses) #check if numpy copies array

		# edge with the max betweenness
		edge = G.es[max_index].tuple

		G.delete_edges(edge)

		if splitGraph(G, edge):

			# edge is a tuple, but we want a list of lists.
			splits += [list(edge)]

	vd = createDendrogram(origGraph, splits)

	vd.optimal_count # TODO: make bug report

	return vd


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

	mergeDict = {}

	while splits:
		# most recent split popped off
		edge = splits.pop()

		edge = findMergedIds(edge, mergeDict)

		merges += [edge]
		
		# since we have merged 2 vertices, we have to replace
		# all occurences of those vertices with the new 
		# "merged" index n.
		for vertex in edge:
			mergeDict[vertex] = n
	#	splits = replaceOccurences(splits, n, edge)
		
		n += 1

	return ig.VertexDendrogram(G, merges)



def findMergedIds(edge, mergeDict):
	return [traverse(vertex, mergeDict) for vertex in edge]

def traverse(vertex, mergeDict):
	while vertex in mergeDict:
		vertex = mergeDict[vertex]
	return vertex



if __name__ == "__main__":
	G = ig.load(sys.argv[1])
	gn(G)