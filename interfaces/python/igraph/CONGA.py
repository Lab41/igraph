import numpy as np
import collections as co
import igraph as ig
import operator

G = ig.read("sampleGraphs/football.gml")
OG = ig.read("sampleGraphs/football.gml")

#todo: look at vertex cover





def CONGA(G):
    
    # We want to take the original graph and store the original ids. we are going to change them
    G.vs['orig'] = G.vs['id']

    # initialize the modularity
    maxMod = 0


    splits = []
    
    # run this loop, at worst, until we have no edges left
    while G.es:


        # calculate edge betweenness list
        eb = G.edge_betweenness()
        
        max_index, max_eb = max(enumerate(eb), key=operator.itemgetter(1))
        # TODO we think we can caluculate vertex betweenness and edge betweenness at the same time
        
        # calculate vertex betweenness and store it
        vb = G.betweenness()
        
        # make a list of verticies where the vertex betweenness is greater than the max
        # edge betweenness. This comes from Gregory 2007 step 3
        vi = [vb.index(i) for i in vb if i >= max_eb]
        
        # find the position within eb of the max, and then grab the corresponding edge from
        # the edge list

        edge = G.es[max_index].tuple
        
        # Check to see if we have any verticies of interest. If not, then remove the edge with
        # the max edge betweenness. Else continue with calculating pair and split betweenness
        if not vi:

            delete_edge(G, edge, splits)


        else:
                
            # Calculate pair betweenness. This is a nested dictionary where the first key is
            # the vertex (v). The next key is a tuple of every permutation v's neighbors (u,w).
            # the value is a pair betweenness score, discussed below
            dic = pair_betweenness(vi)
            
            # calculate max split betweenness. This returns the split betweenness score, vertex id,
            # and a list of how the vertex should be split
            vMax,vNum,vSpl = max_split_betweenness(dic)
            
            # if the split betweenness is greater than the max edge betweenness, then we want
            # to make the split at the spot identified by the split betweenenss function above
            # else remove the maximum edge
            if vMax > max_eb:
                splitVertex(vNum,vSpl, splits)
            else:
                delete_edge(G, edge, splits)
        
        # get current communities
        comm = G.components().membership        
        
        # set the current modularity
        curMod = G.modularity(comm)

        # TODO rewrite: if our modularity starts to drop, we want to stop. The edge removal that
        # started decreasing modularity has already happened, so we want to take
        # step back in time by adding the edge back in and returning that community
        # structure
        if curMod>maxMod:
            maxMod = curMod
            mOrig = G.vs['label']
            mComm = comm
            print maxMod
            print mOrig
            print mComm

    return mOrig,mComm


def default_factory():
    return 1.


def delete_edge(G, edge, splits):
    """ 
    Given a graph G and one of its edges in tuple form, checks if the deletion 
    splits the graph. If so, it adds the edge to the splitlist.
    """
    G.delete_edges(edge)

    # if the graph has been split by this deletion
    check_for_split(splits, edge)


def check_for_split(splits, edge):
    if not G.edge_disjoint_paths(source=edge[0], target=edge[1]):
        splits += [list(edge)]


def nepusz_modularity(G):
    pass




def pair_betweenness(arr):
    # we pass in an array of verticies of interest, defined as those which
    # the vertex betweenness is greater than the max edge betweenness

    # initialize a new dictionary
    dic = {}
    
    # for each vertex of interest, we want to create a list of all neighbors.
    # then within the neighbors, we want to get all permutations of those neighbors.
    # the structure of this resulting dictionary is dic[v][(u,w)] = 0, where
    # v,u,w are defined from the Gregory CONGA paper
    
    # TODO: look into vertex self loops
    
    for vertex in arr:
        dic[vertex] = {}
        neighbors = G.neighbors(vertex)
        
        # TODO: remove vertex from neighbors if exists     
        
        for u in neighbors:
            for w in neighbors:
                if u != w:
                    dic[vertex][(u,w)] = 0
    # initialize scores dictionary
    scores = {}
    
    # for each vertex in the entire graph, we want to start counting the number of
    # ALL shortest paths. We only are interested in the paths with a length greater than
    # three, so we can account for u,v, and w. For every path, we want make a tuple of the
    # beginning (a) and the ending (b) of the path. This keys the dictionary so we can see
    # how many shortest paths traverse from a to b. We want to weight the score at the next
    # step with the inverse of this count, so we don't over weight traversals over the same
    # path
 
    n = G.vcount()
    for vertex in range(n):
        
        # initialize a default dict to 1.
        counts = co.defaultdict(default_factory)
        allPaths = G.get_all_shortest_paths(vertex)
        for path in allPaths:
            if len(path) >= 3:
                key = (path[0],path[len(path)-1])

                for k in counts:
                    counts[k] += 1
 
        # once we have our path list, we still want to make sure the length is greater
        # than 2. We then want to create a key which is a tuple of (u,v,w). We can then
        # use this tuple to look up the corresponding counts, invert them, and increment
        # the score by that amount
        for path in allPaths:
            if len(path) >= 3:
                n = len(path)
                i = 0
                key = (path[0],path[len(path)-1])
                while n >= i+3:
                    k = (path[i],path[i+1],path[i+2])
                    if k not in scores:
                        scores[k] = 1 / counts[key]
                    else:
                        scores[k] += 1 / counts[key]
                    i += 1
            
    # finally for this function, we open up the dictionary for each vertex of interest
    # which was initialized above to 0, find the corresponding score for that path
    # and add it to the previous value. This is to create a weighted score for every
    # vertex of interest (v) where the score is constructed by the number of traversals
    # for every shortest path in the network over every neighbor pair (u,w) of v.
    for v in dic.keys():
        for pair in dic[v].keys():
            k = (pair[0],v,pair[1])
            if k in scores.keys():
                dic[v][pair] += scores[k]
    return dic
   

def max_split_betweenness(dic):
    
    # initialize a max score to 0 (it won't be, but we need a comparison state)
    vMax = 0
    
    # for every vertex of interest, we want to figure out the maximum score achievable
    # by splitting the verticies in various ways, and return that optimal split
    for v in dic:
        
        # get a list of neighbors for v
        neigh = G.neighbors(v)
        
        # initialize a list on how we will map the neighbors to the collapsing matrix
        vMap = []
        
        # copy over the neighbor list to vMap by first wrapping each vertex id into
        # its own list. This will make sense in a sec
        for e in neigh:
            vMap.append([e])
        
        # create a matrix M where the rows and columns headers are neighbor vertex ids
        # and the cells are the scores between the corresponding vertex ids. This has a
        # diag of 0
        M = createMatrix(v,dic[v])
        
        # we want to keep collapsing the matrix until we have a 2x2 matrix and its
        # score. Then we want to remove index j from our vMap list and concatenate
        # it with the vMap[i]. This begins building a way of keeping track of how
        # we are splitting the vertex and its neighbors
        while M.size > 4:
            i,j,M = reduceMatrix(M)
            vMap[i] += vMap.pop(j)
        
        # we keep iterating over all of the verticies of interest until we find the max
        # upon finding the max, we can then save out the vertex id, the score, and the 
        # optimal split
        if M[0,1] > vMax:
            vMax = M[0,1]
            vNum = v
            vSpl = vMap
    return vMax,vNum,vSpl


def matMin(M):
    
    # take a matrix we pass in, and fill the diagonal with the matrix max. This is
    # so that we don't grab any values from the diag.
    np.fill_diagonal(M,np.max(M))
    
    # figure out the indicies of the cell with the lowest value. This is where we
    # will start collapsing the matrix
    i,j = np.unravel_index(M.argmin(), M.shape)
    
    # refill the diagonal with 0's so its returned as we found it
    np.fill_diagonal(M,0)
    
    # return the index where the minimum occurs
    return i,j


# Neo, NOOOOOOO!
def createMatrix(v,subDic):
    # we take in a vertex and a sub dictionary of its neighbor pairs with the scores
    # of the u,v,w path    
    
    # generate a list of neighbors
    neigh = G.neighbors(v)
    
    # get the len of the neighbors to setup size of the new matrix
    n = len(neigh)
    
    # generate the square (nxn) matrix and initialize it with zeros
    M = np.matrix(np.zeros(n**2).reshape((n,n)))
    
    # for each pair in our dictionary, we want to get the indicies. Then we want to get
    # the value and set that cell to it.
    for pair in subDic.keys():
        i = neigh.index(pair[0])
        j = neigh.index(pair[1])
        value = subDic[pair]
        M[i,j] = value
    # We had an issue earlier where u,v,w was counted differently from w,v,u. We can transpose
    # the lower triangle and add it to the upper triangle to fix this.
    M = np.transpose(np.tril(M)) + np.triu(M)
    
    # return the above matrix plus its transpose to get a symmetric score matrix
    return M + np.transpose(M)


def reduceMatrix(M):
    # How can you call for help when you are no longer able to.... speak    
    
    # pass in the matrix and get a minimum value
    i,j = matMin(M)
    
    # add the ith row to the jth row and over write the ith row with those values
    M[i,:] = M[j,:] + M[i,:]
    
    # delete the jth row
    M = np.delete(M, (j), axis=0)
    
    # add the ith column to the jth column and overwrite the ith column with result
    M[:,i] = M[:,j] + M[:,i]
    
    # delete jth column
    M = np.delete(M, (j), axis=1)
    
    # make damn sure our matrix has a diag of 0
    np.fill_diagonal(M,0)
    
    # return the indicies we just collapsed and the resulting matrix
    return i,j,M


def splitVertex(v,s, splits):
    # CHECK might be broken.
    # this is where the new vertex is going to go
    new_index = len(G.vs['id'])
    
    # add the new vertex and preserve the label and the original name
    G.add_vertex()
    G.vs[new_index]['label'] = label = G.vs[v]['label']
    G.vs[new_index]['orig'] = name = G.vs[v]['orig']
    
    # adding all relevant edges to new vertex, deleting
    # from old one.

    print set(G.neighborhood(vertices=v)) == set(s[0] + s[1] + [v])

    for partner in s[0]:
        # TODO: use zip or something with add_edges
        G.add_edge(new_index, partner)
        G.delete_edges((v, partner)) 

    # TODO NEXT STEP keep record of what vertices split into. Perhaps a dict.

    # check if the two new vertices are disconnected.
    check_for_split(splits, (v, new_index))




    
    
def matcher(orig,comm):
    # TODO make this output in the format we want.

    # get a list of original ids
    oList = orig
    
    # get a list of the membership
    cList = comm
    
    # initialize a dictionary to output a ids and their communities
    outList = {}
    
    # initialize a counter
    i = 0
    
    # for each vertex in the oList we want to create a list of assigned communities
    for v in oList:
        if v not in outList:
            outList[v] = [cList[i]]
        else:
            outList[v].append(cList[i])
        i+=1
    return outList


a,b = CONGA(G)
    