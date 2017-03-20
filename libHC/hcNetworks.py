"""
functions that create networks, measure network properties
& help generate input patterns for a given network.
CK 2014
"""
import networkx as nx
import numpy as np
import hcPlotting as plo
import hcUtil as ut
from numpy.random import RandomState
import networkx as nx
# creating networks:

def addlabels(M,N,graph,strength):
    """
    Assign a "grid position" label to each node of the given graph
    & set the input + edge strength attributes

    Args:
        M,N: desired grid dimensions
        graph: a networkx graph
        strength: desired edge strength

    Returns:
        the labeled graph
    """
    oldlabels = graph.nodes()
    newlabels = [(i,j) for j in range(N) for i in range(M)]
    graph = nx.relabel_nodes(graph,dict(zip(oldlabels,newlabels)))
    for n in graph.nodes_iter():
        graph.node[n] = {"input":0}
    for e in graph.edges_iter():
        graph.edge[e[0]][e[1]] = {"strength":strength}
        graph.edge[e[1]][e[0]] = {"strength":strength}
    return graph

# def add_pairwise_inhibition(graph, inputc=None, inh_strength=-1, exc_strength=1, input_to_inh=False):
#     """
#     add an inhibitory interneuron to each cell in the given network,
#     projecting back to its excitatory source neuron

#     Internally, this turns an MxN neuron grid into a M*2xN grid,
#     where odd rows are inhibitory:

#                                     --N--
#                                 e..eee.e.e.e.
#            --N--                i..iii.i.i.i.
#         e..eee.e.e.e            ..e.e.eee.e..
#      |  ..e.e.eee.e..           ..i.i.iii.i..   |
#      M  ee.e.ee.e...e    --->   ee.e.ee.e...e  2*M
#      |  e.e.eee.e..e.           ii.i.ii.i...i   |
#                                 e.e.eee.e..e.
#                                 i.i.iii.i..i.

#     this function also can also adjust inputmasks, so inputs go to the same
#     excitatory cells as before.

#     """
#     M,N = graph.graph['grid_dimensions']
#     new = nx.DiGraph(grid_dimensions=(M*2,N))

#     was_undirected = not isinstance(graph, nx.DiGraph)

#     for i in range(M*2):
#         for j in range(N):
#             attr = {'is_inhibitory': i%2 == 1}
#             new.add_node((i, j), attr_dict=attr)

#     # transfer existing excitatory connections
#     for i in range(M):
#         for j in range(N):
#             for target, attr in graph[(i,j)].items():
#                 u,v = target
#                 src, trg = ((i*2, j), (u*2, v))
#                 new.add_edge(src, trg, attr_dict=attr)
#                 if was_undirected:
#                     new.add_edge(trg, src, attr_dict=attr)
#     # add reciprocal exc/inh connections
#     for i in range(0, M*2-1, 2):
#         for j in range(N):
#             new.add_edge((i,j), (i+1,j), {'strength':exc_strength})
#             new.add_edge((i+1, j), (i,j), {'strength':-abs(inh_strength)})

#     # spread out inputmask to every other line, as well
#     if inputc is not None:
#         M,N = inputc.shape
#         new_inputc = np.zeros((M*2, N))
#         for i in range(M):
#             new_inputc[i*2,:] = inputc[i,:]
#             if input_to_inh and i*2<M:
#                 new_inputc[i*2+1,:] = 0.5*inputc[i,:]

#     return new, new_inputc, M*2, N


def add_neighbouring_inhibition(graph, inputc=None, inh_recurrent=-1, inh_lateral=-1, exc_strength=1, input_to_inh=False):
    """
    add an inhibitory interneuron to each excitatory cell in the given network,
    projecting back recurrently and laterally to nearby excitatory cells.

    Internally, this turns an MxN neuron grid into a M*2xN grid,
    where odd rows are inhibitory:

                                    --N--
                                e..eee.e.e.e.
           --N--                i..iii.i.i.i.
        e..eee.e.e.e            ..e.e.eee.e..
     |  ..e.e.eee.e..           ..i.i.iii.i..   |
     M  ee.e.ee.e...e    --->   ee.e.ee.e...e  2*M
     |  e.e.eee.e..e.           ii.i.ii.i...i   |
                                e.e.eee.e..e.
                                i.i.iii.i..i.

    this function also also adjusts inputmasks, so inputs go to the same
    excitatory cells as before.

    """
    M,N = graph.graph['grid_dimensions']
    new = nx.DiGraph(grid_dimensions=(M*2,N))

    was_undirected = not isinstance(graph, nx.DiGraph)

    for i in range(M*2):
        for j in range(N):
            attr = {'is_inhibitory': i%2 == 1}
            new.add_node((i, j), attr_dict=attr)

    # transfer existing excitatory connections
    for i in range(M):
        for j in range(N):
            for target, attr in graph[(i,j)].items():
                u,v = target
                src, trg = ((i*2, j), (u*2, v))
                new.add_edge(src, trg, attr_dict=attr)
                if was_undirected:
                    new.add_edge(trg, src, attr_dict=attr)

    # add nearby exc/inh connections
    for i in range(0, M*2-1, 2):
        for j in range(N):
            new.add_edge((i,j), (i+1,j), {'strength':exc_strength})

            # back to source neuron:
            new.add_edge((i+1, j), (i,j), {'strength':-abs(inh_recurrent)})

            # to source neuron's neighbours left, right, above, below:
            neighbours = [(i,j-1), (i,j+1), (i-2,j), (i+2,j)]
            for n in neighbours:
                if n in new.node: # if neighbour coordinates exists
                    new.add_edge((i+1, j), n, {'strength':-abs(inh_lateral)})


    # spread out inputmask to every other line, as well
    if inputc is not None:
        M,N = inputc.shape
        new_inputc = np.zeros((M*2, N))
        for i in range(M):
            new_inputc[i*2,:] = inputc[i,:]
            if input_to_inh and i*2<M:
                new_inputc[i*2+1,:] = 0.5*inputc[i,:]

    return new, new_inputc, M*2, N


def grid_empty(M,N):
    """
    return an MxN grid graph without edges
    (it differs from other graphs with N*M nodes only by the i,j node labels)

    Args:
        M,N: desired grid dimensions
    """
    g = nx.Graph(grid_dimensions=(M,N))
    g.add_nodes_from([((i,j),{"input":0}) for j in range(N) for i in range(M)])

    return g


def grid_fourNN(M,N,strength=1.0):
    """
    return an MxN grid graph with 4-nearest-neighbour connectivity

    Args:
        M,N: desired grid dimensions
        strength: desired edge strength
    """
    graph = nx.grid_2d_graph(M, N)
    graph.graph["grid_dimensions"] = (M,N)
    for e in graph.edges_iter():
        graph.edge[e[0]][e[1]] = {"strength":strength}
        graph.edge[e[1]][e[0]] = {"strength":strength}
    return graph


def grid_eightNN(M,N,strength=1.0,islesioned=lambda a,b:False):
    """
    Returns an MxN grid graph with 8-nearest-neighbour-connectivity.
    Optionally, lesions (cuts) can be introduced.

    Args:
        M,N: desired grid dimensions
        strength: desired edge strength
        islesioned: function f(nodeA, nodeB) -> True if the connection
                    from node A to B should be cut, -> False otherwise.
    """
    def blacklisted(origin,to):
        """ some rules which nodes should never connect."""
        outofbound = to[0] < 0 or to[0] > M-1 or to[1] < 0 or to[1] > N-1
        #print origin,to,islesioned(origin,to),islesioned(to,origin)
        unwanted = origin == to or islesioned(origin,to) or islesioned(to,origin)
        return outofbound or unwanted

    g = nx.Graph(grid_dimensions=(M,N))
    g.add_nodes_from([((i,j),{"input":0}) for j in range(N) for i in range(M)])
    neighbours = []
    for i in range(M):
        for j in range(N):
            neighbours = [((i,j),(i+k,j+l),{"strength":strength})
                          for k in [-1,0,1] for l in [-1,0,1] if not blacklisted((i,j),(i+k,j+l))]
            g.add_edges_from(neighbours)
    return g

def grid_random(M,N,edgecount,strength=1.0,seed=1):
    """
    Returns a random graph, with nodes named after MxN grid coordinates.

    Args:
        M,N: desired grid dimensions
        edgecount: number of edges in the graph
        strength: desired edge strength
        seed: random seed
    """
    g = nx.gnm_random_graph(M*N,edgecount,seed)
    g.graph["grid_dimensions"] = (M,N)
    return addlabels(M,N,g,strength)

def grid_smallworld(M,N,nneigh,rewire,tries=100,strength=1.0,seed=1):
    """
    Returns a connected Watts-Strogatz small-world graph, with nodes
    named after MxN grid coordinates.

    Args:
        M,N: desired grid dimensions
        strength: desired edge strength
        nneigh: connect to ~ nearest neighbours
        rewire: rewiring probability
        tries: nr. of attempts to generate a connected graph
        seed: random seed
    """
    g = nx.connected_watts_strogatz_graph(M*N,nneigh,rewire,tries,seed=seed)
    g.graph["grid_dimensions"] = (M,N)
    return addlabels(M,N,g,strength)

def grid_powerlaw(M,N,nedges,ptriangle,strength=1.0,seed=1):
    """
    Returns a graph with powerlaw degree distribution [P. Holme and B. J. Kim 2002]
    with nodes named after MxN grid coordinates

    Args:
        M,N: desired grid dimensions
        nedges: number of edges per node
        ptriangle: probability of inserting a triangle
        strength: desired edge strength
        seed: random seed
    """
    g = nx.powerlaw_cluster_graph(M*N,nedges,ptriangle,seed=seed)
    g.graph["grid_dimensions"] = (M,N)
    return addlabels(M,N,g,strength)

def grid_spatial(M,N,strength=1,a=0.5,b=1,c=0,seed=1):
    """
    Returns a grid graph with random, distance-dependent connectivity

    connection probability between nodes n1 and n2 is max(1/b*d(n1,n2) - c,0)

    Args:
        M,N: desired grid dimensions
        strength: desired edge strength
        b: connection probability: distance weighting factor
        c: connection probability: constant offset
        seed: random seed
    """
    rng = RandomState(seed)
    p_connect = lambda n1,n2: max(1.0/(b*pow(((n1[0]-n2[0])**2 + (n1[1]-n2[1])**2),a))-c,0) if n1!=n2 else 0
    connect = lambda n1,n2: rng.rand()<=p_connect(n1,n2) if n1 != n2 else False

    g = nx.Graph(grid_dimensions=(M,N))
    g.add_nodes_from([((i,j),{"input":0}) for j in range(N) for i in range(M)])
    neighbours = []
    for i in range(M):
        for j in range(N):
            neighbours = [((i,j),(k,l),{"strength":strength})
                          for k in range(M) for l in range(N) if connect((i,j),(k,l))]
            g.add_edges_from(neighbours)
            #debug:
            # plt.clf()
            # plt.imshow(np.array([[p_connect((i,j),(k,l)) for k in range(M)] for l in range(N)]),
            #            interpolation='nearest',vmin=0,vmax=1);plt.colorbar()
            # plt.draw()
            # from time import sleep
            # sleep(0.01) # import pdb; pdb.set_trace()
    return g



# measuring stuff:

def inputASL(network,inputc):
    """
    Returns the average shortest path length within the input-receiving subgraph.

    Args:
        network: networkx graph
        inputc: MxN array, all nonzero positions are treated as 'input receiving'
    """
    inputnodes = [tuple(ind) for ind in np.transpose(inputc.nonzero())]
    lengths = [nx.shortest_path_length(network,src,trg) for src in inputnodes for trg in inputnodes]
    return np.mean(lengths)

def distances_to_roi(network,inputc,roi):
    """
    Returns a list of shortest path lengths from each
    input-receiving cell to all measured cells.

    Args:
        network: networkx graph
        inputc: MxN array, nonzero positions are treated as 'input receiving'
        inputc: MxN array, nonzero positions are treated as 'measured'
    """
    inputnodes = [tuple(ind) for ind in np.transpose(inputc.nonzero())]
    roinodes   = [tuple(ind) for ind in np.transpose(roi.nonzero())]
    lengths = [nx.shortest_path_length(network,src,trg) for src in inputnodes for trg in roinodes]
    return lengths

def resistancedistances(graph):
    """
    Returns the pairwise resistance distances on the given graph.

    Args:
        network: networkx graph

    Returns:
        Dictionary of pairwise resistance distances,
        accessed by the (i,j) node labels
    """
    nodes = graph.nodes()
    nodecount = len(nodes)
    nodenrs = range(nodecount)
    labeling = dict(zip(nodenrs,graph.nodes()))
    L = np.linalg.pinv(nx.laplacian_matrix(graph))
    rdist = {}
    for i in nodenrs:
        rdist[labeling[i]] = {}
        for j in nodenrs:
            rdist[labeling[i]][labeling[j]] = L[i,i] + L[j,j] - L[i,j] - L[j,i]
    return rdist


# finding 'recording sites' & stimuli for given network:

def createroiidxs(network,distance):
    """
    Choose two central nodes, some distance apart, and return their (i,j) indices.

    Args:
        network: networkx graph
        distance: how far apart the two nodes should be.

    Returns:
        A tuple of two (i,j) indices / node labels
    """
    nodes,centralities = zip(*nx.closeness_centrality(network).items())
    # sort nodes from most central to least central:
    centr_arxs = np.argsort(centralities)
    nodes_sorted = [n for n in reversed(np.array(nodes)[centr_arxs])]
    k = 0
    while k<len(nodes_sorted):
        # pick some node in the middle of the graph (high centrality)
        middlenode = tuple(nodes_sorted[k])
        # now pick the most central node that meets the given distance criterion.
        # [since we dont want to end up near the boundaries)
        for n in nodes_sorted:
            if nx.shortest_path_length(network,middlenode,tuple(n)) == distance:
                return middlenode,tuple(n)
        # if that didnt work, try starting with a different, less central middlenode.
        k = k+1
    raise Exception("speficied distance to high for this network")

def createroi(M,N,network,distance):
    """
    Chose two nodes, some distance apart, and returns an image

    Args:
        network: networkx graph
        distance: how far apart the two nodes should be.

    Returns:
        MxN array of zeros, with the two chosen coordinates := 1

    """
    nodes = createroiidxs(network,distance)
    return ut.idxtoimg(M,N,nodes)


def createstimulusidxs(network,roi,K,scatter=0,method="mean_shortest",cutoff=8):
    """
    Choose a more or less scattered subpopulation of size K on the given network,
    return its indices.

    Args:
        network: networkx graph
        roi: MxN array, nonzero positions are treated as measured nodes
        K: size of the subpopulation to pick
        scatter: how dense or scattered the population should be.
                 suitable values depend on method.
        method: How to pick nodes.
            "bridge_roi":   pick nodes on the shortest paths between the 2 roi cells,
                            or on slightly longer such paths, depending on scatter.
                            0<scatter<1

            "mean_shortest": greedily pick nodes closest (2nd closest, 3rd closest,
                             scatter'th closest..) to the nodes picked so far. "close"
                             means low average shortest path length.
            "mean_rdist":    ..."close" means low average resistance distance.
            "norm_shortest": ..."close" means low mean squared distances
        cutoff: determines the longest paths considered in method 'bridge_roi'
    Returns:
        List of indices / node labels
    """
    roinodes  = [tuple(ind) for ind in np.transpose(roi.nonzero())]
    inputnodes = roinodes[:]
    k = len(roinodes)
    allnodes = list(set(network.nodes()).difference(inputnodes)) #surprisingly the most efficient way
    if method == "bridge_roi":
        assert len(roinodes) == 2 # todo: instead just pick 2.
        assert scatter >= 0 and scatter <= 1 # this method needs a normalized scatter value.
        roiA,roiB = roinodes
        # get all paths between the two roi cells, sorted by length:
        # that can take really long, so at least we store the result for next time.
        if not hasattr(network,"allpaths"):
            network.allpaths = {}
        if not network.allpaths.has_key((cutoff,tuple(roinodes))):
            print "create stimulus: calc. paths"
            paths_roi = [p for p in nx.all_simple_paths(network,roiA,roiB,cutoff=cutoff)]
            if len(paths_roi) < 2:
                raise Exception("found only 1 path between ROIs")
            network.allpaths[(cutoff,tuple(roinodes))] = paths_roi
            print "."
        else:
            paths_roi = network.allpaths[(cutoff,tuple(roinodes))]
        lpaths = [len(p) for p in paths_roi]
        paths_roi = np.array(paths_roi)[np.argsort(lpaths)]
        len_paths_roi = len(paths_roi)
        relax_p = 0
        while k<K:
            # pick a path between roi A and roi B -- a short one
            # if scatter and relax_p are low, and the next longer path
            # as relax_p grows when more nodes are needed (K>len(path)):
            pathidx = int(scatter*len_paths_roi)+relax_p
            try:
                path = paths_roi[pathidx]
            except IndexError:
                print "ran out of paths to try. try smaller scatter value & larger cutoff"
                inputnodes = allnodes
                break

            len_path = len(path)
            # pick nodes on that path, starting in the middle and
            # spreading outwards left & right, up until the ends
            # of the path or until we have enough nodes (K).
            # for scatter > 0, start a bit off the middle node and
            # thus jump out to longer paths a bit earlier:
            middle = int(len_path/2)
            relax_n = int(scatter*middle) # so, for scatter 0: 0, for scatter 1: distance from one end to the middle
            reached_L,reached_R = (False,False)
            while k<K and not (reached_L and reached_R):
                nodeidx = middle + relax_n
                relax_n = -relax_n if relax_n > 0 else -relax_n + 1 # 1,-1,2,-2,3,-3,4,-4,.....
                if nodeidx <= 0:
                    nodeidx = 0
                    reached_L = True
                if nodeidx >= len_path-1:
                    nodeidx = len_path-1
                    reached_R = True

                node = path[nodeidx]
                if not node in inputnodes:
                    inputnodes.append(node)
                    k = k+1
                ## debug:
                # plt.clf()
                # plo.plotsetup(network,ut.idxtoimg(10,10,inputnodes),roi,axes=plt.gca(),grid_as="graph")
                # plt.draw()
                # import pdb; pdb.set_trace()

            relax_p = relax_p + 1
    else:
        while k<K:
            distances = []
            # find the mean distance of each node to all nodes so far receiving input
            for v in allnodes:
                if method=="norm_shortest":
                    distances.append(np.linalg.norm([nx.shortest_path_length(network,v,vi)
                                                     for vi in inputnodes]))
                elif method=="mean_shortest":
                    distances.append(np.mean([nx.shortest_path_length(network,v,vi)
                                              for vi in inputnodes]))
                elif method=="mean_rdist":
                    if network.graph.has_key("resistancedistances"):
                        rd = network.graph["resistancedistances"]
                    else:
                        rd = resistancedistances(network)
                        network.graph["resistancedistances"] = rd
                    distances.append(np.mean([rd[v][vi] for vi in inputnodes]))
                else:
                    raise Exception("wrong option")
            arxs = np.argsort(distances)
            # take the closest node (scatter=0) or n'th-closest (scatter=n)
            scatter_ = min(scatter,len(arxs)-1)
            bestnode = allnodes[arxs[scatter_]]

            allnodes.remove(bestnode)
            inputnodes.append(bestnode)
            k = k+1
            ## debug
            # plt.clf()
            # plo.plotsetup(network,ut.idxtoimg(20,20,inputnodes),roi,axes=plt.gca(),grid_as="graph")
            # plt.draw()
            # import pdb; pdb.set_trace()
            #
    return inputnodes

def createstimulus(M,N,network,roi,K,scatter=0,method="mean_shortest",cutoff=8):
    """
    Choose a more or less scattered subpopulation of size K on the given network,
    return an image.

    Args:
        network: networkx graph
        roi: MxN array, nonzero positions are treated as measured nodes
        K: size of the subpopulation to pick
        scatter: how dense or scattered the population should be.
                 suitable values depend on method.
        method: How to pick nodes.
            "bridge_roi":   pick nodes on the shortest paths between the 2 roi cells,
                            or on slightly longer such paths, depending on scatter.
                            0<scatter<1

            "mean_shortest": greedily pick nodes closest (2nd closest, 3rd closest,
                             scatter'th closest..) to the nodes picked so far. "close"
                             means low average shortest path length.
            "mean_rdist":    ..."close" means low average resistance distance.
            "norm_shortest": ..."close" means low mean squared distances
        cutoff: determines the longest paths considered in method 'bridge_roi'
    Returns:
        An MxN array of zeros, with the chosen subpopulation := 1
    """
    nodes = createstimulusidxs(network,roi,K,scatter,method,cutoff)
    return ut.idxtoimg(M,N,nodes)


def diffuse_stimulus(network,inputc,dist,seed=1):
    """
    move each input-receiving node along a random
    walk of specified distance.

    Args:
        network: networkx graph
        inputc: MxN array, nonzero positions are treated as
                input-receiving ndes
        dist: distance

    Returns:
       An MxN array of zeros, with the scrambled input population := 1
    """
    rng = RandomState(seed)
    inputnodes = [tuple(n) for n in np.transpose(inputc.nonzero())]
    diffused_nodes = []
    for node in inputnodes:
        d=0
        while d<dist:
            neigh = network.neighbors(node)
            node_ = neigh[rng.randint(0,len(neigh))]
            d = d+1
        diffused_nodes.append(node_)
    return ut.idxtoimg(inputc.shape[0],inputc.shape[1],diffused_nodes)




def scramble_stimulus(network,inputc,sct,seed=1):
    """
    with some probability, move each input-receiving node along a random
    walk of specified distance.

    Args:
        network: networkx graph
        inputc: MxN array, nonzero positions are treated as
                input-receiving ndes
        sct: scattering distance

    Returns:
       An MxN array of zeros, with the scrambled input population := 1
    """
    rng = RandomState(seed)
    inputnodes = [tuple(n) for n in np.transpose(inputc.nonzero())]
    prob = sct/float(len(inputnodes))
    shouldscramble = dict([(n,rng.uniform()<prob) for n in inputnodes])
    in_scrambled = []
    for node in inputnodes:
        d=0
        if shouldscramble[node]:
            while d<sct or node in inputnodes:
                neigh = network.neighbors(node)
                node = neigh[rng.randint(0,len(neigh))]
                d = d+1
        in_scrambled.append(node)
    return ut.idxtoimg(inputc.shape[0],inputc.shape[1],in_scrambled)






# 3D networks were added as an afterthought - so here are some horrible converters to
# get them back to the common grid format used throughout the simulation code.

def convert_network_3Dto2D(network):
    """
    Reshape a 3D grid graph into a 2D one, preserving
    adjacency and edge&node attributes.
    """
    assert network.graph.has_key("grid3D_dimensions")
    M,N,K = network.graph["grid3D_dimensions"]

    reshape = lambda node: (node[0],node[1] + node[2]*N)

    nodes_2D = [(reshape(node_attr[0]),node_attr[1]) for node_attr in network.node.items()]

    edges_3D = nx.to_edgelist(network)
    edges_2D = [(reshape(e[0]),reshape(e[1]),e[2]) for e in edges_3D]

    g = nx.from_edgelist(edges_2D)
    g.add_nodes_from(nodes_2D)
    g.graph["grid_dimensions"] = (M,N*K)
    g.graph["original_grid3D_dimensions"] = (M,N,K)
    return g


def extract_input_grid(network):
    """
    Returns an array of input strengths of the size of the graph's grid,
    based on the 'input' node attribute.
    """
    inputc = np.zeros(network.graph["grid_dimensions"])
    for node_attr in network.node.items():
        inputc[node_attr[0]] = node_attr[1]["input"]
    return inputc
