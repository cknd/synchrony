"""
Samples random spatial networks with strong links clustered in certain random input patterns.
Then, tests synchrony for input patterns that more or less resemble these imprinted patterns.

This script takes a long time to run.

"""


import numpy as np
from matplotlib.pyplot import *
import os,sys
hc_path = "../../libHC"
sys.path.append(os.path.realpath(hc_path))
import networkx as nx
import hcLab as lab
import hcNetworks as net
import hcPlotting as plo
import hcUtil as ut
from numpy.random import RandomState
close('all')

downsample=100

M=15 # network size
N=15 # '' ''
num_patterns_initial= 2000 # initial size of pattern pool from which to sample - increases itself as needed:
patterns_per_bin = 1       # pattern pool is increased until this nr of patterns is found in each bin.
num_imprinted=10 # nr of high prior patterns
pattern_b=1 # pattern size: activation probability dropoff rate with distance from pattern center
pattern_c=0.2 # pattern size: activation probability cutoff with distance
conn_b_bck=1 # network connectivity dropoff rate with distance from other cell, for non-coactivated (background) cell pairs
conn_c_bck=0.3 # network connectivity cutoff with distance, for non-coactivated (background) cell pairs
conn_b=1 # dropoff rate for co-activated cells
conn_c=0.15 # relaxed cutoff for co-activated cells. Try 0.1: Stronger sync difference between high and low similarity, but connectivtiy structure seems very dense. Or try 0.2: Rather sparse-looking connectivity and more washed out sync result.

bins = np.arange(0.2,1.1,0.1) # edges of the desired similarity bins
n_samples = 100 # repetitions of the whole sampling procedure (networks & patterns)

experiments = []
def setup(seed,seednr,num_patterns):
    print "sampling network",seednr,"with a pool of",num_patterns,"patterns"
    rng = RandomState(seed)

    # generate patterns by choosing a point on the network and activating a random choice of cells near it
    patterns = np.zeros((M,N,num_patterns))
    for pat in range(num_patterns):
        margin = 2
        center = rng.randint(margin,M-margin),rng.randint(margin,N-margin)
        for i in range(M):
            for j in range(N):
                p_on = max(1.0/(pattern_b*np.sqrt((center[0]-i)**2 + (center[1]-j)**2))-pattern_c,0) if (i,j)!=center else 1
                #patterns[i,j,pat] = p_on
                if rng.rand() < p_on:
                   patterns[i,j,pat] = 1
        ## visualize patterns:
        # clf()
        # imshow(patterns[:,:,pat]);colorbar()
        # import pdb;pdb.set_trace()
    rng = RandomState(seed) # reinitialize rng so the sampled network is not dependent on the nr of previously sampled patterns

    # generate the network:
    # random network with distance-dependent connection probability,
    # with stronger links between cells that participate in the first num_imprinted patterns.
    network = net.grid_empty(M,N)
    nodes = network.nodes()
    for i,u in enumerate(nodes):
        for v in nodes[i+1:]:
            # if both nodes participate in the same pattern, make a strong link,
            # with some probability depending on distance
            in_pattern=False
            for pat in range(num_imprinted):
                if patterns[u[0],u[1],pat] and patterns[v[0],v[1],pat]:
                    in_pattern = True
                    break

            p_connect_pattern    = max(1.0/(conn_b*np.sqrt((u[0]-v[0])**2 + (u[1]-v[1])**2))-conn_c,0)
            p_connect_background = max(1.0/(conn_b_bck*np.sqrt((u[0]-v[0])**2 + (u[1]-v[1])**2))-conn_c_bck,0)
            if in_pattern and rng.rand()<p_connect_pattern:
                network.add_edge(u,v,{"strength":15})
            # fewer and weaker background connections are created where there was no common input.
            elif rng.rand()<p_connect_background:
                network.add_edge(u,v,{"strength":1})

    # create a setup (experiment object) for each pattern to be presented to the network
    experiments_this_net = []
    similarities_this_net = []
    for i in range(num_patterns):
        current = patterns[:,:,i]
        ex = lab.experiment(network,[rng.randint(1,10000)],inputc=current, name="seed "+str(seednr)+" pattern "+str(i), downsample=downsample, verbose=True, con_upstr_exc=2,
                                            measures=[lab.spikey_rsync(roi=current,name="rsync",tau=10.0/downsample),
                                                      lab.mean_spikecount(roi=current,name="spikes"),
                                                      ])
        # calculate this pattern's similarity to imprinted patterns
        # (the fraction of its cells it shares with an imprinted pattern)
        overlaps = [np.sum(current*patterns[:,:,j])/float(np.sum(current)) for j in range(num_imprinted)]
        nr_active = np.sum(current) # nr of active cells in the pattern (for normalization)
        all_imprinted = np.sum(patterns[:,:,0:num_imprinted],axis=2)
        all_imprinted[all_imprinted>1] = 1
        similarity = np.sum(current*all_imprinted)/float(nr_active)

        activated_subnet = network.subgraph([node for node in zip(*np.where(current))])
        edges = [edge for edge in activated_subnet.edges_iter(data=True) if edge[2]["strength"]>1]

        ex.network_match = len(edges)/float(np.sum(current > 0))

        # import ipdb; ipdb.set_trace()

        ex.similarity = similarity
        ex.similar_to = zip(overlaps,[patterns[:,:,j].copy() for j in range(num_imprinted)])
        similarities_this_net.append(similarity)
        # if i<num_imprinted:
        #     ex.name+="_imprinted"
        experiments_this_net.append(ex)


    # sort all experiments that use this network by pattern similarity
    sort = np.digitize(similarities_this_net,bins,right=True)
    experiments_binned = [[] for _ in bins]
    similarities_binned = [[] for _ in bins]
    for i,ex in enumerate(experiments_this_net):
        experiments_binned[sort[i]].append(ex)
        similarities_binned[sort[i]].append(ex.similarity)

    # check whether there are enough experiments in each pattern similarity bin
    if np.min([len(s) for s in similarities_binned]) >= patterns_per_bin:
        return np.array([column[0:patterns_per_bin] for column in experiments_binned]).flatten()
    elif num_patterns<num_patterns_initial*100:
        print "seednr "+str(seednr)+": "+str(num_patterns)+" sample patterns not enough, trying with more"
        return setup(seed,seednr,num_patterns*2)
    else:
        raise Exception("couldn't find required number of samples in each bin after "+str(num_patterns)+" patterns")

# repeatedly run the whole sampling procedure
for i in np.arange(n_samples):
    experiments.extend(setup(i,i,num_patterns_initial))



# sort all experiments on all networks, based on pattern similarity
similarities = [ex.similarity for ex in experiments]
experiments_binned = [[] for _ in bins]
sort = np.digitize(similarities,bins,right=True)
similarities_binned = [[] for _ in bins]
for i,ex in enumerate(experiments):
    experiments_binned[sort[i]].append(ex)
    similarities_binned[sort[i]].append(ex.similarity)

experiments_binned.append([e for e in experiments_binned[-1] if e.network_match >= 2])
bins = list(bins)+[1]


def doboxplot(data,xticklabels, do_scatter=False):
    grid(linestyle='-', which='major', axis='y',color='black',alpha=0.3)
    boxplot(data,notch=True,boxprops={'color':'black'},flierprops={'color':'black'},
                    medianprops={'color':'red'},whiskerprops={'color':'black','linestyle':'-'})
    # background scatterplot:
    if do_scatter:
        for i in range(len(data)):
            displacement = np.array([0.97+0.06*np.random.randn() for _ in range(len(data[i]))])
            scatter(i+displacement,data[i],marker='o',s=30,alpha=0.15,c=(0,0.2,0.8),linewidth=0)
    xlim((0,len(data)+1))
    xticks(np.arange(0.5,len(data)+1.5),xticklabels, rotation=0)


def plot_setups(experiments,save=True):
    for i,ex in enumerate(experiments):
        figure(figsize=(3,3))
        plo.eplotsetup(ex,'rsync')
        title("similarity "+str(ex.similarity))
        if save:
            savefig(ex.name+'.pdf', bbox_inches='tight')

# plot one example from each similarity category
picture_seed = 79
plot_setups([column[picture_seed] for column in experiments_binned[:-1]])
# make a video of an example from the highest similarity bin
last = experiments_binned[-2][picture_seed].saveanimtr(0,10,2,grid_as='graph')


figure(figsize=(3,3))
plo.plotsetup(experiments_binned[0][79].network,np.zeros((M,N)),np.zeros((M,N)),gca(),grid_as='graph')
title('network')
savefig('network.pdf', bbox_inches='tight')

# fetch synchrony measurements from trials where there was at least 1 spike
# (this triggers the simulation to be run)
rsyncs = [[] for _ in bins]
# spikecounts_ = [[] for _ in bins]
for i,column in enumerate(experiments_binned):
    for j,ex in enumerate(column):
        print "\n bin",i,"ex",j
        spikecount = ex.getresults('spikes')
        if np.mean(spikecount) >= 0.01:
            rsyncs[i].append(ex.getresults('rsync'))
            # spikecounts_[i].append(spikecount)

print "nr of samples per bin:", [len(s) for s in rsyncs]

# plot them
# figure(figsize=(5,4))
# doboxplot(spikecounts_,[0]+bins.tolist())
# ylabel("spikecount")
# savefig('spikecount.pdf', bbox_inches='tight')

figure(figsize=(3,3))
doboxplot(rsyncs,[0]+bins)
ylabel("Rsync")
ylim(0,1)
xlabel("Similarity")
savefig('rsync.pdf', bbox_inches='tight')



figure(figsize=(3,3))
title('connectivity of inout-receiving cells')
doboxplot([[e.network_match for e in bin] for bin in experiments_binned], [0]+bins)
ylabel("# connections / # input-receiving")
ylim(ymin=-0.1)
xlabel("Similarity index")
savefig('network_sampling_variability.pdf', bbox_inches='tight')




figure()
lowest_sync_highest_similarity = experiments_binned[-2][np.argmin([exp.getresults('rsync') for exp in experiments_binned[-2]])]
plo.eplotsetup(lowest_sync_highest_similarity, measurename='rsync')
title('example of a situation with low sync despite high similarity index')
savefig('setup__low_sync_high_similarity.pdf', bbox_inches='tight')



print('\a')