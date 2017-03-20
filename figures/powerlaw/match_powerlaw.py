"""
Series of stimulus patterns with increasing dispersion on a grid network.

~

This script takes a long time to run. To see something happen quickly,
    - run fewer trials, by replacing "seeds = standard_seeds[0:50]" with e.g. "...[0:10]"
    - reduce the resolution of growing window calculations by setting
      both occurences of "increment=10" in this file to e.g. 100.

CK '14
"""

import numpy as np
from matplotlib.pyplot import *
import os,sys
hc_path = "../../libHC"
sys.path.append(os.path.realpath(hc_path))
import hcLab as lab
import hcNetworks as net
import hcPlotting as plo
import hcUtil as ut
import networkx as nx

downsample=50

for par in ['strongnet','strongnet_moreinput', 'weaknet']:
    if par == 'strongnet':
        strength = 15
        con_upstr_exc = 2

    if par == 'strongnet_moreinput':
        strength = 15
        con_upstr_exc = 3

    elif par == 'weaknet':
        strength = 5
        con_upstr_exc = 3.5


    # create the network
    M,N = 8,10
    network = net.grid_powerlaw(M,N,1,0.1,strength=strength,seed=1)
    assert nx.is_connected(network)

    # define a stimulus seed pattern (two spots somewhere centrally)
    init = net.createroi(M,N,network,4)

    # set up experiments
    seeds = lab.standard_seeds[0:50]

    scattervalues = [0,40]

    experiments = []
    for scatter in scattervalues:
        pattern = net.createstimulus(M,N,network,init,10,scatter=scatter,method="mean_shortest")
        experiments.append(lab.experiment(network,seeds,inputc=pattern,transient=1000, name="scatter "+str(scatter),verbose=True, downsample=downsample,con_upstr_exc=con_upstr_exc,
                              measures=[lab.spikey_rsync(roi=pattern,name="rsync", tau=10.0/downsample),
                                        lab.spikey_rsync(roi=pattern,window="growing",increment=2000/downsample,name="growing_rsync", tau=10.0/downsample),
                                        lab.mean_spikecount(roi=pattern,window="growing",increment=2000/downsample,name="growing_spikecount")
                                        ]))



    # # Plot
    close('all')


    experiments[0].viewtrial()
    savefig(par+'_example_trial.pdf', bbox_inches='tight')

    delta_t = experiments[0].simulation_kwargs['delta_t']

    layout = plo.get_reusable_springlayout(network)
    plo.compare(experiments,grid_as=layout,plot_as='boxplot',label_stats=['input asp'],measurename="rsync", vrange=[0, 1])
    savefig(par+'_overview_graphs.pdf', bbox_inches='tight')

    plo.compare_windowed(experiments,'growing_rsync', unit=delta_t*downsample/1000)
    savefig(par+'_rsync_change.pdf', bbox_inches='tight')

    plo.compare_windowed(experiments,'growing_spikecount', unit=delta_t*downsample/1000)
    savefig(par+'_spikecount_change.pdf', bbox_inches='tight')

    nb = ut.naivebayes([experiments[0],experiments[-1]],"growing_rsync", range(len(seeds)/2))

    nb.plot_posteriors()
    savefig(par+'_posteriors.pdf', bbox_inches='tight')


    nb.evaluation()
    savefig(par+'_evaluation.pdf', bbox_inches='tight')



    nb.plot_all_times_to_correct_decision(thr=0.95,stay_above=False, unit='spikes')
    savefig(par+'_timetocorrectdecision.pdf', bbox_inches='tight')


    nb.plot_all_times_to_correct_decision(thr=0.5,stay_above=True, unit='spikes')
    savefig(par+'_timetocorrectdecision_2.pdf', bbox_inches='tight')


