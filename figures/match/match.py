"""
Series of stimulus patterns with increasing dispersion on a grid network.

This script takes a long time to run. To see something happen quickly, run fewer trials
 by replacing "seeds = lab.standard_seeds[0:50]" with e.g. "...[0:10]"
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

downsample=50

# define a horizontal line stimulus
def horizontal_lines(M=6,N=None,lines=[(2,10)],width=3,margin=2):
    if not N:
        N = max([l[1] for l in lines]) + 2*margin
    inputc = np.zeros((M,N))
    for line in lines:
        (start,end) = line
        inputc[M/2-1:M/2+2,margin+start:margin+end] = 1
    return M,N,inputc

M,N,line = horizontal_lines(lines=[(1,6)],N=10,M=10)

pars = [
        'strongnet',
        # 'strongnet_moreinput',
        'weaknet'
        ]

for par in pars:
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
    network = net.grid_eightNN(M,N,strength=strength)


    # set up experiments
    seeds = lab.standard_seeds[0:50]


    scattervalues = [0,5,10]
    names = ['A','B','C']
    experiments = []
    for name,scatter in zip(names,scattervalues):
        pattern = net.scramble_stimulus(network,line,sct=scatter,seed=1)
        experiments.append(lab.experiment(network,seeds,inputc=pattern,transient=1000, name=name, verbose=True, downsample=downsample,con_upstr_exc=con_upstr_exc,
                              measures=[lab.spikey_rsync(roi=pattern,name="$R_{syn}$", tau=10.0/downsample),
                                        lab.spikey_rsync(roi=pattern,window="growing",increment=2000/downsample,name="$R_{syn, t}$", tau=10.0/downsample),
                                        lab.mean_spikecount(roi=pattern,window="growing",increment=2000/downsample,name="$spikecount_{t}$")
                                        ]))

    # # Plot
    close('all')

    plo.eplotsetup(experiments[0], measurename='$R_{syn}$')

    experiments[0].viewtrial()
    savefig(par+'_example_trial.pdf', bbox_inches='tight')

    delta_t = experiments[0].simulation_kwargs['delta_t']

    plo.compare(experiments,grid_as="graph",plot_as='boxplot',measurename="$R_{syn}$", vrange=[0, 1], label_names=True)
    savefig(par+'_overview_graphs.pdf', bbox_inches='tight')

    plo.compare_windowed(experiments,'$R_{syn, t}$', unit=delta_t*downsample/1000, plot_as='bandplot', do_title=False)
    savefig(par+'_rsync_change.pdf', bbox_inches='tight')

    plo.compare_windowed(experiments,'$spikecount_{t}$', unit=delta_t*downsample/1000, do_title=False)
    savefig(par+'_spikecount_change.pdf', bbox_inches='tight')

    nb = ut.naivebayes([experiments[0],experiments[-1]],"$R_{syn, t}$", range(len(seeds)/2))

    nb.plot_posteriors()
    savefig(par+'_posteriors.pdf', bbox_inches='tight')


    nb.evaluation()
    savefig(par+'_evaluation.pdf', bbox_inches='tight')



    nb.plot_all_times_to_correct_decision(thr=0.95,stay_above=False, unit='spikes', spikemeasure="$spikecount_{t}$", do_title=False)
    savefig(par+'_timetocorrectdecision.pdf', bbox_inches='tight')


    nb.plot_all_times_to_correct_decision(thr=0.5,stay_above=True, unit='spikes', spikemeasure="$spikecount_{t}$", do_title=False)
    savefig(par+'_timetocorrectdecision_2.pdf', bbox_inches='tight')


