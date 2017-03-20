
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
rng = RandomState(1)
save=True
delta_t=0.005
downsample = 20
M,N = 4,18
ms_per_step = delta_t*downsample
# par = 'A'

def line(start,end,thickness=2):
    inputc = np.zeros((M,N))
    center = np.floor(M/2)
    hi = int(center+np.floor(thickness/2.0))
    lo = int(center-np.ceil(thickness/2.0))
    inputc[lo:hi,start:end] = 1
    return inputc


# define measurement spots
mrg = 2
left_spot  = line(mrg,mrg+2)
right_spot  = line(N-2-mrg,N-mrg)

connected_line = line(mrg,N-mrg)


patterns = [connected_line]


# generate the network
conn_b=1
conn_c=0.2
conn_b_bck=1
conn_c_bck=0.3
network_template = net.grid_empty(M,N)
nodes = network_template.nodes()
for i,u in enumerate(nodes):
    for v in nodes[i+1:]:
        # see if this pair of cells is part of the line:
        in_pattern=False
        for pat in patterns:
            if pat[u[0],u[1]] and pat[v[0],v[1]]:
                in_pattern = True
                break

        dist = np.sqrt((u[0]-v[0])**2 + (u[1]-v[1])**2)
        # p = max(1.0/(conn_b*dist-conn_c),0)

        p_patt = max(1.0/(conn_b*dist)-conn_c,0)
        p_backg = max(1.0/(conn_b_bck*dist)-conn_c_bck,0)

        if in_pattern and (dist <= 1.5 or rng.rand() < p_patt):
            network_template.add_edge(u,v,{"strength":-1})
        elif not in_pattern and (dist <= 1.5 or rng.rand() < p_backg):
            network_template.add_edge(u,v,{"strength":1})


parlist = [
           'weak',
           # 'weak_jitter5',
           # 'weak_jitter10',
           'strong',
           # 'strong_lessinput',
           ]

for par in parlist:
    if par == 'strong_lessinput':
        strength = 15
        con_upstr_exc = 2
        jitter = 0
    elif par == 'strong':
        strength = 15
        con_upstr_exc = 3
        jitter = 0
    elif par == 'weak':
        strength = 5
        con_upstr_exc = 3.5
        jitter = 0
    elif par == 'weak_jitter5':
        strength = 5
        con_upstr_exc = 3.5
        jitter =5
    elif par == 'weak_jitter10':
        strength = 5
        con_upstr_exc = 3.5
        jitter = 10
    else:
        raise Exception('wrong name')

    network = network_template.copy()
    for u,v,d in network.edges(data=True):
        strn = d['strength']
        if strn == -1:
            d['strength'] = strength

    # set up experiments
    def setup(inputc,name):
        return lab.experiment(network,lab.standard_seeds[0:16],name=name,inputc=inputc, T=5000, verbose=True, downsample=downsample, con_upstr_exc=con_upstr_exc,
                              measures=[lab.spikey_rsync(roi=left_spot+right_spot,name="$R_{syn}$", tau=10.0/downsample),
                                        lab.spikedetect_add_jitter(roi=left_spot,name="left",jitter=jitter),
                                        lab.spikedetect_add_jitter(roi=right_spot,name="right",jitter=jitter),
                                        ])

    gap,connected = setup(left_spot+right_spot,'A'),setup(connected_line,'B')

    left, right = setup(left_spot, 'left'), setup(right_spot,'right')


    def correlogram(experiment):
        left = experiment.getresults('left')
        right = experiment.getresults('right')
        corr_trials = [np.correlate(l,r,'same') for l,r in zip(left,right)]
        return np.mean(corr_trials,axis=0)

    def plot_correlograms(experiments):
        cgrams = [[0]+correlogram(e).tolist()+[0] for e in experiments]
        allc = np.concatenate(cgrams)
        vmin, vmax = np.min(allc), np.max(allc)*1.05

        for cgram, experiment in zip(cgrams, experiments):
            figure(figsize=(5,2))
            leng = len(cgram)
            fill(np.arange(-leng/2,leng/2)*ms_per_step, cgram, 'k')
            xmax = 75 if 'weak' in par else 200
            xlim(-xmax, xmax)
            xlabel('ms')
            ylabel('spikes')
            ylim(vmin, vmax)
            title(experiment.name)
            if save:
                savefig(par+'_corr_'+experiment.name+'.pdf', bbox_inches='tight')

    def plot_setups(experiments):
        for ex in experiments:
            figure(figsize=(0.15*N,0.15*M))
            plo.eplotsetup(ex,'$R_{syn}$')
            if save:
                savefig(par+'_'+ex.name+'.pdf', bbox_inches='tight')

    plot_setups([gap,connected])


    connected.viewtrial()
    savefig(par+'_example_trial_connected.pdf', bbox_inches='tight')

    # def animate(ex):
    #     close('all')
    #     ex.saveanimtr(trialnr=0, start=0, skip=5, stop=5000, grid_as='graph', filename="{}_{}.mp4".format(par, ex.name), ms_per_step=ms_per_step, dpi=60)

    # animate(connected)

    close('all')
    connected.viewtrial()

    gap.viewtrial()
    savefig(par+'_example_trial_gap.pdf', bbox_inches='tight')

    plot_correlograms([gap, connected])


    rsfig = plo.compare([gap,connected],'$R_{syn}$',grid_as=None,plot_as='boxplot', label_names=True, vrange=[0,1], rotation=90)
    rsfig.set_size_inches(1.5,5, forward=True)
    if save:
        savefig(par+'_rsync.pdf', bbox_inches='tight')

print("\a")