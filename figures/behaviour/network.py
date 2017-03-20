"""
show spike activity for various parameter combinations, including inhibitory connections
"""

import numpy as np
from matplotlib.pyplot import *
import os,sys
hc_path = "../../libHC"
sys.path.append(os.path.realpath(hc_path))
import hcNetworks as net
import hcWrapSim as wrp
import hcPlotting as plo
import hcLab as lab
from hcUtil import slug
seed = 1234


T = 1000
transient = 500
delta_t = 0.005
downsample = 1



                # All excitatory, various conductances:
parameters = [{'strength':0, 'con_upstr_exc':3.5, 'inh_recurrent':0, 'inh_lateral':0, 'inh_random':0},
              {'strength':2, 'con_upstr_exc':3.5, 'inh_recurrent':0, 'inh_lateral':0, 'inh_random':0},
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':0, 'inh_lateral':0, 'inh_random':0, 'lineplots':'inh_lateral, inh_recurrent, inh_random'},
              {'strength':10,'con_upstr_exc':3.5, 'inh_recurrent':0, 'inh_lateral':0, 'inh_random':0},
              {'strength':15,'con_upstr_exc':3.5, 'inh_recurrent':0, 'inh_lateral':0, 'inh_random':0},
              {'strength':20,'con_upstr_exc':3.5, 'inh_recurrent':0, 'inh_lateral':0, 'inh_random':0},
               # add some purely recurrent inhibition:
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':5,'inh_lateral':0, 'inh_random':0 , 'lineplots':'inh_recurrent'},
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':10,'inh_lateral':0, 'inh_random':0, 'lineplots':'inh_recurrent'},
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':15,'inh_lateral':0, 'inh_random':0, 'lineplots':'inh_recurrent'},
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':20,'inh_lateral':0, 'inh_random':0, 'lineplots':'inh_recurrent'},
              {'strength':15,'con_upstr_exc':3.5, 'inh_recurrent':15,'inh_lateral':0, 'inh_random':0},
              {'strength':15,'con_upstr_exc':3.5, 'inh_recurrent':30,'inh_lateral':0, 'inh_random':0},
               # add some lateral inhibition:
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':5,'inh_lateral':5, 'inh_random':0,   'lineplots':'inh_lateral'},
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':10,'inh_lateral':10, 'inh_random':0, 'lineplots':'inh_lateral'},
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':15,'inh_lateral':15, 'inh_random':0, 'lineplots':'inh_lateral'},
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':20,'inh_lateral':20, 'inh_random':0, 'lineplots':'inh_lateral'},
              {'strength':15,'con_upstr_exc':3.5, 'inh_recurrent':15,'inh_lateral':15, 'inh_random':0},
              {'strength':15,'con_upstr_exc':3.5, 'inh_recurrent':30,'inh_lateral':30, 'inh_random':0},
                # purely random inhibition:
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':0, 'inh_lateral':0, 'inh_random':5 , 'lineplots':'inh_random'},
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':0, 'inh_lateral':0, 'inh_random':10, 'lineplots':'inh_random'},
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':0, 'inh_lateral':0, 'inh_random':15, 'lineplots':'inh_random'},
              {'strength':5, 'con_upstr_exc':3.5, 'inh_recurrent':0, 'inh_lateral':0, 'inh_random':20, 'lineplots':'inh_random'},
              {'strength':15,'con_upstr_exc':3.5, 'inh_recurrent':0, 'inh_lateral':0, 'inh_random':5},
              {'strength':15,'con_upstr_exc':3.5, 'inh_recurrent':0, 'inh_lateral':0, 'inh_random':15},

              {'strength':5, 'con_upstr_exc':2,   'inh_recurrent':0, 'inh_lateral':0, 'inh_random':0},
              {'strength':15,'con_upstr_exc':2,   'inh_recurrent':0, 'inh_lateral':0, 'inh_random':0},
              ]

rsyncs = []
for k,pars in enumerate(parameters):
    titlestr = "$g^{{net}}={}$".format(pars['strength'])
    titlestr += ", $g^{{up}}={}$".format(pars['con_upstr_exc'])
    titlestr += ", $g^{{inh, recurr.}}$={}".format(pars['inh_recurrent']) if pars['inh_recurrent'] != 0 else ""
    titlestr += ", $g^{{inh, lateral}}$={}".format(pars['inh_lateral']) if pars['inh_lateral'] != 0 else ""
    titlestr += ", $g^{{inh, random}}$={}".format(pars['inh_random']) if pars['inh_random'] != 0 else ""

    M, N = 4, 10
    inputc_ = 0.0 * np.ones((M, N))
    inputc_[M/2-1:M/2+1, 2:N-2] = 1.0

    network = net.grid_eightNN(M,  N,  strength=pars['strength'])

    network, inputc, M_inh, N_inh = net.add_neighbouring_inhibition(network, inputc_, inh_recurrent=pars['inh_recurrent'],
                                                                    inh_lateral=pars['inh_lateral'], exc_strength=100, input_to_inh=False)

    fig = figure(figsize=(N*0.7,M*0.7))
    ax = fig.add_subplot(111)
    fig.patch.set_visible(False)
    ax.axis('off')
    is_exc_only = pars['inh_recurrent']==0 and pars['inh_lateral']==0
    plo.plotsetup(network,inputc,np.zeros_like(inputc),grid_as='graph', axes=ax, nodesize=3,
                  subgraph='excitatory' if is_exc_only else None, edgecolor_offset=0.6 if is_exc_only else 0)
    margins(0)
    tight_layout()
    savefig('setup__{}.pdf'.format(slug(titlestr)), dpi=300)

    volt, recov, spikes = wrp.run_cached(graph=network,
                                         inputc=inputc,
                                         inputc_inh=inputc,
                                         lamini=1,
                                         seed=seed,
                                         T=T,
                                         delta_t=delta_t,
                                         transient=transient,
                                         downsample=downsample,
                                         con_upstr_exc=pars['con_upstr_exc'],
                                         con_upstr_inh=pars['inh_random'],
                                         verbose=1
                                         )

    show_inhibitory_cells = False
    show_inactive_cells = True
    shuffle = False

    if not show_inhibitory_cells:
        volt, spikes = volt[::2,:,:], spikes[::2,:,:]
        plot_M = M
    else:
        plot_M = M_inh

    if shuffle:
        order = np.random.RandomState(123).permutation(plot_M*N)
    else:
        order = range(plot_M*N)

    n_cells = plot_M*N
    volt_unrolled = volt.reshape(n_cells, -1)[order]
    spikes_unrolled = spikes.reshape(n_cells, -1)[order]

    if not show_inactive_cells:
        active_idxs = np.sum(spikes_unrolled, axis=1) > 0
        volt_unrolled = volt_unrolled[active_idxs, :]
        spikes_unrolled = spikes_unrolled[active_idxs, :]
        n_cells = sum(active_idxs)


    ms_per_step = delta_t*downsample
    s_per_step = ms_per_step/1000
    max_seconds = volt.shape[-1]*s_per_step

    # figure()
    # plo.plotsetup(network, inputc,inputc, grid_as='graph')
    # savefig('setup_{}.pdf'.format(k))

    figure(figsize=(7, 2))
    # imshow(volt_unrolled, vmin=-70, vmax=30, aspect='auto', interpolation='nearest', cmap='bone',  extent=[0, max_seconds, n_cells, 0])
    # [t.set_color('white') for t in gca().xaxis.get_ticklines()]
    # colorbar()
    # add spike markers
    idxs, spiketimes = np.nonzero(spikes_unrolled)
    scatter(spiketimes*s_per_step, idxs+5.0/N, marker='|', s=15, alpha=1, linewidth=1) #, color='k')
    ylabel('Neuron Nr.')
    ylim(0,n_cells)
    xlim(0,max_seconds)
    title(titlestr, fontsize=10)
    # xlabel('t (s)')


    tight_layout()
    savefig('activity__{}.pdf'.format(slug(titlestr)), dpi=300)

    if pars['strength'] == 0:
        figure(figsize=(14, 1.5))
        trace = volt[2,4,:]
        plot(np.linspace(0,max_seconds,volt.shape[-1]), trace, linewidth=0.5)
        xlim(0,max_seconds)
        ylabel('mV')
        tight_layout()
        savefig('singlecell_{}.pdf'.format(k), dpi=300)

    close('all')


    import hcLab
    rs_s = hcLab.spikey_rsync(tau=2, delta_t=delta_t*downsample, roi=inputc_)
    rsync = rs_s.compute(np.array(spikes, dtype='bool'))
    print rsync
    rsyncs.append(rsync)


for plottype,label in zip(["inh_random", "inh_recurrent", "inh_lateral"], ["$g^{{inh, recurr.}}$", "$g^{{inh, lateral}}$", "$g^{{inh, random}}$"]):
    figure(figsize=(3,2))
    plot_rsyncs = [(p[plottype],r) for r,p in zip(rsyncs, parameters) if plottype in p.get('lineplots',[])]
    plot(*zip(*plot_rsyncs))
    hlines(rsyncs[0], 0, 20, linestyles=':')
    ylim(0,0.3)
    xlabel(label)
    ylabel("$R_{syn}$")
    tight_layout()
    savefig("rsync_{}.pdf".format(plottype))
