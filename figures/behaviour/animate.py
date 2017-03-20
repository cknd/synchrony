import numpy as np
from matplotlib.pyplot import *
import os,sys
hc_path = "../../libHC"
sys.path.append(os.path.realpath(hc_path))
import hcLab as lab
import hcNetworks as net
import hcPlotting as plo
import hcUtil as ut

downsample=20
M,N = 15,15

pars = [
        '$g^{net}=5$',
        '$g^{net}=15$',
        '$g^{net}=0$'
        ]

for par in pars:
    if par == '$g^{net}=15$':
        lateral_strength = 15
        con_upstr_exc = 2

    elif par == '$g^{net}=5$':
        lateral_strength = 5
        con_upstr_exc = 3.5

    elif par == '$g^{net}=0$':
        lateral_strength = 0
        con_upstr_exc = 3.5

    # create the network
    network = net.grid_eightNN(M,N,strength=lateral_strength)

    # set up experiments
    seeds = lab.standard_seeds[0:1]
    inputc = np.ones((M,N))
    inputc[0,:] = 0
    inputc[-1,:] = 0
    inputc[:,0] = 0
    inputc[:,-1] = 0

    ex = lab.experiment(network,seeds,inputc=inputc, T=300, transient=1000, name=par, verbose=True,
                        downsample=downsample, con_upstr_exc=con_upstr_exc, measures=[lab.spikey_rsync(roi=inputc, name='rsync')])

    delta_t = ex.simulation_kwargs['delta_t']
    ms_per_step = delta_t*downsample

    close('all')
    ex.saveanimtr(trialnr=0, start=0, skip=6, grid_as='graph', filename="{}.mp4".format(ut.slug(ex.name)), ms_per_step=ms_per_step, dpi=70)


print '\a'