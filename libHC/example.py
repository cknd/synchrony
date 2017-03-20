"""
shows two simple example experiments.

if run in an interactive python session, two figures should appear.
"""


import numpy as np
import hcPlotting as plo
import matplotlib.pyplot as plt

import hcNetworks as net
import hcLab as lab

# define a simple stimulus. this is done with a grid array.
inputc = np.zeros((5,7))
inputc[2:4,2:6] = 1.0

# the subpopulation whose synchrony we want to measure is defined as a binary mask.
roi = np.zeros(inputc.shape)
roi[1:4, 1:4] = 1


# let's say we want to compare two networks with different synapse strengths.
M,N = inputc.shape
networkA = net.grid_eightNN(M, N, strength=10)
networkB = net.grid_eightNN(M, N, strength=1)


# set up the experiments. we want to run 5 repetitions and each time measure the synchrony in the measured region.
seeds = lab.standard_seeds[0:5]
exprA = lab.experiment(networkA, seeds, inputc=inputc, measures=[lab.spikey_rsync(roi)], T=300, name="A", verbose=True)
exprB = lab.experiment(networkB, seeds, inputc=inputc, measures=[lab.spikey_rsync(roi)], T=300, name="B", verbose=True)


# plot voltage traces of all cells from one of the repetitions:
exprA.viewtrial()  # causes a single simulation to be computed
plt.show()

# # compare measurements from the two experiments:
plo.compare([exprA, exprB], measurename='spikey_rsync', grid_as='graph') # causes all trials to be computed

# # # show an animated simulation run:
# exprA.viewtrial(animate=True,grid_as='graph')


