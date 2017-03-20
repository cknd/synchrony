### Source files to accompany the article "Cortical Spike Synchrony as a Measure of Input Familiarity".


## Overview

These sources can be divided in three subsets:

- a numerical simulation of noise-driven FitzHugh-Nagumo and Izhikevich spiking networks written in C.
- a python wrapper for the simulation, offering an epidemic of functions for preparing & visualizing experiments on such networks.
- a number of python scripts that generate the figure panels shown in the article along with some additional plots.

A note on code quality:
I'm making this public because scientific code should be public, even and especially if its creator is not entirely proud of it. This is not software. This is a collection of experiments that had to survive many shifts in requirements as the science progressed. Part of the codebase started out back when python 2 looked like a good idea. Having said all that, I also want to say, fear not: Everything produces colorful plots and the main pathways are quite well-aired and documented.


## Dependencies
- numpy & matplotlib (http://www.scipy.org/)
- the "networkx" graph library (http://networkx.github.io/)

- the GNU scientific library (http://www.gnu.org/software/gsl/)

- (ffmpeg, but only if you want to export animations. https://www.ffmpeg.org/)


#### Tested with:
- OSX 10.9 - 10.12, Ubuntu 12.04
- LLVM/clang 8 (via xcode 8.2)
- GNU scientific library 2.2.1

- Python 2.7.11
- Numpy  1.12.0
- matplotlib  2.0.0
- networkx  1.8.1

## Installation:
with all dependencies installed,

- compile the simulation using the makefile in `libHC/ekkesim` (i.e. in that directory, type `make izhikevich`)
    - in case you have the GSL installed in a non-standard path, you may use `make local PREFIX={the path prefix used when configuring & installing the GSL}` for convenience. This just passes PREFIX/include and PREFIX/lib to the gcc's -I and -L options, respectively.
- run the example script in an interactive python session: `python -i libHC/example.py`


## Getting started
If your aim is to retrace in detail the computations that lead to each figure, I suggest the following reading order:

- run & read `libHC/example.py`
- read `libHC/hcLab.py`, starting with the `experiment` class. Each such object controls the simulations & measurements for one stimulus condition.
- the interface to the C simulation code is in `libHC/hcWrapSim.py`. Consider setting a breakpoint at the end of `run_sim()` in that file to examine the arguments passed to the C simulation & to view its unprocessed output.
- The simulation code is in `libHC/ekkesim/src/simulation_izhikevich.c`.
- The various figure-generating scripts live in `./figures`. they make references to some additional modules e.g. for network generation or for plotting.
