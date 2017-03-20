"""
experiment setup, running & evaluation
CK 2014
"""
import copy
import numpy as np
import networkx as nx
import hcWrapSim as wrp
import hcNetworks as net
import hcPlotting as plo
from hcUtil import printprogress
from matplotlib import pyplot as plt
from numpy.random import RandomState

# provide a fixed set of random seeds
rng = RandomState(1)
standard_seeds = rng.randint(0,100000,200)

class measure:
    def __init__(self, roi=None, wait=0, name=None, window=None, windowlength=1000, increment=1):
        """
        A measure runs some analysis on raw simulation data & remembers the result.
        It accumulates results when applied repeatedly.

        Args:
            roi: a bool or 0/1 array of the size of the network grid, determines
                 which cells to measure.
            wait: ignore the first x steps of the recording
            name: name of this measurement
            window: Which, if any, windowed analysis to perform
                - None: measure on the whole recording
                            (starting after the first 'wait' steps)
                - "moving": measure repeatedly in a stepwise-moving section of
                            the recording (starting at 0)...
                - "tiling": ...in subsequent, non-overlapping sections of
                            the recording (starting at 0)...
                - "growing": ...in a growing section of the recording
                             (starting at the self.wait'th step)
            windowlength: Length of the window, if a windowed analysis is used.
            increment: If a growing window is used, grow it by ~ steps at a time.
        """
        self.roi = roi.astype(bool) if (roi is not None) else None
        self.name = name
        self.wait = wait
        self.window = window
        self.wlength = windowlength
        self.grw_startsize = 100
        self.increment = increment
        self.reset()

    def reset(self):
        """ delete all accumulated measurements"""
        self.results = []

    def apply(self, voltages, spikes, verbose=False):
        """
        analyze the given voltage traces or spiketrains & store the result.

        Args:
            recording: MxNxT numerical array, where M,N match the network's grid dimensions
        """
        ## breakpoint to verify match between measured region & recording.
        # f=300
        # import pdb; pdb.set_trace()
        # plt.clf();plt.imshow(recording[:,:,f],interpolation='nearest',vmax=2)
        # plt.imshow(self.roi,alpha=0.3,interpolation='nearest');f=f+10;plt.show()

        if isinstance(self, measure_spikebased):
            assert spikes.dtype == 'bool'
            recording = spikes
        else:
            recording = voltages

        if self.roi is not None:
            assert self.roi.shape == (recording.shape[0], recording.shape[1])

        # trigger slightly different computations depending on windowing mode:
        if self.window is None:
            result = self.compute(recording[:, :, self.wait:])
        elif self.window == "moving":
            result = self.compute_movingwindow(recording)
        elif self.window == "tiling":
            result = self.compute_tilingwindow(recording)
        elif self.window == "growing":
            result = self.compute_growingwindow(recording)
        else:
            raise Exception("Didnt understand window argument: " + self.window)
        if verbose:
            print 'result ', self.name, result
        self.results.append(result)
    def compute_movingwindow(self, recording):
        """ run the measure repeatedly in a moving window.

        Args:
            recording: MxNxT numerical array, where M,N match the network's grid dimensions
        """
        wl = self.wlength
        res = np.zeros(recording.shape[2]-wl)
        for step in range(len(res)):
            res[step] = self.compute(recording[:, :, step:step+wl])
            printprogress("calc. moving window, step ", step, len(res))
        return res

    def compute_tilingwindow(self, recording):
        """ run the measure repeatedly in subsequent, nonoverlapping windows

        Args:
            recording: MxNxT numerical array, where M,N match the network's grid dimensions
        """
        wl = self.wlength
        nrtiles = int(recording.shape[2]/wl) # floor via int()
        res = np.zeros(nrtiles)
        for tile in range(nrtiles):
            res[tile] = self.compute(recording[:, :, tile*wl:tile*wl+wl])
            printprogress("calc. tiling window, no. ", tile, nrtiles)
        return res

    def compute_growingwindow(self, recording):
        """ run the measure repeatedly in a growing window

        Args:
            recording: MxNxT numerical array, where M,N match the network's grid dimensions
        """
        tr = self.wait
        ssz = self.grw_startsize
        usablesteps = recording.shape[2]-tr-ssz
        res = np.zeros(int(np.ceil(usablesteps/float(self.increment))))
        for i, step in enumerate(range(0, usablesteps, self.increment)):
            res[i] = self.compute(recording[:, :, tr:tr+ssz+step])
            printprogress("calc. growing window, step ", i, len(res))
        return res

    def compute(self, recording):
        """ Measure something on the given (piece of) recording. Override this.

        Args:
            recording: MxNxT' numerical array, where M,N match the network's grid dimensions

        Returns:
            Some kind of measurement.
        """
        raise NotImplementedError()
        return None

class rsync(measure):
    """
    Strogatz' zero lag synchrony measure Rsync: The variance of the
    population mean normalized by the population's mean variance.
    """
    vrange = (0,1)  # plotting code will check whether a measure has a vrange attribute,
                    # if so, will use it to set fixed axis limits.
    def compute(self, recording):
        roii = self.roi.nonzero() # get the indices of the region to be measured
        selected_cells = recording[roii[0], roii[1], :]
        meanfield = np.mean(selected_cells, axis=0) # spatial mean across cells, at each time
        variances = np.var(selected_cells, axis=1)  # variance over time of each cell
        return np.var(meanfield)/np.mean(variances)



class measure_spikebased(measure):
    spike_based = True
    """ base class for spike time based sync measures """
    def __init__(self, roi, tau=1, delta_t=0.005, **kwargs):
        kernel = self.def_kernel(tau, delta_t)
        self.conv = lambda singlecell: np.convolve(singlecell, kernel, mode='valid')
        measure.__init__(self, roi, **kwargs)

    def def_kernel(self, tau, delta_t):
        ts = np.arange(0, tau*10, delta_t)
        decay = np.exp(-ts/tau)
        thr = 1e-3
        decay = decay[0:np.nonzero(decay < thr)[0][0]]
        kernel = np.concatenate((np.zeros_like(decay), decay))
        return kernel

    def get_convolved(self, recording, inspect=False):
        assert recording.dtype == 'bool'
        roii = self.roi.nonzero() # get the indices of the region to be measured
        selected_cells = recording[roii[0], roii[1], :]

        convd = np.apply_along_axis(self.conv, axis=1, arr=selected_cells)

        if inspect:
            plt.figure()
            plt.subplot(211)
            plt.imshow(convd)
            plt.subplot(212)
            plt.plot(np.mean(convd, axis=0))
        return convd


class mean_spikecount(measure_spikebased):
    """ average number of spikes in the measured population during the measured period """
    def compute(self, recording):
        roii = self.roi.nonzero()
        selected_cells = recording[roii[0], roii[1], :]
        return np.sum(selected_cells)/len(roii[0])

class spikedetect_additive(measure_spikebased):
    """ summed nr of spikes in the measured population at each time step"""
    def compute(self, recording):
        thr = 1
        roii = self.roi.nonzero() # get the indices of the region to be measured
        selected_cells = recording[roii[0], roii[1], :]
        return np.sum(selected_cells, axis=0)

class spikedetect_add_jitter(measure_spikebased):
    """ a noisy spike recorder """
    def __init__(self, jitter=0, delta_t=0.005, downsample=10, **kwargs):
        self.jitter_steps = jitter/(delta_t*downsample)
        measure_spikebased.__init__(self, **kwargs)

    def compute(self, recording):
        thr = 1
        roii = self.roi.nonzero() # get the indices of the region to be measured
        selected_cells = recording[roii[0], roii[1], :]
        if self.jitter_steps > 0:
            # import ipdb; ipdb.set_trace()
            offsets = (rng.rand(selected_cells.shape[0])-0.5) * self.jitter_steps
            for i,o in enumerate(offsets):
                selected_cells[i,:] = np.roll(selected_cells[i,:], int(o))
        return np.sum(selected_cells, axis=0)


class spikey_rsync(measure_spikebased):
    """ Rsync measure based on smoothed spike data """
    vrange = (0, 0.7)
    def compute(self, recording):
        convd = self.get_convolved(recording, inspect=False)
        meanfield = np.mean(convd, axis=0)
        variances = np.var(convd, axis=1)
        return np.var(meanfield)/np.mean(variances)


class vanrossum(measure_spikebased):
    """ the van Rossum distance """
    def compute(self, recording):
        convd = self.get_convolved(recording)

        pairwise_dist = []
        for trace_a in convd:
            for trace_b in convd:
                pairwise_dist.append(np.sqrt(np.mean((trace_a - trace_b)**2)))

        return np.mean(pairwise_dist)


class experiment:
    """
    Stores the specification of an experiment, gathers simulation results & applies measures.

    An experiment consists of a simulation setup (network, input pattern and other simulation parameters),
    a list of random seeds and a list of measures. The object then gives access to raw simulation results
    and results of measurements.
    """
    def __init__(self, network, seeds, measures=[], inputc=None, inputc_inh=None, name="Acme experiment", **simulation_kwargs):
        """
        Args:
            network: a networkx graph meeting the following assumptions: it has the graph attribute
                     network.graph["grid_dimensions"] holding the tuple (M,N), nodes are labeled as (i,j)
                     tuples to mark spatial coordinates in an MxN grid and each edge is labeled with
                     the float attribute "strength".
            seeds: list of random seeds - determines the number of repetitions
            measures: list of measures to apply
            inputc: MxN array, scales the rate of excitatory input pulses arriving at each cell
            inputc_inh: MxN array, scales the rate of inhibitory input pulses arriving at each cell
            simulation_kwargs: see `hcWrapSim.run_sim`

        """
        self.name = name
        self.network = network
        self.inputc = inputc
        self.inputc_inh = inputc_inh
        self.verbose = simulation_kwargs.get('verbose', False)

        simulation_kwargs['delta_t'] = simulation_kwargs.get('delta_t', 0.005)
        simulation_kwargs['T'] = simulation_kwargs.get('T', 1000)
        simulation_kwargs['downsample'] = simulation_kwargs.get('downsample', 10)

        self.simulation_kwargs = simulation_kwargs

        assert len(set(seeds)) == len(seeds)
        self.seeds = seeds
        self.measures = dict([(ms.name, copy.deepcopy(ms)) if ms.name
                                else (ms.__class__.__name__, copy.deepcopy(ms)) for ms in measures])

    def run(self):
        """Run all simulation trials & apply all measurements"""
        le = len(self.seeds)
        [meas.reset() for meas in self.measures.values()]
        for i, s in enumerate(self.seeds):
            printprogress('running "' + self.name + '", repetition ', i, le)
            voltage, spikes = self.getraw(i)
            for meas in self.measures.values():
                meas.apply(voltage, spikes, verbose=self.verbose)

    def getresults(self, which):
        """
        Return measurements from one specific measure.

        Args:
            which: name of the measure from which to return results

        Returns:
            List of measurement results, one for each random seed.
            Result type depends on the particular measure object.
        """
        if not self.measures[which].results:
            self.run()
        return self.measures[which].results

    def getraw(self, trialnr=0):
        """
        Return raw simulation data (voltage traces) of one trial.

        Args:
            trialnr: From which trial to return data
                     (which position in the list of seeds)

        Returns:
            MxNxT array of voltage traces
        """
        seed = self.seeds[trialnr]
        volt, _, spikes = wrp.run(self.network, seed, inputc_inh=self.inputc_inh, inputc=self.inputc, **self.simulation_kwargs)

        return volt, np.array(spikes, dtype='bool')

    def viewtrial(self, trialnr=0, animate=False, start=0, skip=10, grid_as="image",
                  shuffle=False, shuffle_seed=1, spikes_only=False, cmap=plt.cm.bone):
        """
        Quick & dirty visualization of the simulation data from one trial.

        Args:
            trialnr: From which trial to show data
            animate: If False, display a M*NxT trace image. If True, show a movie.
            start: When to start the animation. Applies if animate is true.
            skip: How many time steps to skip for each animation frame (higher = faster)
                  Applies if animate is true.
            grid_as: How to display the animation. Applies only if animate is true.
                     "image" - draw each node as a pixel using the node
                               labels as positions
                     "graph" - draw as a grid graph using the node labels
                               as positions
                     a networkx layout function - draw a graph with node
                               positions computed by that function
            shuffle: If true, shuffle the order of cells in the trace image.
                     Applies only if animate is false.
            shuffle_seed: int random seed for shuffling
            cmap: a matplotlib colormap object,
                  specifies how the voltage traces will be coloured
        """
        idx = 1 if spikes_only else 0
        if animate:
            if grid_as == "image":
                plo.viewanim(self.getraw(trialnr)[idx], start, skip, title=self.name, cmap=cmap)
            else:
                print "showing graph animation, using the first measure's r-o-i"
                plo.animate_graph(self.getraw(trialnr)[idx], self.network, self.measures.values()[0].roi,
                                  self.inputc, start, skip, grid_as, title=self.name, cmap=cmap)
        elif spikes_only:
            plo.view_spikes(self.getraw(trialnr)[1], title=self.name, shuffle=shuffle, shuffle_seed=shuffle_seed)
        else:
            s_per_step = self.simulation_kwargs['delta_t'] * self.simulation_kwargs['downsample'] / 1000
            plo.view_voltages(self.getraw(trialnr)[0], title=self.name, shuffle=shuffle, shuffle_seed=shuffle_seed, s_per_step=s_per_step)

    def plotsetup(self,measure=None):
        """ plot the network, input region and region of interest of the selected measure"""
        plo.eplotsetup(self,measure)

    def saveanimtr(self, trialnr=0, start=0, skip=4, stop=None, grid_as="image", dpi=120, cmap=plt.cm.bone, filename=None, ms_per_step=None):
        """ save an animation of a recording """
        if grid_as == "image":
            plo.saveanim(self.getraw(trialnr)[0], start, skip, title=self.name, dpi=dpi, cmap=cmap, filename=filename, ms_per_step=ms_per_step)
        else:
            print "saving graph animation, using the first measure's r-o-i"
            plo.saveanim_graph(self.getraw(trialnr)[0], self.network, self.measures.values()[0].roi,
                           self.inputc, start, skip, stop=stop, grid_as=grid_as, title=self.name, dpi=dpi, cmap=cmap, filename=filename, ms_per_step=ms_per_step)

    def statistics(self):
        """Return some topological information about the experiment"""
        stat = {}
        stat["net diameter"] = nx.diameter(self.network)
        stat["net radius"]   = nx.radius(self.network)
        stat["net asp"]     = nx.average_shortest_path_length(self.network)
        stat["input asp"] = net.inputASL(self.network, self.inputc)
        for m in self.measures.values():
            distr = net.distances_to_roi(self.network, self.inputc,m.roi)
            stat["stim to roi distances, mean",m.name] = np.mean(distr)
            stat["stim to roi distances, var",m.name] = np.var(distr)
            centrs = nx.closeness_centrality(self.network)
            stat["roi centralities",m.name] = [centrs[tuple(node)]
                                                for node in np.transpose(m.roi.nonzero())]
        return stat
