import os
import cPickle
import ctypes as ct
import numpy as np
import networkx as nx
from hashlib import sha1

path_sim = os.path.join(os.path.dirname(os.path.abspath(__file__)),"ekkesim/simulation_izhikevich.so")
               # assume the simulation lib lives in the same directory as this file here.
cachepath = '/tmp/hc_cache'
runtime_cache = {}

sim = ct.CDLL(path_sim)
sim.sagwas()


def run(graph, seed, use_cache=True, **kwargs):
    if use_cache:
        recordings = run_cached(graph, seed, **kwargs)
    else:
        recordings = run_sim(graph, seed, **kwargs)
    return recordings

class hcsetup:
    """ comparable & hashable simulation parameterization (needed for caching).
        A parameterisation the set of parameters that fully determines the outcome of a simulation:
        network, input pattern, duration, random seed..."""
    def __init__(self, graph, inputc=None, inputc_inh=None, **par):

        self.inputc = inputc.copy(order='C') if inputc is not None else None
        self.inputc_inh = inputc_inh.copy(order='C') if inputc_inh is not None else None
        self.edges = [str(e) for e in nx.to_edgelist(graph)]
        self.par = par

        par_s = [str((key, self.par[key])) for key in sorted(self.par.keys())]
        self.hashable = ''.join(self.edges)
        self.hashable += ''.join(par_s)
        self.hashable += str(self.inputc)
        self.hashable += str(self.inputc_inh)

    def __eq__(self, other):
        return (self.edges == other.edges
                and np.all(self.inputc == other.inputc)
                and np.all(self.inputc_inh == other.inputc_inh)
                and self.par == other.par)

    def __hash__(self):
        return hash(self.hashable)

    def longhash(self):
        return ("hcsetup_"+sha1(self.hashable).hexdigest())


def run_cached(graph, seed, verbose=False, **kwargs):
    """
    Return a simulation run of the given setup, reading from
    memory or from disk as appropriate
    """

    setup = hcsetup(graph, seed=seed, verbose=verbose, **kwargs)

    if setup in runtime_cache:
        if verbose:
            print '(reading from memory)'
        v,r,s = runtime_cache[setup]
    else:
        pthis = os.path.join(cachepath, setup.longhash())
        psetup = os.path.join(pthis, "setup.pk")
        precording = os.path.join(pthis, "recording.npz")

        if os.path.exists(pthis):
            if verbose:
                print "(reading from disk)"
            with open(psetup, 'rb') as fsetup:
                cached_setup = cPickle.load(fsetup)
                assert cached_setup == setup # if cache contains wrong setups, something stinks and we should crash.
            arch = np.load(precording)
            v,r,s = arch['v'], arch['r'], arch['s']
            runtime_cache[cached_setup] = (v,r,s)
        else:
            v,r,s = run_sim(graph, seed, verbose=verbose, **kwargs)
            runtime_cache[setup] = (v, None, s)
            if verbose:
                print '(cached in memory)'
    return v,r,s


def clear_runtime_cache():
    global runtime_cache
    runtime_cache = {}


def save_cache():
    for i, (setup, (v,r,s)) in enumerate(runtime_cache.items()):
        print '\r saving cached simulation', i
        pthis = os.path.join(cachepath,setup.longhash())
        psetup = os.path.join(pthis,"setup.pk")
        precording = os.path.join(pthis,"recording.npz")
        if not os.path.exists(pthis):
            os.makedirs(pthis)
            with open(psetup,'wb') as fsetup:
                cPickle.dump(setup,fsetup)
            np.savez_compressed(precording, v=v, r=None, s=s)


def run_sim(graph,
            seed,
            inputc=None,
            inputc_inh=None,
            laminex=40,
            lamini=30,
            con_upstr_exc=2.5,
            con_upstr_inh=0.,
            izhi_a=0.01,
            izhi_b=-0.1,
            izhi_reset_c=-65,
            izhi_recovery_d=12,
            taus=0.02,
            alp=8.0,
            bet=8.0,
            tauni=0.02,
            ani=8.0,
            bni=8.0,
            tauna=0.02,
            ana=8.0,
            bna=8.0,
            activation_noise=0.4,
            verbose=0,
            T=1000,
            delta_t=0.005,
            downsample=10,
            transient=500):
    """
    Return a simulation of the given network structure und the given input stimulus,
    extitatory and inhibitory input rate, neuron parameters etc.

    Args:
        graph: a networkx graph, with nodes labeled as the (i,j) coordinates of an MxN grid,
               and a graph attribute "grid_dimensions", graph.graph["grid_dimensions"] = (M,N),
               and edges labeled with a "strength" attribute.
        seed: a number > 0, will be cast to built-in int

        inputc: MxN array, determines the input rate for each node
        inputc_inh: MxN array, determines the inhibitory input rate for each node
        laminex: lambda_in_ex, the rate of excitatory input spikes to input-receiving nodes
        lamini: lambda_in_in,  the rate of inhibitory input spikes to input-receiving nodes
        con_upstr_exc: synapse conductivity of the excitatory random inputs
        con_upstr_inh: synapse conductivity of the inhibitory random inputs

        # neuron
        izhi_a: parameter a of the izhikevich neuron
        izhi_b: parameter b of the izhikevich neuron
        izhi_reset_c: reset constant of the neuron
        izhi_recovery_d: recovery reset constant

        # lateral synapses
        taus: duration of transmitter presence after spike
        alp: rising time constant of fraction of open receptors
        bet: decay time constant of fraction of open receptors

        # inhibitory synapses
        tauni: duration of transmitter presence after spike
        ani: rising time constant of fraction of open receptors
        bni: decay time constant of fraction of open receptors

        # external excitatory synapses
        tauna: duration of transmitter presence after spike
        ana: rising time constant of fraction of open receptors
        bna: decay time constant of fraction of open receptors

        activation_noise: scale of white noise added to each neuron's activation variable
        verbose: print a lot or not
        T: simulation duration in ms
        delta_t: integration step width
        downsample: discard every n'th step
        transient: discard the first n ms

    Return:
        tuple of arrays of shape MxNx(T/(delta_t*downsample)), containing
            - voltage traces.
            - recovery variable traces.
            - spike events
    """
    assert seed > 0 # GNU scientific RNG needs positive seeds
    T = T+transient
    n_steps = int(T/delta_t)

    M, N = graph.graph["grid_dimensions"]

    # set up some ctypes data structures:
    #    ...input pattern:
    #       (which we copy because it may have the wrong memory layout,
    #       e.g. if it is a slice of a larger array)
    if inputc is None:
        inputc = np.zeros((M,N))
    else:
        inputc = inputc.copy(order='C')

    if inputc_inh is None:
        inputc_inh = np.zeros((M,N))
    else:
        inputc_inh = inputc_inh.copy(order='C')

    ct_inputcontour_exc = inputc.ctypes.data_as(ct.POINTER(ct.c_double))
    ct_inputcontour_inh = inputc_inh.ctypes.data_as(ct.POINTER(ct.c_double))

    #    ...output buffers:
    recording_voltage = np.zeros((M,N,n_steps),order='C',dtype=np.double)
    ct_recording_voltage = recording_voltage.ctypes.data_as(ct.POINTER(ct.c_double))

    recording_recov = np.zeros((M,N,n_steps),order='C',dtype=np.double)
    ct_recording_recov = recording_recov.ctypes.data_as(ct.POINTER(ct.c_double))

    recording_spikes = np.zeros((M,N,n_steps),order='C',dtype=np.double)
    ct_recording_spikes = recording_spikes.ctypes.data_as(ct.POINTER(ct.c_double))

    #    ...network connectivity callback:
    #      allows ekke's simulation to read out the graph's adjacency structure.
    #      (the simulation separately pulls the two coordinates (modes 0, 1) and
    #      the connection strength (mode s) to each node's l neighbours)
    get_neighbors = graph.predecessors if isinstance(graph, nx.DiGraph) else graph.neighbors
    def ekkesim_connectivity(i,j,l,mode):
        neighbors = get_neighbors((i,j))
        le = len(neighbors)
        if mode == '0':
            return float(neighbors[l][1] if l<le else -1)
        elif mode == '1':
            return float(neighbors[l][0] if l<le else -1)
        elif mode == 's':
            if l<le:
                s = float(graph.get_edge_data(neighbors[l], (i,j))["strength"])
                return s
            else:
                return 0.0
    #   ...wrap connectivity callback as ctypes function:
    CMPFUNC = ct.CFUNCTYPE(ct.c_double, ct.c_int, ct.c_int, ct.c_int, ct.c_char)
    ct_connectivity = CMPFUNC(ekkesim_connectivity)

    #   ...wrap all non-integer numerals as ctypes double:
    ct_laminex = ct.c_double(laminex)
    ct_lamini  = ct.c_double(lamini)
    ct_izhi_a = ct.c_double(izhi_a)
    ct_izhi_b = ct.c_double(izhi_b)
    ct_izhi_reset_c = ct.c_double(izhi_reset_c)
    ct_izhi_recovery_d = ct.c_double(izhi_recovery_d)
    ct_con_upstr_exc = ct.c_double(con_upstr_exc)
    ct_con_upstr_inh = ct.c_double(con_upstr_inh)
    ct_delta_t = ct.c_double(delta_t)

    ct_taus = ct.c_double(taus)
    ct_alp = ct.c_double(alp)
    ct_bet = ct.c_double(bet)
    ct_tauni = ct.c_double(tauni)
    ct_ani = ct.c_double(ani)
    ct_bni = ct.c_double(bni)
    ct_tauna = ct.c_double(tauna)
    ct_ana = ct.c_double(ana)
    ct_bna = ct.c_double(bna)
    ct_activation_noise = ct.c_double(activation_noise)

    maxdegree = max(graph.degree().values())

    # Call the simulation:
    # import pdb; pdb.set_trace() # <-- last breakpoint before the highway
    sim.simulate(M,
                 N,
                 n_steps,
                 maxdegree,
                 ct_connectivity,
                 ct_recording_voltage,
                 ct_recording_recov,
                 ct_recording_spikes,
                 ct_inputcontour_exc,
                 ct_inputcontour_inh,
                 ct_laminex,
                 ct_lamini,
                 ct_izhi_a,
                 ct_izhi_b,
                 ct_izhi_reset_c,
                 ct_izhi_recovery_d,
                 ct_con_upstr_exc,
                 ct_con_upstr_inh,
                 ct_taus,
                 ct_alp,
                 ct_bet,
                 ct_tauni,
                 ct_ani,
                 ct_bni,
                 ct_tauna,
                 ct_ana,
                 ct_bna,
                 ct_activation_noise,
                 int(seed),
                 int(verbose),
                 ct_delta_t)

    if transient is None:
        transient = 0
    transient_steps = int(transient/delta_t)

    volt = np.array(recording_voltage[:, :, transient_steps:n_steps:downsample], dtype='float32') # bodge
    recov = np.array(recording_recov[:, :, transient_steps:n_steps:downsample], dtype='float32')

    si, sj, spiketimes = np.nonzero(recording_spikes[:, :, transient_steps:])
    spikes = np.zeros((M, N, (n_steps-transient_steps)/downsample), dtype='bool')
    spikes[si, sj, spiketimes/downsample] = True

    return volt, recov, spikes


def rantest(seed,N=100):
    """get some random numbers"""
    buff = np.zeros(N,dtype=np.double)
    ct_buff = buff.ctypes.data_as(ct.POINTER(ct.c_double))
    sim.rantest(seed,N,ct_buff)
    return buff


