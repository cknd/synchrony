"""
plotting. here be dragons.
CK 2014
"""
import numpy as np
import matplotlib
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import animation
from hcUtil import printprogress
from numpy.random import RandomState
plt.ion()

golden_mean = (np.sqrt(5)-1.0)/2.0
fig_width = 2*3.27 # column width in inches
fig_height = fig_width*golden_mean
fig_size = (fig_width,fig_height)
fig_size_wide = (fig_width,fig_height*0.75)
params = {'backend': 'ps',
          #'figure.autolayout' : True,
          'font.size' : 10,
          'axes.labelsize' : 12,
          'legend.fontsize': 12,
          'xtick.labelsize' : 10,
          'ytick.labelsize' : 10}

matplotlib.rcParams.update(params)

# plotting of raw simulation data:

def createanim(data,start,skip,title=None,cmap='bone', ms_per_step=None):
    """
    Return an animation of a single simulation run, each node
    represented (via its node label) as pixel (i,j) in an MxN image.
    So this will only be useful for grid graphs.

    Args:
        data: MxNxT array of voltage traces
        start: first timestep to show
        skip: timesteps to advance in each frame (higher -> faster)
        title: figure title
        cmap: matplotlib colormap (name OR object)

    Return:
        matplotlib animation object
    """
    plt.ioff()
    fig = plt.figure(figsize=fig_size)
    titlefont = {'color'  : 'black', 'size':12}
    ax  = plt.axes()#(xlim=(0, data.shape[1]), ylim=(data.shape[0]))
    picture = ax.imshow(data[:, :, start], vmin=data.min(), vmax=data.max(),
                         interpolation="nearest", cmap=cmap)
    plt.colorbar(picture,ax=ax,shrink=0.7)
    def init():
        picture.set_data(data[:,:,start])
        return picture

    def animate(i):
        printprogress("animating frame",start+i*skip,data.shape[2])
        if i*skip < data.shape[2]:
            picture.set_data(data[:,:,start+i*skip])

            t = " {}ms".format(start+i*skip * ms_per_step) if ms_per_step is not None else ""
            plt.title(title+t,titlefont)
        return picture
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                          frames=(data.shape[2]-start)/skip,interval=1, blit=False)
    return anim

def viewanim(data,start=0,skip=2,title=None,cmap='bone', ms_per_step=None):
    """
    Show an animation of a single simulation run, each node
    represented (via its node label) as pixel (i,j) in an MxN image.
    So this will only be useful for grid graphs.

    Args:
        data: MxNxT array of voltage traces
        start: first timestep to show
        skip: timesteps to advance in each frame (higher -> faster)
        title: figure title
        cmap: matplotlib colormap (name OR object)
    """
    #plt.ioff()
    anim = createanim(data,start,skip,title=title,cmap=cmap, ms_per_step=ms_per_step)
    plt.show()
    plt.ion()

def saveanim(data,start=0,skip=1,filename="animation.mp4",title=None,dpi=40,cmap='bone', ms_per_step=None):
    """
    Save an animation of a single simulation run, each node
    represented (via its node label) as pixel (i,j) in an MxN image.
    So this will only be useful for grid graphs.

    Args:
        data: MxNxT array of voltage traces
        start: first timestep to show
        skip: timesteps to advance in each frame (higher -> faster)
        filename: where to save
        title: figure title
        dpi: resolution
        cmap: matplotlib colormap (name or object)
    """
    # tested with matplotlib 1.3.1 + Imagemagick 6.8.7 installed
    anim = createanim(data,start,skip,title=title,cmap=cmap, ms_per_step=ms_per_step)
    anim.save(filename, writer='ffmpeg', bitrate=10000, fps=25, dpi=dpi)

#.. and, as a late addition, the same thing again but showing the network topology:

def animate_graph(recording,network,is_measured,inputc,start,skip,stop=None,grid_as="graph",
                  title="",dontshow=False,cmap=plt.cm.bone, ms_per_step=None):
    """
    Return and display a movie where the given recording is mapped back
    onto the network graph structure, i.e. show a blinking graph.

    Args:
        recording: MxNxT array of voltage traces
        network: networkx graph
        is_measured: MxN array, nonzero positions are treated as measured nodes
        inputc: MxN array, nonzero positions are treated as input nodes
        start: first timestep to show
        skip: timesteps to advance in each frame (higher -> faster)
        grid_as: How to draw the network.
            "graph" -- confusingly, draw as a grid graph using the node labels as positions
            any networkx layout function -- draw according to that layout function
        title: figure title
        dontshow: if True, just return a matplotlib animation object. If False, do show it.
        cmap: matplotlib colormap object
        ms_per_step: if given, show time in ms

    Return:
        matplotlib animation object
    """
    # plt.ioff()
    fig = plt.figure(facecolor=(0.05,0.05,0.05),tight_layout=True,figsize=(int(recording.shape[1]/2), int(recording.shape[0]/2)))
    ax  = plt.axes(axisbg=(0.05,0.05,0.05))
    ax.set_aspect('equal', 'datalim')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    nodelist = network.nodes()
    if grid_as == "graph":
        positions = dict(zip(nodelist,[(n[1],-n[0]) for n in nodelist]))
    else:
        positions = grid_as(network)
    # apply a colormap to the normalized activation values of all nodes:
    norm = matplotlib.colors.Normalize(vmin=recording.min(),vmax=recording.max())
    nodecolors = lambda t: cmap(norm([recording[n[0],n[1],t] for n in nodelist]))
    measured_color = (0.2,0.8,0.9)#(0.2,0.8,0.4)
    roinodes   = [nd for nd in nodelist if is_measured[nd]]
    inputnodes = [nd for nd in nodelist if (inputc>0)[nd]]
    titlefont = {'color'  : 'black', 'size':12} if dontshow else {'color'  : 'white', 'size': 12}
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # http://stackoverflow.com/a/11558629
    plt.colorbar(sm, label='mV')
    recording = recording[:,:,0:stop] if stop is not None else recording
    def animate(i):
        plt.cla()
        printprogress("animating frame",start+i*skip,recording.shape[2])
        if i*skip < recording.shape[2]:
            nx.draw_networkx_edges(network,positions,edge_color='gray')
            nx.draw_networkx_nodes(network.subgraph(roinodes),positions,with_labels=False,
                                    node_size=140,node_color=[measured_color]*len(roinodes),linewidths=0)

            nx.draw_networkx_nodes(network,positions,with_labels=False,
                                    node_size=100,node_color=nodecolors(start+i*skip),linewidths=0)

            nx.draw_networkx_nodes(network.subgraph(inputnodes),positions,with_labels=False,
                                    node_size=7,node_color=[(0.8,0,0)]*len(inputnodes),linewidths=0)
            t = "  ({}ms)".format(start+i*skip * ms_per_step) if ms_per_step is not None else ""
            plt.title(title+t,titlefont)

    anim = animation.FuncAnimation(fig, animate, frames=(recording.shape[2]-start)/skip,
                                    interval=1, blit=False)
    if not dontshow: # I know..
        plt.show()
    return anim


def saveanim_graph(recording,network,is_measured,inputc,start,skip,stop=None,grid_as="graph",
                   title="",dpi=120,cmap=plt.cm.bone, filename=None, ms_per_step=None):
    """ (see animate_graph) """
    anim = animate_graph(recording, network, is_measured, inputc, start, skip, stop=stop, grid_as=grid_as,
                          title=title, dontshow=True, cmap=cmap, ms_per_step=ms_per_step)

    out_path = 'animation_graph'+str(np.random.randint(10**10))+'.mp4' if filename is None else filename
    anim.save(out_path, writer='ffmpeg', fps=25, bitrate=10000,dpi=dpi)

# --

def view_voltages(data,title=None, shuffle=False, shuffle_seed=1, vmin=-70, vmax=35, s_per_step=None):
    """
    Show a complete simulation run in an (M*N)xT trace image.

    Args:
        data: MxNxT array of voltage traces
        title: figure title
        shuffle: If true, shuffle the order of cells in the trace image.
        s_per_step: seconds per step - if given, display a proper time axis
    """
    plt.figure(figsize=fig_size)
    vtraces = data.reshape(-1,data.shape[2])[:]
    if shuffle:
        rng = RandomState(shuffle_seed)
        vtraces = rng.permutation(vtraces)

    if s_per_step is None:
        T = data.shape[-1]
    else:
        T = data.shape[-1] * s_per_step
    plt.imshow(vtraces, cmap='bone', vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest', extent=[0, T, vtraces.shape[0], 0])
    plt.colorbar()
    if title:
        plt.title(title)
    plt.show()

def view_spikes(data, title=None, shuffle=False, shuffle_seed=1):
    spikesflat = data.reshape(-1, data.shape[2])[:]
    if shuffle:
        rng = RandomState(shuffle_seed)
        spikesflat = rng.permutation(spikesflat)

    idxs, spiketimes = np.nonzero(spikesflat)
    plt.figure(figsize=fig_size)
    plt.scatter(spiketimes, idxs, marker='|', s=50, alpha=0.7, color='k')
    if title:
        plt.title(title)
    plt.show()

def plotgraph(g):
    """Plot a grid graph, using (i,j) node labels as positions"""
    plt.figure(figsize=fig_size)
    positions = dict(zip(g.nodes(),g.nodes()))
    nx.draw_networkx(g,positions,with_labels=False,node_size=50)


# plotting of measurement results:

def compare2(experiments,measurename=None,which=(0,1)):
    """ draw a scatter plot to compare trial-by-trial results of two experiments."""
    if not measurename:
        measures = set([meas for exp in experiments for meas in exp.measures])
        if len(measures) == 1:
            measurename = measures.pop()
        elif len(measures)>1:
            raise Exception("Experiments have multiple measures, need to supply a measure name.")
        else:
            raise Exception("No measures? What?")
    plt.figure(figsize=fig_size)
    dataA = experiments[which[0]].getresults(measurename)
    dataB = experiments[which[1]].getresults(measurename)

    plt.scatter(dataA,dataB)
    dataab = np.sort(dataA+dataB)
    absmin = dataab[0]
    absmax = dataab[-1]
    plt.plot([absmin,absmax], [absmin,absmax],'--')

    plt.xlabel(measurename+", "+experiments[which[0]].name)
    plt.ylabel(measurename+", "+experiments[which[1]].name)
    plt.title("trial-by-trial comparison (pairs with same initial conditions)")



def compare_windowed(experiments,measurename,plot_as="bandplot",step=10, unit=None, do_title=True):
    """ draw a band plot, showing how values from multiple experiments evolve over time."""
    plt.figure(figsize=fig_size_wide)
    # color cycle for bandplots:
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i,exp in enumerate(experiments):
        meas = exp.measures[measurename]
        if not meas.results:
            exp.run()
        data = np.array(meas.results)


        nrsteps = data.shape[1]
        # import ipdb; ipdb.set_trace()
        if meas.window == "moving":
            xaxis = np.arange(meas.wlength, nrsteps*meas.increment+meas.wlength, meas.increment)
            ticklocs = xaxis[0::nrsteps/10]
            ticklabels = np.arange(meas.wlength,nrsteps*meas.increment+meas.wlength,meas.increment*nrsteps/10)
        elif meas.window == "growing":
            xaxis = np.arange(meas.grw_startsize+meas.wait, nrsteps*meas.increment+meas.grw_startsize+meas.wait, meas.increment)
            ticklocs = xaxis[0::nrsteps/10]
            ticklabels = np.arange(meas.grw_startsize+meas.wait,nrsteps*meas.increment+meas.grw_startsize+meas.wait,meas.increment*nrsteps/10)

        if unit is not None:
            xaxis, ticklocs, ticklabels = xaxis*unit, None, None
            plt.xlabel("t")
        else:
            plt.xlabel("step")

        cl = colors[np.mod(i,len(colors))]
        if "raw" in plot_as:
            plt.plot(xaxis,data[0,:],color=cl,alpha=0.2,label=exp.name, linewidth=1)
            plt.plot(xaxis,data[1:,:].T,color=cl,alpha=0.2,label="_nolegend_", linewidth=1)
            if do_title:
                plt.title(meas.window + " window,"
                      + ("length " + str(meas.wlength) if meas.window=="moving" else ""))

        if "finebands" in plot_as:
            medianline = np.nanpercentile(data,50,axis=0)
            plt.plot(xaxis,medianline,color=cl,label=exp.name)
            for prctl in range(0,100+step,step):
                lb = prctl - step/2.0
                ub = prctl + step/2.0
                if lb < 0:
                    lb = 0
                if ub > 100:
                    ub = 100

                perc_lower  = np.nanpercentile(data,lb,  axis=0)
                perc_higher = np.nanpercentile(data,ub,axis=0)
                #print prctl, lb, ub, (50-abs(50-prctl))/50.0
                plt.fill_between(xaxis, perc_lower, perc_higher,
                                  color=cl, linewidth=0.0, alpha=(50-abs(50-prctl))/75.0)
            if do_title:
                plt.title(meas.window + " window,"
                      +("length " + str(meas.wlength) if meas.window=="moving" else "")
                      + "\nline: median, color: percentiles")

        if "bandplot" in plot_as:
            quartile0 = np.nanpercentile(data, 0, axis=0)
            quartile1 = np.nanpercentile(data, 25, axis=0)
            medianline = np.nanpercentile(data, 50, axis=0)
            quartile3 = np.nanpercentile(data, 75, axis=0)
            quartile4 = np.nanpercentile(data, 100, axis=0)

            plt.fill_between(xaxis, quartile0, quartile1, alpha=0.1, color=cl)
            plt.fill_between(xaxis, quartile1, medianline, alpha=0.5, color=cl)

            plt.plot(xaxis, medianline, color=cl, label=exp.name, alpha=0.9)

            plt.fill_between(xaxis, medianline, quartile3, alpha=0.5, color=cl)
            plt.fill_between(xaxis, quartile3, quartile4, alpha=0.1, color=cl)
            if do_title:
                plt.title(meas.window + " window,"
                      +("length " + str(meas.wlength) if meas.window=="moving" else "")
                      + "\nline: median, bands: quartiles")
    plt.legend(loc='upper right')
    plt.ylabel(measurename)
    if ticklocs is not None:
        plt.xticks(ticklocs,ticklabels)
    plt.draw()



def compare(experiments,measurename=None,grid_as="image",plot_as="scatter,boxplot",
            title=None,subgraph=False,label_stats=[],label_names=False,vrange=None, rotation=0):
    """ Plot the results from a list of experiments as a boxplot and/or scattered,
    plus a row of thumbnails that show a visual overview of each experiment.

    Args:
        experiments: list of experiment objects
        measurename: values from which type of measurement to draw (if there's just one
                     in all experiments, use that)
        grid_as: how to draw thumbnails.
            None: don't draw thumbnails
            "image": draw each grid node as a pixel using the (i,j) node labels as positions
            "graph": draw as a grid graph using the node labels as positions
            a networkx layout function: draw a graph with node positions computed by that function
            ("featurenet_stored": use the precomputed experiment.thumbnail -- weird construct used
                                  only in the weird 3D feature network case)
        subgraph:
            "input": in thumbnails, draw only the input receiving subgraph
            "input+neigh": ...plus next neighbors
            list of nodes: draw only those nodes & their connections
        plot_as: style of main plot. "scatter", "boxplot" or both
        label_stats: list of valid keys for experiment.statistics() - values appear in x tick labels
        vrange: tuple of two numbers (lower,upper), to override the y axis limits of the boxplot.
    """
    fig = plt.figure(figsize=fig_size_wide)
    # create main axes
    if grid_as:
        hmargin = 0.15  # horizontal global margin
        vmargin = 0.01 # vertical global margin
        thumb_v = 0.4 # vertical space for all thumbnails
        thumbs_vmargin = 0.1
        thumbs_hmargin = 0.01 # margin between thumbnails
        main = plt.axes((hmargin,thumb_v,1-2*hmargin,1-thumb_v-vmargin-0.01))
    else:
        main = plt.axes()
    if title:
        plt.title(title)
    # if the same thing was measured in all experiments, plot that.
    # otherwise, expect to be told which measurement to plot:
    if not measurename:
        measures = set([meas for exp in experiments for meas in exp.measures])
        if len(measures) == 1:
            measurename = measures.pop()
        elif len(measures)>1:
            raise Exception("Experiments have multiple measures, need to supply a measure name.")
        else:
            raise Exception("No measures? What?")
    data = [exp.getresults(measurename) for exp in experiments]
    # draw the main plot:
    plt.grid(linestyle='-', which='major', axis='y',color='black',alpha=0.2)
    if "boxplot" in plot_as:
        plt.boxplot(data,notch=True,boxprops={'color':'black'},flierprops={'color':'black'},
                        medianprops={'color':'red'},whiskerprops={'color':'black','linestyle':'-'})
    if "scatter" in plot_as:
        displacement = np.array([0.98+0.04*np.random.randn() for _ in range(len(data[0]))])
        for i in range(len(experiments)):
            plt.scatter(displacement+i,data[i],marker='o',s=20,alpha=0.3,c=(0,0.4,0),linewidth=0)
        plt.xlim((0,len(experiments)+1))

    labels = []
    if label_stats:
        for e in experiments:
            stat = e.statistics()
            labels.append("\n".join([e.name] + [str(s)+" "+str(round(stat[s],2)) for s in label_stats]))
    elif label_names:
        labels = [e.name for e in experiments]
    plt.xticks(range(1,len(experiments)+1),labels, rotation=rotation)
    plt.ylabel(measurename)
    plt.yticks(rotation=rotation)

    if not vrange:
        try:
            vrange = exp.measures[measurename].vrange # measure may have a vrange attribute
            plt.ylim(vrange)
        except AttributeError:
            pass
    if vrange:
        plt.ylim(vrange)

    # draw thumbnails:
    if grid_as:
        thumb_hz = (1-2*hmargin)/len(experiments) # horizontal space for each thumbnail
        norminv = lambda arr: 1-(arr / arr.max()) if arr.max() > 1 else 1-arr # scale [0,1] & invert
        for i,e in enumerate(experiments):
            # create thumbnail axes
            t_left = hmargin + i*(thumb_hz)
            t_bottom = vmargin
            t_width = thumb_hz-thumbs_hmargin
            t_height = thumb_v-thumbs_vmargin
            thumb = plt.axes((t_left,t_bottom,t_width,t_height), aspect='equal')
            # import ipdb; ipdb.set_trace()
            plt.setp(thumb, xticks=[], yticks=[])
            # finally, draw stuff into each thumbnail:
            if grid_as == "featurenet_stored": # the rare, weird case of 3D feature networks
                thumb.imshow(e.thumbnail, interpolation='nearest', vmin=0,
                              vmax=e.network.graph["original_grid3D_dimensions"][2], cmap='bone')
            else: # the usual case of 2D grid networks
                plotsetup(e.network, e.inputc, e.measures[measurename].roi,
                           axes=thumb, grid_as=grid_as, subgraph=subgraph)
            plt.show()
    return fig


def plotsetup(network,inputstrength,is_measured,axes=None,grid_as="image",subgraph=False, nodesize=0.25, edgecolor_offset=0):
    """
    Draws the network graph, with color markers for the input-receiving and the
    measured population.

    Args:
        network: networkx graph, with nodes labeled as (i,j) coordinates of an MxN grid
        inputestrength: MxN array of stimulus strength per node
        is_measured: MxN binary array, True at measured nodes
        axes: matplotlib axes in which to draw
        grid_as:
            "image": draw each grid node as a pixel using the (i,j) node labels as positions
            "graph": draw as a grid graph using the node labels as positions
            a networkx layout function: draw a graph with node positions computed by that function
        subgraph:
            "input": draw only the input receiving subgraph
            "input+neigh": ...plus next neighbors
            list of nodes: draw only those nodes & their connections
    """
    if axes is None:
        plt.figure()
        axes = plt.gca()
    if grid_as == "image":
        axes.imshow(inputstrength,interpolation='nearest',vmin=0,vmax=1.5,cmap=plt.cm.gray_r)
        axes.imshow(np.ma.masked_where(is_measured == False, np.ones(is_measured.shape)), alpha=0.2,
                    interpolation='nearest', cmap=plt.cm.hsv, vmax=3, vmin=0)

    else: #... as graph:
        if subgraph=="input":
            inputnodes = [tuple(ind) for ind in np.transpose(inputstrength.nonzero())]
            network = network.subgraph(inputnodes)
        if subgraph=="input+neigh":
            nodes = set([n for ind in np.transpose(inputstrength.nonzero())
                            for n in network.neighbors(tuple(ind))+[tuple(ind)]])
            network = network.subgraph(nodes)
        elif subgraph=='excitatory':
            exc_nodes = [n for n in network.nodes() if not network.node[n]["is_inhibitory"]]
            network = network.subgraph(exc_nodes)
        elif subgraph:
            network = network.subgraph(subgraph)

        norminv = lambda arr: 1-(arr / float(arr.max())) if arr.max() > 1 else 1-arr # scale [0,1] & invert
        # select colors for each node and edge
        nodelist = network.nodes()
        nodecolors = np.array([np.array([inputstrength[node],inputstrength[node],inputstrength[node]])
                               for node in nodelist])
        nodecolors = norminv(nodecolors)
        edges = network.edges()
        edge_strengths = np.array([network.edge[ed[0]][ed[1]]['strength']*1 for ed in edges])
        order = list(reversed(np.argsort(edge_strengths)))
        edges = [edges[order[i]] for i,_ in enumerate(edges)]
        edge_strengths = edge_strengths[order]

        is_inh = edge_strengths < 0
        is_0 = edge_strengths == 0
        edge_strengths_exc = edge_strengths.copy()
        edge_strengths_inh = -edge_strengths.copy()

        edge_strengths_exc[is_inh] = 0
        edge_strengths_inh[~is_inh] = 0


        edgecolors_exc = np.minimum(norminv(np.array([np.array([st,st,st]) for st in edge_strengths_exc])), 0.7)
        edgecolors_exc[is_0,:] = 1. # make 0 strength edges invisible

        edgecolors_inh = np.zeros_like(edgecolors_exc)
        edgecolors_inh[:, 0] = 0.7
        edgecolors_inh[:, 1] = 0.2
        edgecolors_inh[:, 2] = 0.2
        edgecolors_inh[is_0, :] = 1. # make 0 strength edges invisible

        edgecolors_exc[is_inh,:] = 0
        edgecolors_inh[~is_inh,:] = 0

        edgecolors = edgecolors_exc + edgecolors_inh
        visible = np.any(edgecolors!=1, axis=1)

        edgecolors_plot = edgecolors[visible,:] + edgecolor_offset
        edges_plot = [e for i,e in enumerate(edges) if visible[i]]


        # edgewidths = 1+np.array(edge_strengths)
        # draw edges.
        # first construct the dict of nodename:position that the nx.draw_... routines expect.
        if grid_as == "graph":
            # either built the dict from the (i,j) node names - results in a grid graph.
            # transpose and flip to get an orientation like imshow: origin in top left
            coords = []
            for nd in nodelist:
                try:
                    is_inh = network.node[nd]['is_inhibitory']
                    offset = -0.5 if is_inh else 0
                except KeyError:
                    is_inh = False
                    offset = 0
                jitter = 0.0
                coords.append((nd[1]-offset+np.random.rand()*jitter,-nd[0]-offset+np.random.rand()*jitter))

            positions = dict(zip(nodelist, coords))
        else:
            #..or assume grid_as IS a networkx layout function and use that.
            positions = grid_as(network)
            is_inh = False
        nx.draw_networkx_edges(network,positions,edgelist=edges_plot, edge_color=edgecolors_plot,ax=axes, width=0.75)
        # draw cyan measurement markers:
        measured_color = (0.2,0.6,0.9) #(0.2,0.8,0.4)q
        roinodes = [nd for nd in nodelist if is_measured[nd]]
        nx.draw_networkx_nodes(network.subgraph(roinodes), positions, ax=axes, with_labels=False,
                                node_size=100*nodesize,node_color=measured_color,linewidths=0)
        # draw node outlines
        nx.draw_networkx_nodes(network, positions, ax=axes, with_labels=False, node_size=30*nodesize,
                                node_color='k')
        # draw nodes:
        nx.draw_networkx_nodes(network, positions, ax=axes, with_labels=False, node_size=10*nodesize,
                                node_color=nodecolors)

        axes.axis('off')


def eplotsetup(experiment,measurename=None,grid_as="graph",subgraph=False):
    """same as plotsetup(), but takes a complete experiment object for convenience"""
    if measurename:
        roi = experiment.measures[measurename].roi
    else:
        measures = set(experiment.measures)
        if len(measures) == 1:
            measurename = measures.pop()
            roi = experiment.measures[measurename].roi
        elif len(measures)>1:
            raise Exception("Experiment has multiple measures, need to supply a measure name.")
        else:
            roi = np.zeros(experiment.network.graph['grid_dimensions'])

    plt.clf()
    plotsetup(experiment.network,experiment.inputc,roi,plt.axes(),grid_as=grid_as,subgraph=subgraph)


def get_reusable_springlayout(network,seed=0):
    """returns a networkx spring layout function with deterministic behaviour
    (useful to draw a series of thumbnails)"""
    rng = RandomState(seed)
    nodelist = network.nodes()
    initpos = dict(zip(nodelist,rng.rand(len(nodelist),2)))
    return lambda net:nx.spring_layout(net,pos=initpos)
