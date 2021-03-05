# ----------------------------------------------------------------------------
# Contributors: Tawan T. A. Carvalho
#               Luana B. Domingos
#               Renan O. Shimoura
#               Nilton L. Kamiji
#               Vinicius Lima
#               Mauro Copelli
#               Antonio C. Roque
# ----------------------------------------------------------------------------
# References:
#
# *Context-dependent encoding of fear and extinctionmemories in a large-scale
# network model of the basal amygdala*,
# I. Vlachos, C. Herry, A. Lüthi, A. Aertsen and A. Kumar,
# PLoS Comput. Biol. 7 (2011), e1001104.
# ----------------------------------------------------------------------------
# File description:
#
# This code contains the main function to create the spiking network structure
# and the synaptic and external input connections. It also contains the auxiliary
# functions to calculate the synaptic weight normalization constant and to smooth
# curves in graphs.
# ----------------------------------------------------------------------------


import numpy as np
from scipy import signal

from params     import *
from models_eq  import *

#############################################################################
# Beta function normalization factor
#############################################################################
# function to calculate normalization factor for beta function
# if tau_rise == tau_decay, the returned normalization factor is the same as that for alpha function!
def beta_normalization_factor(tau_rise, tau_decay):
    numeric_limit = 1e-16*ms
    # difference between rise and decay time constants; used to determine whether use alpha or beta functions
    tau_diff = tau_decay - tau_rise

    # auxiliary parameters for beta function
    peak_value = 0.0

    if abs(tau_diff) > numeric_limit:  # tau_rise != tau_decay; use beta function
        # time to peak
        t_peak = tau_decay * tau_rise * np.log( tau_decay / tau_rise ) / tau_diff
        # peak_value of the beta function (difference of exponentials)
        peak_value = np.exp( -t_peak / tau_decay ) - np.exp( -t_peak / tau_rise )

    if abs(peak_value) < numeric_limit/ms: # tau_rise == tau_decay; use alpha function
        normalization_factor = np.exp(1) / tau_decay
    else: # tau_rise != tau_decay; use beta function
        normalization_factor = (1. / tau_rise - 1. / tau_decay ) / peak_value

    #############################################################################
    return normalization_factor

Gexc_0 = beta_normalization_factor(tauexc_rise, tauexc_decay)
Ginh_0 = beta_normalization_factor(tauinh_rise, tauinh_decay)

#############################################################################
# Network structure with all inputs connected
#############################################################################
def amygdala_net(input=False, input_vars=input_vars, pcon=pcon, wsyn=wsyn, sdel=sdelay,record_weights=True):

    #############################################################################
    # Creating neuron nodes
    #############################################################################
    neurons = NeuronGroup(NE+NI, eq_LIF, threshold='v>Vt', reset=reset_LIF, refractory=tref, method='rk4')
    neurons.v    = 'E0 + randn()*3.0*mV' #initial condition
    neurons.wcs  = wcs
    neurons.wctx = wctx

    pop = []
    pop.append(neurons[0:NE]) #excitatory neurons
    pop.append(neurons[NE:])  #inhibitory neurons

    pop_A = pop[0][:NA]  #subneuronsulation A - 20% of excitatory neurons
    pop_B = pop[0][-NB:] #subneuronsulation B - 20% of excitatory neurons

    #############################################################################
    # Creating synapse connections
    #############################################################################
    # # normalization factor of synaptic function
    G_0    = [Gexc_0, Ginh_0]

    conn = [] # Stores connections
    for pre in range(0,2):
        for post in range(0,2):
            ws  = wsyn[pre][post]
            g_0 = G_0[pre]

            conn.append(Synapses(pop[pre], pop[post], model = syn_model, on_pre=pre_eq[pre]))
            conn[-1].connect(condition='i!=j', p=pcon[pre][post])
            conn[-1].w     = '(randn()*0.1*nS + ws)'
            conn[-1].delay = sdel[pre][post]

    ###########################################################################
	# Creating poissonian background inputs
	###########################################################################
    Pe = PoissonInput(pop[0], 'Gexc_aux', 1000, rate_E, weight=w_e*Gexc_0)
    Pi = PoissonInput(pop[1], 'Gexc_aux', 1000, rate_I, weight=w_e*Gexc_0)

    if input==True:
        ###########################################################################
    	# Creating CS and CTX inputs
    	###########################################################################
        #initially the inputs are not active.
        PG_cs    = PoissonGroup(len(neurons), rates = input_vars['cs_rate'])
        PG_ctx_A = PoissonGroup(len(pop_A), rates = input_vars['ctxA_rate'])
        PG_ctx_B = PoissonGroup(len(pop_B), rates = input_vars['ctxB_rate'])

        ###########################################################################
    	# Connecting CS and CTX to neuron populations
    	###########################################################################
        #connecting CS to all excitatory neurons using the plasticity rule
        CS_e = Synapses(PG_cs[:NE], pop[0], model = syn_plast, on_pre=pre_cs)
        CS_e.connect(j='i')
        # CS_e.m = input_vars['mt_array']

        #connecting CS to all inhibitory neurons using static synapses
        CS_i = Synapses(PG_cs[NE:], pop[1], model = syn_model, on_pre = pre_exc)
        CS_i.connect(j='i')
        CS_i.w = 'randn()*0.1*nS + 0.9*nS'

        #Context A connected with subpopulation A using synaptic plasticity
        CTX_A = Synapses(PG_ctx_A, pop_A, model = syn_plast, on_pre=pre_ctx)
        CTX_A.connect(j='i')
        # CTX_A.m = input_vars['mt_array']

        #Context B connected with subpopulation B using synaptic plasticity
        CTX_B = Synapses(PG_ctx_B, pop_B, model = syn_plast, on_pre=pre_ctx)
        CTX_B.connect(j='i')
        # CTX_B.m = input_vars['mt_array']

    else:
         PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B = [],[],[],[],[],[],[]

	###########################################################################
	# Creating monitors
	###########################################################################
    spikemon_ne = SpikeMonitor(pop[0], record=True)
    spikemon_ni = SpikeMonitor(pop[1], record=True)
    spikemon_A  = SpikeMonitor(pop_A, record=True)
    spikemon_B  = SpikeMonitor(pop_B, record=True)

    spikemon = [spikemon_ne, spikemon_ni, spikemon_A, spikemon_B]

    statemon_CS    = StateMonitor(pop[0], 'wcs', record=record_weights)
    statemon_CTX_A = StateMonitor(pop_A, 'wctx', record=record_weights)
    statemon_CTX_B = StateMonitor(pop_B, 'wctx', record=record_weights)

    statemon = [statemon_CS, statemon_CTX_A, statemon_CTX_B]

	###########################################################################
	# Running simulation
	###########################################################################
    net = Network(collect())
    net.add(neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B)
    return(net, neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B)

#############################################################################
# Function to smooth curves in graphs
#############################################################################
def matlab_smooth(data, window_size):
    # assumes the data is one dimensional
    n = data.shape[0]
    c = signal.lfilter(np.ones(window_size)/window_size, 1, data)
    idx_begin = range(0, window_size - 2)
    cbegin = data[idx_begin].cumsum()
    # select every second elemeent and divide by their index
    cbegin = cbegin[0::2] / range(1, window_size - 1, 2)
    # select the list backwards
    idx_end = range(n-1, n-window_size + 1, -1)
    cend = data[idx_end].cumsum()
    # select every other element until the end backwards
    cend = cend[-1::-2] / (range(window_size - 2, 0, -2))
    c = np.concatenate([cbegin, c[window_size-1:], cend])
    return c
