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
# Main parameter values.
# ----------------------------------------------------------------------------

from brian2 import *

'''
Define network parameters
'''

#############################################################################
# Network parameters
#############################################################################
NE = 3400   #excitatory neurons
NI = 600    #inhibitory neurons

NA = NB = int(NE*0.2) #number of excitatory neurons divided in subpopulation

#connection probabilities
pcon =  [[0.01,     # excitatory to excitatory
          0.15],    # excitatory to inhibitory
         [0.15,     # inhibitory to excitatory
          0.10]]    # inhibitory to inhibitory

#############################################################################
# Neuron parameters
#############################################################################
Vt      = -50.0*mV         # threshold
tref    = 2.0*ms           # refractory time
Ek      = -70.0*mV         # reset potential
E0      = -70.0*mV         # resting potential
Eexc    = 0.0*mV           # reversal potential for excitatory synapses
Einh    = -80.0*mV         # reversal potential for inhibitory synapses
taum    = 15.0*ms          # membrane time constant
Cm      = 250.0*pF         # membrane capacitance
Gl      = 16.7*nS          # leakage conductance

#############################################################################
# Synapse parameters
#############################################################################]
tauexc_rise  = 0.326*ms # excitatory rise time constant
tauexc_decay = 0.326*ms # excitatory decay time constant
tauinh_rise  = 0.326*ms # inhibitory rise time constant
tauinh_decay = 0.326*ms # inhibitory decay time constant

# synaptic weights (in nS)
wsyn = [[1.25*nS,  #wee: excitatory to excitatory
         1.25*nS], #wei: excitatory to inhibitory
        [2.5*nS,   #wie: inhibitory to excitatory
         2.5*nS]]  #wii: inhibitory to inhibitory

wcs  = 'randn()*0.1*nS + 0.9*nS'    #from CS to all neurons
wctx = 'randn()*0.05*nS + 0.4*nS'   #from CTX to all neurons

# synaptic delay
sdelay = [['(randn()*0.1 + 2.0)*ms',
           '(randn()*0.1 + 2.0)*ms'],
          ['(randn()*0.1 + 2.0)*ms',
           '(randn()*0.1 + 2.0)*ms']]

# synaptic plasticity parameters
tauh = 10.0*ms
tauc = 10.0*ms
w_min = 0.4*nS
w_max = 4.0*nS
alpha = 1.6e-3
c_u = 0.35
h_u = 0.35

#############################################################################
# Poisson background input parameters
#############################################################################
w_e     = 1.25*nS   # synaptic weight
rate_E  = 5.0*Hz    # Poisson spiking firing rate to excitatory neurons
rate_I  = 6.0*Hz    # Poisson spiking firing rate to inhibitory neurons

#############################################################################
# Defining input parameters
#############################################################################
fCS			= 500.0			# CS firing rate
fCTX		= 300.0			# CTX firing rate

nCSA 		= 5				# Number of CS presentations to population A
nCSB 		= 6				# Number of CS presentations to population B
tCS_dur  	= 50.0			# CS duration in ms
tCS_off 	= 150.0			# Time in ms between two consecutive CS presentation

tCTXA_dur = nCSA*(tCS_dur+tCS_off)	# CTX_A duration in ms
tCTXB_dur = nCSB*(tCS_dur+tCS_off)	# CTX_B duration in ms
tCTX_off  = 100.0					# Time with both CTX turned off

tinit	 = 100.0									# Initial time for transient
tsim 	 = tinit + tCTXA_dur + tCTX_off + tCTXB_dur	# Total time of simulation
delta_tr = 0.1                						# Temporal resolution (ms)
nbins    = int(tsim/delta_tr)        				# Length of simulation (bins)
tstim	 = np.arange(0.0, tsim, delta_tr)			# Times discretized

input_vars={
            'cs_rate'  : 0.0*Hz,
            'ctxA_rate': 0.0*Hz,
            'ctxB_rate': 0.0*Hz,
            }

#############################################################################
# Defining simulation parameters
#############################################################################
defaultclock.dt =  delta_tr*ms                      # time step for numerical integration

#default settings for plotting
rcParams["figure.figsize"] = [10,6]
rcParams.update({'font.size': 18})
