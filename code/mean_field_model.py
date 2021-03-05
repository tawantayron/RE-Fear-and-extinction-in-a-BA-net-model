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
# Script to simulate and run the mean-field model.
# ----------------------------------------------------------------------------
import numpy             as np 
import matplotlib.pyplot as plt
import os
plt.rcParams.update({'font.size': 15})
#############################################################################
# Defining model equations
############################################################################
def activation_function(x):
	r''' Sigmoid used as activation function for the rate model (Eq. 1 and 2 from ref).
   	Input:
   	x: value in which the function is computed on.
   	Output:
   	f(x)
   	'''
	return 1.0 / (1.0 + np.exp(-1.2*(x-2.8)) )

def rate_equation(R,k,r,tau,inputs,mu,std):
	r'''
	Equation for the rate model (Eq. 1 and 2 in ref. paper)
	'''
	return ( -R + np.random.normal(mu, std, size = 1) + (k - r * R) * activation_function(inputs) ) / tau

def weight_equation(alpha, CS, CTX):
	r'''
	Equation for learning rule (Eq. 3 and 4 in ref. paper)
	Input:
	alpha: Learning rate
	CS: CS stimulus array.
	CTX: CTX stimulus array.
	'''
	return alpha * CS * CTX

##########################################################################
# Defining input generators
##########################################################################

def gen_inputs(Tsim, dt, dur, t_off_a, t_on_b):
	r'''
	Generate the weights used to stimulate the rate neuron model.
	Inputs:
	Tsim: Simulation time.
	dt: Time resolution of the simulation.
	dur: Duration of each CS activation.
	t_off_a: Time when CS_a is turned off.
	t_on_b: Time when CS_b is turned on.
	Outputs:
	m: Mask indicating the time in which CS is active.
	CS and CTX time series.
	'''
	starting_times_cs = (np.arange(500, 1350, 200)/dt).astype(int)
	starting_times_cs = np.append( starting_times_cs, (np.arange(1550, 2600, 200)/dt).astype(int) ) 

	cs = np.zeros(int(Tsim/dt))
	for t0 in starting_times_cs:
		cs[t0:t0+int(dur/dt)] = 1

	CTX = np.zeros([2,int(Tsim/dt)])
	CTX[0,:int(t_off_a/dt)] = 1
	CTX[1,int(t_on_b/dt):] = 1
	return cs*0.5, CTX*0.3 

Tsim    = 3000 # Simulation time.
dt      = 0.1  # Simulation time resolution.
dur     = 50   # Duration of each CS activation.
t_off_a = 1400 # Time when CS_a is turned off.
t_on_b  = 1450 # Time when CS_b is turned on.
tau     = 10   # Time constant
Npop    = 2    # Size of the population
alpha   = .15  # Learning rate
w       = -1   # Inter-population weight
w_ctx   = 1    # CTX weight
k       = 0.97 # 
r       = 1e-3 #
w_cs = np.ones([Npop, int(Tsim/dt)])                  # CS weight
R       = np.zeros([Npop,int(Tsim/dt)])               # Rate array
T       = np.arange(0, Tsim, dt)                      # Time array
CS, CTX = gen_inputs(Tsim, dt, dur, t_off_a, t_on_b)  # CS/CTX array
###############################################################################################
# Simulating the rate based model
###############################################################################################
for t in range(T.shape[0]-1):
    for i in range(Npop):
        inputs = R[np.abs(i-1), t] * w + w_cs[np.abs(i-1), t] * CS[t] + w_ctx * CTX[np.abs(i-1),t] 
        R[i,t+1] =R[i,t]+dt*rate_equation(R[i,t],k,r,tau,inputs,0.02,0.002)    
        w_cs[np.abs(i-1),t+1] = w_cs[np.abs(i-1),t] + dt * alpha * CS[t] * CTX[np.abs(i-1),t]
###############################################################################################
# Compute the mean rate in a windowed fashion (as in the ref. paper)
###############################################################################################
def compute_mean_rate(rates, m, window_length = 500):
	idx      = np.nonzero(m)[0]
	Nwindows = int( len(idx) / window_length )
	r1 = rates[0, idx].reshape((Nwindows, window_length))
	r2 = rates[1, idx].reshape((Nwindows, window_length))

	return r1.mean(axis=1), r2.mean(axis=1)
# Computing the windowed-mean rate
rmean = compute_mean_rate(R,CS)

os.system('mkdir mean_field_model')
###############################################################################################
# Plotting the results
###############################################################################################
plt.figure(figsize=(12,6))
plt.subplot2grid((3,2), (0,0), rowspan = 3)
plt.plot(range(1, len(rmean[0])+1),rmean[1],'s',label=r'$pop_A$',ms=9,color=plt.cm.tab10(1))
plt.plot(range(1, len(rmean[0])+1),rmean[0],'o',label=r'$pop_B$',ms=9,color=plt.cm.tab10(0))
plt.legend()
plt.ylabel('Population rates [Hz]')
plt.xlabel('CS appication number')
plt.ylim([0.03,0.6])
plt.vlines(5.5, 0.03, 0.6, linestyle='--', color = 'k')
plt.text(-1, 0.58,"A", weight="bold", fontsize=18)
plt.subplot2grid((3,2), (0,1), rowspan = 2)
plt.plot(T, w_cs[1])
plt.plot(T, w_cs[0])
plt.ylabel(r'$w_{CS}$')
plt.xticks([])
plt.text(-500, 7.7,"B", weight="bold", fontsize=18)
plt.subplot2grid((3,2),(2,1) , rowspan =1)
plt.plot(T,CS,label=r'$CS$',c='k')
plt.plot(T,CTX[0],label=r'$CTX_{A}$',c=plt.cm.tab10(1))
plt.plot(T,CTX[1],label=r'$CTX_{B}$',c=plt.cm.tab10(0))
plt.xlabel('Time [ms]')
plt.legend(prop={'size': 12})
plt.text(-500, 0.45,"C", weight="bold", fontsize=18)
plt.tight_layout()
plt.savefig('mean_field_model/result_mean_field_model.png', dpi = 200)
plt.show()
