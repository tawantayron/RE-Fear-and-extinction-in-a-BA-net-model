 # ----------------------------------------------------------------------------
# Contributors: Tawan T. A. Carvalho
#               Luana B. Domingos
#               Renan O. Shimoura
#               Nilton L. Kamiji
#               Vinicius Lima
#               Mauro COpelli
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
# Definition of the protocols used to reproduce the results for the spiking
# model.
# ----------------------------------------------------------------------------

from params     import *
from amygdala   import *

import sys
import os
from   joblib           import Parallel, delayed
import multiprocessing as mp
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

protocol = int(sys.argv[1])

seed(100) # seed of the random number generator


###############################################################################
# Simulation protocols
###############################################################################



'''

protocol = 0:   simulation of spontaneous network activity (Figure 4)

protocol = 1:   dynamics of conditioning and extinction processes (Figures 5 and 6)

protocol = 2:   fear renewal (Figures 7 and 8)

protocol = 3:   gamma oscillations for high network connectivity (Figure 9)

protocol = 4:   effects of connectivity, synaptic weights and delays of the
                inhibitory population on synchronization (Figures 10 and 11)
                
protocol = 5:   blockage of inhibition (Figure 12 and 13)


'''
# protocol 0: spontaneous activity
if protocol == 0:
    os.system('mkdir spontaneous_activity')
    tsim = 1000.0
    net, neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B = amygdala_net(input=False)
    net.run(tsim*ms, report='stdout')

    spikemon_ne, spikemon_ni, spikemon_A, spikemon_B = [spikemon[0],spikemon[1],spikemon[2],spikemon[3]]
    statemon_CS, statemon_CTX_A, statemon_CTX_B = [statemon[0], statemon[1], statemon[2]]

    # firing rate
    f_e = spikemon_ne.num_spikes/(NE*tsim/1000.0)*Hz
    f_i = spikemon_ni.num_spikes/(NI*tsim/1000.0)*Hz

    plt.figure(figsize=(15,7))
    plt.plot(spikemon_ne.t/ms, spikemon_ne.i, 'k.', ms=2) #all excitatory exneurons
    plt.plot(spikemon_ni.t/ms, spikemon_ni.i + NE, 'r.', ms=2) #inhibitory neurons
    #plt.title('f_e=' + str('%.2f' % f_e) + ', f_i=' + str('%.2f' % f_i))
    plt.title(r"$f_{E}$" + "= {:.2f} Hz, ".format(f_e) + r"$f_{I}$" + "= {:.2f} Hz".format(f_i))
    plt.ylabel('# Neuron')
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.savefig("spontaneous_activity/BKG_raster_plot.png", dpi = 200)
    plt.show()

    # saving files with spike times
    data = open("spontaneous_activity/spike_times_NE.txt", "w+")
    for j in range(len(spikemon_ne.i)):
        data.write(str((spikemon_ne.t/ms)[j]) + ' ' + str(spikemon_ne.i[j]) + '\n')
    data.close()
    data = open("spontaneous_activity/spike_times_NI.txt", "w+")
    for j in range(len(spikemon_ni.i)):
        data.write(str((spikemon_ni.t/ms)[j]) + ' ' + str(spikemon_ni.i[j] + NE) + '\n')
    data.close()
    
    
#protocol 1: Standard simulation, objective: to qualitatively reproduce Figure 5 of the original paper
elif protocol == 1:
    os.system('mkdir conditioning_extinction')
    aux	= np.zeros(int((tCS_dur+tCS_off)/delta_tr))
    aux[:int(tCS_dur/delta_tr)] = 1
    m_array = np.concatenate([np.zeros(int(tinit/delta_tr)),np.tile(aux, nCSA),\
                        np.zeros(int(tinit/delta_tr)),np.tile(aux, nCSB)])

    mt       = TimedArray(m_array, dt=delta_tr*ms)
    stimulus = TimedArray(m_array*fCS*Hz, dt=delta_tr*ms)

    new_input_vars={
                'cs_rate'  : 'stimulus(t)',
                'ctxA_rate': '(t<=' + str(tinit+tCTXA_dur) + '*ms)*' + str(fCTX) + '*Hz',
                'ctxB_rate': '(t>=' + str(tinit+tCTXA_dur+tinit) + '*ms)*' + str(fCTX) + '*Hz',
                # 'mt_array' : 'mt(t)'
                }
    input_vars.update(new_input_vars)


    n_simulations = 30 #total of simulations

    def conditioning_extinction_simulations(l):
        seed(100+l)
        print("Running simulations: please wait")
        print("Simulation ID: {}".format(l+1))
        start_scope()
        net, neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B = amygdala_net(input=True, input_vars=input_vars)
        net.run(tsim*ms, report='stdout')

        spikemon_ne, spikemon_ni, spikemon_A, spikemon_B = [spikemon[0],spikemon[1],spikemon[2],spikemon[3]]
        statemon_CS, statemon_CTX_A, statemon_CTX_B = [statemon[0], statemon[1], statemon[2]]

        ###########################################################################
        # Results
        ###########################################################################

        nonzero_id = np.nonzero(m_array)
        winsize  = int(tCS_dur/delta_tr)

        ind  = 0
        cs_intervals = []
        for i in range(nCSA+nCSB):
            cs_intervals.append([tstim[nonzero_id][ind],tstim[nonzero_id][ind+winsize-1]])
            ind+=winsize


        ###########################################################################
        # Average firing rate for each subpopulation
        ###########################################################################
        print("Calculating the average firing rate for each subpopulation")

        timesA = spikemon_A.t/ms
        timesB = spikemon_B.t/ms

        fr_A = []
        fr_B = []

        for i, j in cs_intervals:
            fr_A.append(np.sum((timesA>=i) & (timesA<=j))/(NA*tCS_dur/1000.0))
            fr_B.append(np.sum((timesB>=i) & (timesB<=j))/(NB*tCS_dur/1000.0))

        ###########################################################################
        # Average of CS and CTX weights
        ###########################################################################
        print("Calculating the average CS and CTX weights")

        wCS_A = np.array(statemon_CS.wcs[:NA])
        wCS_B = np.array(statemon_CS.wcs[-NB:])

        wCS_A = wCS_A.mean(axis=0)
        wCS_B = wCS_B.mean(axis=0)

        wCS_A = wCS_A[nonzero_id]
        wCS_B = wCS_B[nonzero_id]

        #CTX
        wCTX_A = np.array(statemon_CTX_A.wctx)
        wCTX_B = np.array(statemon_CTX_B.wctx)

        wCTX_A = wCTX_A.mean(axis=0)
        wCTX_B = wCTX_B.mean(axis=0)

        wCTX_A = wCTX_A[nonzero_id]
        wCTX_B = wCTX_B[nonzero_id]

        CS_A = []
        CS_B = []
        CTX_A = []
        CTX_B = []
        aux = 0

        for i in range(nCSA+nCSB):
            CS_A.append(wCS_A[aux:aux+winsize].mean())
            CS_B.append(wCS_B[aux:aux+winsize].mean())

            CTX_A.append(wCTX_A[aux:aux+winsize].mean())
            CTX_B.append(wCTX_B[aux:aux+winsize].mean())

            aux+=winsize
        
        #saving the data from the last simulation
        if((l+1)==30):
            np.save('conditioning_extinction/last_simulation_data.npy', \
                    {'spk_ni_t': spikemon_ni.t/ms,
                    'spk_ne_t': spikemon_ne.t/ms,
                    'spk_A': spikemon_A.t/ms,
                    'spk_B': spikemon_B.t/ms,
                    'ID_ni': np.array(spikemon_ni.i),
                    'ID_ne': np.array(spikemon_ne.i),
                    'ID_A': np.array(spikemon_A.i),
                    'ID_B': np.array(spikemon_B.i),
                    'cs_intervals': np.array(cs_intervals)})
        
        return(fr_A, fr_B, CS_A/nS, CS_B/nS, CTX_A/nS, CTX_B/nS)

    processing_simulations = mp.Pool(3)
    results = processing_simulations.map(conditioning_extinction_simulations, range(n_simulations))

    #loading the data from the last simulation
    data = np.load('conditioning_extinction/last_simulation_data.npy', allow_pickle=True)
    spk_ni = data.item().get('spk_ni_t')
    spk_ne = data.item().get('spk_ne_t')
    spk_A = data.item().get('spk_A')
    spk_B = data.item().get('spk_B')
    ID_ni = data.item().get('ID_ni')
    ID_ne = data.item().get('ID_ne')
    ID_A = data.item().get('ID_A')
    ID_B = data.item().get('ID_B')
    cs_intervals = data.item().get('cs_intervals')

    bin_f = 15 #time bin to calculate the time series of the trigger rate of the last simulation.
    fr_IH_last = np.histogram(spk_ni, bins=np.arange(0, tsim, bin_f))
    fr_A_last = np.histogram(spk_A, bins=np.arange(0, tsim, bin_f))
    fr_B_last = np.histogram(spk_B, bins=np.arange(0, tsim, bin_f))


    ###########################################################################
    # Final results
    ###########################################################################

    # Raster plot for the last simulation
    fig = plt.figure(constrained_layout=False, figsize=(15,10))
    gs = fig.add_gridspec(13,1)

    ax = fig.add_subplot(gs[0:7, 0])
    ax.plot(spk_ne, ID_ne, 'k.', ms=2) #all excitatory exneurons
    ax.plot(spk_ni, ID_ni + 3400, 'r.', ms=2) #inhibitory neurons
    ax.plot(spk_A, ID_A, '.', ms=4, color="tab:orange") #sub-pop A - fear neurons
    ax.plot(spk_B, ID_B + 2720 , '.', color="tab:blue", ms=4) #sub-pop B - extinction neurons

    for i,j in cs_intervals:
        ax.plot([i,j],[-100,-100],'k-', lw = 3)
    # plt.plot(tstim[nonzero_id],-100.0*np.ones(size(nonzero_id)),'.k')

    ax.text(350, -350, 'CONDITIONING')
    ax.text(1650, -350, 'EXTINCTION')
    ax.axvline(tinit + tCTXA_dur, ls='--', lw=2, color='black')
    ax.set_ylim(-400,NE+NI)
    ax.set_xlim(50,tsim)
    ax.set_ylabel('# Neuron')
    ax.set_xlabel('Time (ms)')
    ax.text(-200, 3800,"A", weight="bold", fontsize=30)

    ax.get_xaxis().set_visible(False)

    ax = fig.add_subplot(gs[7:10, 0])

    ax.plot(fr_B_last[1][:-1], matlab_smooth(fr_B_last[0]*1000/(bin_f*NI), 5), lw=2)
    ax.plot(fr_A_last[1][:-1], matlab_smooth(fr_A_last[0]*1000/(bin_f*NI), 5), lw=2)

    for i,j in cs_intervals:
        ax.plot([i,j],[-0.5,-0.5],'k-', lw = 3)
    ax.axvline(tinit + tCTXA_dur, ls='--', lw=2, color='black')
    ax.set_xlim(50,tsim)
    ax.set_ylim(-0.7,4.5)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (ms)")
    ax.text(-200, 3.8,"B", weight="bold", fontsize=30)
    ax.get_xaxis().set_visible(False)

    ax = fig.add_subplot(gs[10:, 0])
    #ax.plot(fr_BB[1][:-1], fr_BB[0]*1000/(bin_f*NI),lw=2)
    #ax.plot(fr_AA[1][:-1], fr_AA[0]*1000/(bin_f*NI),lw=2)
    #ax.plot(fr_IH[1][:-1], fr_IH[0]*1000/(bin_f*NI), 'r-')
    ax.plot(fr_IH_last[1][:-1], matlab_smooth(fr_IH_last[0]*1000/(bin_f*NI), 5), 'r-')
    #plt.plot(fr_EX[1][:-1], fr_EX[0]*1000/(bin_f*NI), 'b-')

    for i,j in cs_intervals:
        ax.plot([i,j],[-2,-2],'k-', lw = 3)
    ax.axvline(tinit + tCTXA_dur, ls='--', lw=2, color='black')
    ax.set_ylim(-3,20)
    ax.set_xlim(50,tsim)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (ms)")
    #ax.get_xaxis().set_visible(False)
    ax.text(-200, 17,"C", weight="bold", fontsize=30)
    plt.tight_layout()
    plt.savefig('conditioning_extinction/raster.png', dpi = 200)
    #plt.show()

    ###########################################################################
    # Average firing rate,CS and CTX for each subpopulation
    ###########################################################################
    n_CS = 11 #total CS presentation
    fig = plt.figure(constrained_layout=True, figsize=(7,10))
    gs = fig.add_gridspec(6,1)

    ax = fig.add_subplot(gs[0:2, 0])
    ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[0], yerr=np.std(results, axis=0)[0], fmt='s', ms=10, capsize=3, label = r"$pop_A$", color=plt.cm.tab10(1))
    ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[1], yerr=np.std(results, axis=0)[1], fmt='o', ms=10, capsize=3, label = r"$pop_B$", color=plt.cm.tab10(0))
    ax.axvline(5.5, ls='--', lw=2, color='black')
    ax.set_ylim(-0.2,5)
    ax.set_xticks(range(n_CS+1))
    ax.text(1.2, 4.5, 'CONDITIONING')
    ax.text(7, 4.5, 'EXTINCTION')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_xlabel('CS presentations')
    legend = ax.legend(bbox_to_anchor=(0.25,0.87), fontsize=15)
    legend.get_frame().set_alpha(0)
    ax.get_xaxis().set_visible(False)
    ax.text(-2, 4.4,"A", weight="bold", fontsize=30)


    ax = fig.add_subplot(gs[2:4, 0])
    ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[3], yerr=np.std(results, axis=0)[3], fmt='o', ms=10, capsize=3)
    ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[2], yerr=np.std(results, axis=0)[2], fmt='s', ms=10, capsize=3)
    ax.axvline(5.5, ls='--', lw=2, color='black')
    ax.set_xticks(range(n_CS+1))
    ax.set_ylabel('CS weights (nS)')
    ax.set_xlabel('CS presentations')
    ax.set_ylim(0,3)
    ax.get_xaxis().set_visible(False)
    ax.text(-2, 2.7,"B", weight="bold", fontsize=30)


    ax = fig.add_subplot(gs[4:6, 0])
    ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[5], yerr=np.std(results, axis=0)[5], fmt='o', ms=10, capsize=3)
    ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[4], yerr=np.std(results, axis=0)[4], fmt='s', ms=10, capsize=3)
    ax.axvline(5.5, ls='--', lw=2, color='black')
    ax.set_xticks(range(n_CS+1))
    ax.set_ylabel('HPC weights (nS)')
    ax.set_xlabel('CS presentations')
    ax.text(-2, 2.7,"C", weight="bold", fontsize=30)
    ax.set_ylim(0,3)

    plt.savefig('conditioning_extinction/avarages.png', dpi = 200)
    plt.show()

#protocol 2: Renewal of fear, objective: to qualitatively reproduce Figure 7 of the original paper
elif protocol == 2:
    os.system('mkdir renewal_fear')
    aux	= np.zeros(int((tCS_dur+tCS_off)/delta_tr))
    aux[:int(tCS_dur/delta_tr)] = 1
    m_array = np.concatenate([np.zeros(int(tinit/delta_tr)),np.tile(aux, nCSA),\
                            np.zeros(int(tCTX_off/delta_tr)),np.tile(aux, nCSB),\
                            np.zeros(int(tCTX_off/delta_tr)),np.tile(aux, 1)])

    mt       = TimedArray(m_array, dt=delta_tr*ms)
    stimulus = TimedArray(m_array*fCS*Hz, dt=delta_tr*ms)

    t1 = tinit+tCTXA_dur
    t2 = t1+tCTX_off
    t3 = t2+tCTXB_dur+tCTX_off
    new_input_vars={
                'cs_rate'  : 'stimulus(t)',
                'ctxA_rate': '((t>='+str(tinit)+'*ms)*(t<='+str(t1)+'*ms)+(t>='+str(t3)+'*ms))*'+str(fCTX)+'*Hz',
                'ctxB_rate': '((t>='+str(t2)+'*ms)*(t<='+str(t2+tCTXB_dur)+'*ms))*'+str(fCTX)+'*Hz'
                }
    input_vars.update(new_input_vars)
    tsim = t3 + tCS_dur + tCS_off
    tstim= np.arange(0.0, tsim, delta_tr)			# Times discretized

    n_simulations = 30 #total of simulations

    def renewal_fear_simulations(l):
        seed(100+l)
        print("Running simulations: please wait")
        print("Simulation ID: {}".format(l+1))
        start_scope()
        net, neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B = amygdala_net(input=True, input_vars=input_vars)
        net.run(tsim*ms, report='stdout')

        spikemon_ne, spikemon_ni, spikemon_A, spikemon_B = [spikemon[0],spikemon[1],spikemon[2],spikemon[3]]
        statemon_CS, statemon_CTX_A, statemon_CTX_B = [statemon[0], statemon[1], statemon[2]]

        ###########################################################################
        # Results
        ###########################################################################
        nonzero_id = np.nonzero(m_array)
        winsize  = int(tCS_dur/delta_tr)

        ind  = 0
        cs_intervals = []
        for i in range(nCSA+nCSB+1):
            cs_intervals.append([tstim[nonzero_id][ind],tstim[nonzero_id][ind+winsize-1]])
            ind+=winsize


        ###########################################################################
        # Average firing rate for each subpopulation
        ###########################################################################
        print("Calculating the average firing rate for each subpopulation")

        timesA = spikemon_A.t/ms
        timesB = spikemon_B.t/ms

        fr_A = []
        fr_B = []

        for i, j in cs_intervals:
            fr_A.append(np.sum((timesA>=i) & (timesA<=j))/(NA*tCS_dur/1000.0))
            fr_B.append(np.sum((timesB>=i) & (timesB<=j))/(NB*tCS_dur/1000.0))

        ###########################################################################
        # Average of CS and CTX weights
        ###########################################################################
        print("Calculating the average CS and CTX weights")

        wCS_A = np.array(statemon_CS.wcs[:NA])
        wCS_B = np.array(statemon_CS.wcs[-NB:])

        wCS_A = wCS_A.mean(axis=0)
        wCS_B = wCS_B.mean(axis=0)

        wCS_A = wCS_A[nonzero_id]
        wCS_B = wCS_B[nonzero_id]

        #CTX
        wCTX_A = np.array(statemon_CTX_A.wctx)
        wCTX_B = np.array(statemon_CTX_B.wctx)

        wCTX_A = wCTX_A.mean(axis=0)
        wCTX_B = wCTX_B.mean(axis=0)

        wCTX_A = wCTX_A[nonzero_id]
        wCTX_B = wCTX_B[nonzero_id]

        CS_A = []
        CS_B = []
        CTX_A = []
        CTX_B = []
        aux = 0

        for i in range(nCSA+nCSB+1):
            CS_A.append(wCS_A[aux:aux+winsize].mean())
            CS_B.append(wCS_B[aux:aux+winsize].mean())

            CTX_A.append(wCTX_A[aux:aux+winsize].mean())
            CTX_B.append(wCTX_B[aux:aux+winsize].mean())

            aux+=winsize

        #saving the data from the last simulation
        if((l+1)==30):
            np.save('renewal_fear/last_simulation_data.npy', \
                    {'spk_ni_t': spikemon_ni.t/ms,
                    'spk_ne_t': spikemon_ne.t/ms,
                    'spk_A': spikemon_A.t/ms,
                    'spk_B': spikemon_B.t/ms,
                    'ID_ni': np.array(spikemon_ni.i),
                    'ID_ne': np.array(spikemon_ne.i),
                    'ID_A': np.array(spikemon_A.i),
                    'ID_B': np.array(spikemon_B.i),
                    'cs_intervals': np.array(cs_intervals)})

        return(fr_A, fr_B, CS_A/nS, CS_B/nS, CTX_A/nS, CTX_B/nS)


    processing_simulations = mp.Pool(3)
    results = processing_simulations.map(renewal_fear_simulations, range(n_simulations))


    #loading the data from the last simulation
    data = np.load('renewal_fear/last_simulation_data.npy', allow_pickle=True)
    spk_ni = data.item().get('spk_ni_t')
    spk_ne = data.item().get('spk_ne_t')
    spk_A = data.item().get('spk_A')
    spk_B = data.item().get('spk_B')
    ID_ni = data.item().get('ID_ni')
    ID_ne = data.item().get('ID_ne')
    ID_A = data.item().get('ID_A')
    ID_B = data.item().get('ID_B')
    cs_intervals = data.item().get('cs_intervals')

    bin_f = 15 #time bin to calculate the time series of the trigger rate of the last simulation.
    fr_IH_last = np.histogram(spk_ni, bins=np.arange(0, tsim, bin_f))
    fr_A_last = np.histogram(spk_A, bins=np.arange(0, tsim, bin_f))
    fr_B_last = np.histogram(spk_B, bins=np.arange(0, tsim, bin_f))


    ###########################################################################
    # Final results
    ###########################################################################

    # Raster plot for the last simulation
    fig = plt.figure(constrained_layout=False, figsize=(15,10))
    gs = fig.add_gridspec(13,1)

    ax = fig.add_subplot(gs[0:7, 0])
    ax.plot(spk_ne, ID_ne, 'k.', ms=2) #all excitatory exneurons
    ax.plot(spk_ni, ID_ni + 3400, 'r.', ms=2) #inhibitory neurons
    ax.plot(spk_A, ID_A, '.', ms=4, color="tab:orange") #sub-pop A - fear neurons
    ax.plot(spk_B, ID_B + 2720 , '.', color="tab:blue", ms=4) #sub-pop B - extinction neurons

    for i,j in cs_intervals:
        ax.plot([i,j],[-100,-100],'k-', lw = 3)
    # plt.plot(tstim[nonzero_id],-100.0*np.ones(size(nonzero_id)),'.k')

    ax.text(350, -350, 'CONDITIONING')
    ax.text(1650, -350, 'EXTINCTION')
    ax.text(2430, -350, 'RENEWAL')
    ax.axvline(t1, ls='--', lw=2, color='black')
    ax.axvline(t2+tCTXB_dur, ls='--', lw=2, color='black')
    ax.set_ylim(-400,NE+NI)
    ax.set_xlim(50,tsim)
    ax.set_ylabel('# Neuron')
    ax.set_xlabel('Time (ms)')
    ax.text(-200, 3800,"A", weight="bold", fontsize=30)

    ax.get_xaxis().set_visible(False)

    ax = fig.add_subplot(gs[7:10, 0])

    ax.plot(fr_B_last[1][:-1], matlab_smooth(fr_B_last[0]*1000/(bin_f*NI), 5), lw=2)
    ax.plot(fr_A_last[1][:-1], matlab_smooth(fr_A_last[0]*1000/(bin_f*NI), 5), lw=2)

    #ax.plot(fr_BB[1][:-1], fr_BB[0]*1000/(bin_f*NI),lw=2)
    #ax.plot(fr_AA[1][:-1], fr_AA[0]*1000/(bin_f*NI),lw=2)
    #ax.plot(fr_IH[1][:-1], fr_IH[0]*1000/(bin_f*NI), 'r-')
    #plt.plot(fr_EX[1][:-1], fr_EX[0]*1000/(bin_f*NI), 'b-')

    for i,j in cs_intervals:
        ax.plot([i,j],[-0.5,-0.5],'k-', lw = 3)
    ax.axvline(t1, ls='--', lw=2, color='black')
    ax.axvline(t2+tCTXB_dur, ls='--', lw=2, color='black')
    ax.set_xlim(50,tsim)
    ax.set_ylim(-0.7,4.5)
    ax.axvline(t2+tCTXB_dur, ls='--', lw=2, color='black')

    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (ms)")
    ax.text(-200, 3.8,"B", weight="bold", fontsize=30)
    ax.get_xaxis().set_visible(False)

    ax = fig.add_subplot(gs[10:, 0])
    #ax.plot(fr_BB[1][:-1], fr_BB[0]*1000/(bin_f*NI),lw=2)
    #ax.plot(fr_AA[1][:-1], fr_AA[0]*1000/(bin_f*NI),lw=2)
    #ax.plot(fr_IH[1][:-1], fr_IH[0]*1000/(bin_f*NI), 'r-')
    ax.plot(fr_IH_last[1][:-1], matlab_smooth(fr_IH_last[0]*1000/(bin_f*NI), 5), 'r-')
    #plt.plot(fr_EX[1][:-1], fr_EX[0]*1000/(bin_f*NI), 'b-')

    for i,j in cs_intervals:
        ax.plot([i,j],[-2,-2],'k-', lw = 3)
    ax.axvline(t1, ls='--', lw=2, color='black')
    ax.axvline(t2+tCTXB_dur, ls='--', lw=2, color='black')
    ax.set_ylim(-3,20)
    ax.set_xlim(50,tsim)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (ms)")
    #ax.get_xaxis().set_visible(False)
    ax.text(-200, 17,"C", weight="bold", fontsize=30)
    plt.tight_layout()
    plt.savefig('renewal_fear/raster.png', dpi = 200)


    ###########################################################################
    # Average firing rate,CS and CTX for each subpopulation
    ###########################################################################
    n_CS = 12 #total CS presentation

    fig = plt.figure(constrained_layout=True, figsize=(7,10))
    gs = fig.add_gridspec(6,1)

    ax = fig.add_subplot(gs[0:2, 0])
    ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[0], yerr=np.std(results, axis=0)[0], fmt='s', ms=10, capsize=3, label = r"$pop_A$", color=plt.cm.tab10(1))
    ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[1], yerr=np.std(results, axis=0)[1], fmt='o', ms=10, capsize=3, label = r"$pop_B$", color=plt.cm.tab10(0))
    ax.axvline(5.5, ls='--', lw=2, color='black')
    ax.axvline(11.5, ls='--', lw=2, color='black')

    ax.set_xticks(range(n_CS+1))
    ax.set_ylim(-0.2,5)
    ax.text(0.8, 4.5, 'CONDITIONING')
    ax.text(6.9, 4.5, 'EXTINCTION')
    ax.text(11.9, 3.5, 'RENEWAL', horizontalalignment='center', verticalalignment='center', rotation=270)
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_xlabel('CS presentations')
    legend = ax.legend(bbox_to_anchor=(0.25,0.87), fontsize=15)
    legend.get_frame().set_alpha(0)
    ax.get_xaxis().set_visible(False)
    ax.text(-2, 4.4,"A", weight="bold", fontsize=30)


    ax = fig.add_subplot(gs[2:4, 0])
    ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[3], yerr=np.std(results, axis=0)[3], fmt='o', ms=10, capsize=3)
    ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[2], yerr=np.std(results, axis=0)[2], fmt='s', ms=10, capsize=3)
    ax.axvline(5.5, ls='--', lw=2, color='black')
    ax.axvline(11.5, ls='--', lw=2, color='black')

    ax.set_xticks(range(n_CS+1))
    ax.set_ylabel('CS weights (nS)')
    ax.set_xlabel('CS presentations')
    ax.set_ylim(0,3)
    ax.get_xaxis().set_visible(False)
    ax.text(-2, 2.7,"B", weight="bold", fontsize=30)


    ax = fig.add_subplot(gs[4:6, 0])
    ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[5], yerr=np.std(results, axis=0)[5], fmt='o', ms=10, capsize=3)
    ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[4], yerr=np.std(results, axis=0)[4], fmt='s', ms=10, capsize=3)
    ax.axvline(5.5, ls='--', lw=2, color='black')
    ax.axvline(11.5, ls='--', lw=2, color='black')

    ax.set_xticks(range(n_CS+1))
    ax.set_ylabel('HPC weights (nS)')
    ax.set_xlabel('CS presentations')
    ax.text(-2, 2.7,"C", weight="bold", fontsize=30)
    ax.set_ylim(0,3)
    plt.savefig('renewal_fear/avarages.png', dpi = 200)
    plt.show()


#protocol 3: increase in network connectivity, objective: to check gamma oscillations during the application of CS
#            analogous to figure 8A of the original paper
elif protocol == 3:
    os.system('mkdir gamma_oscillations')
    nCSA = 10
    tCTXA_dur = nCSA*(tCS_dur+tCS_off)	# CTX_A duration in ms

    pcon =  [[0.01,     # excitatory to excitatory
            0.50],    # excitatory to inhibitory
            [0.50,     # inhibitory to excitatory
            0.50]]    # inhibitory to inhibitory

    aux	= np.zeros(int((tCS_dur+tCS_off)/delta_tr))
    aux[:int(tCS_dur/delta_tr)] = 1
    m_array = np.concatenate([np.zeros(int(tinit/delta_tr)),np.tile(aux, nCSA)])

    mt       = TimedArray(m_array, dt=delta_tr*ms)
    stimulus = TimedArray(m_array*fCS*Hz, dt=delta_tr*ms)

    new_input_vars={
                'cs_rate'  : 'stimulus(t)',
                'ctxA_rate': '((t>='+str(tinit)+'*ms)*(t<='+str(tinit+tCTXA_dur)+'*ms))*'+str(fCTX)+'*Hz',
                }
    input_vars.update(new_input_vars)

    tsim = tinit+tCTXA_dur
    tstim= np.arange(0.0, tsim, delta_tr)			# Times discretized

    net, neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B = amygdala_net(input=True, input_vars=input_vars, pcon=pcon)
    net.run(tsim*ms, report='stdout')

    spikemon_ne, spikemon_ni, spikemon_A, spikemon_B = [spikemon[0],spikemon[1],spikemon[2],spikemon[3]]
    statemon_CS, statemon_CTX_A, statemon_CTX_B = [statemon[0], statemon[1], statemon[2]]

	###########################################################################
	# Results
	###########################################################################
    nonzero_id = np.nonzero(m_array)
    winsize  = int(tCS_dur/delta_tr)

    ind  = 0
    cs_intervals = []
    for i in range(nCSA):
        cs_intervals.append([tstim[nonzero_id][ind],tstim[nonzero_id][ind+winsize-1]])
        ind+=winsize

    fig = plt.figure(constrained_layout=False, figsize=(15,13))
    gs = fig.add_gridspec(14,1)

    ax = fig.add_subplot(gs[0:6, 0])
    ax.plot(spikemon_ne.t/ms, spikemon_ne.i, 'k.', ms=2) #all excitatory exneurons
    ax.plot(spikemon_ni.t/ms, spikemon_ni.i + NE, 'r.', ms=2) #inhibitory neurons
    ax.plot(spikemon_A.t/ms, spikemon_A.i, '.', ms=4, color="tab:orange") #sub-pop A - fear neurons
    ax.plot(spikemon_B.t/ms, spikemon_B.i + NE - NB , '.', color="tab:blue", ms=4) #sub-pop B - extinction neurons

    for i,j in cs_intervals:
        ax.plot([i,j],[-50,-50],'k-', lw = 3)

    ax.set_ylim(-100,NE+NI)
    ax.set_xlim(-50 + cs_intervals[6][0],cs_intervals[9][1] + 50)
    ax.set_ylabel('# Neuron')
    ax.set_xlabel('Time (ms)')
    ax.text(1175, 3800,"A", weight="bold", fontsize=30)
    ax.get_xaxis().set_visible(False)


    ax = fig.add_subplot(gs[6:10, 0])
    fr = np.concatenate([spikemon_ne.t/ms,spikemon_ni.t/ms])
    fr = fr[ (fr > (-50 + cs_intervals[6][0])) & (fr<(cs_intervals[9][1] + 50))]

    histo = np.histogram(fr, bins=np.arange(-50 + cs_intervals[6][0],cs_intervals[9][1] + 50 + 1,1))

    ax.plot(cs_intervals[6],[-5,-5],'k-',lw = 3)
    ax.plot(cs_intervals[7],[-5,-5],'k-',lw = 3)
    ax.plot(cs_intervals[8],[-5,-5],'k-',lw = 3)
    ax.plot(cs_intervals[9],[-5,-5],'k-',lw = 3)
    axhline(0, ls='-', c='black', lw=1)

    ax.set_xlim(-50 + cs_intervals[6][0],cs_intervals[9][1] + 50)

    ax.plot(histo[1][:-1],matlab_smooth(histo[0], 5)*1.5,'k-',lw=1.5)
    #ax.plot(histo[1][:-1],histo[0])

    ax.set_ylabel('Activity')
    ax.set_xlabel('Time (ms)')
    ax.text(1175, 44,"B", weight="bold", fontsize=30)

    ax = fig.add_subplot(gs[10:, 0])
    f, Pxx_den = signal.welch(histo[0], fs=1000) #PSD

    ax.plot(f, Pxx_den,lw=2)

    def annot_max(x,y, ax=None):
        xmax = x[np.argmax(y)]
        ymax = y.max()
        text= "Peak = {:.2f} Hz".format(xmax, ymax)
        if not ax:
            ax=plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->")
        kw = dict(xycoords='data',textcoords="axes fraction",
                arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        plt.annotate(text, xy=(xmax, ymax), xytext=(0.64,0.98), **kw)

    annot_max(f, Pxx_den)

    #plt.ylim([0.5e-3, 1])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.text(-77, 0.53,"C", weight="bold", fontsize=30)

    #plt.savefig('1-raster.svg', dpi = 150)
    #plt.xlim(0,80)
    plt.tight_layout()
    plt.savefig('gamma_oscillations/results.png', dpi = 200)
    plt.show()
    
    inh_activity,_ =  np.histogram(spikemon_ni.t/ms, bins=np.arange(-50 + cs_intervals[6][0],cs_intervals[9][1] + 50 + 1,1))
    print("Synchrony index: {:.2f}".format(np.var(inh_activity)/np.mean(inh_activity)))

#protocol 4: in network connectivity and change in inhibitory synaptic weights, objective: to check the increase of synchrony
#            analogous to figure 8B of the original paper
elif protocol == 4:

    nCSA = 10
    tCTXA_dur = nCSA*(tCS_dur+tCS_off)	# CTX_A duration in ms

    aux	= np.zeros(int((tCS_dur+tCS_off)/delta_tr))
    aux[:int(tCS_dur/delta_tr)] = 1
    m_array = np.concatenate([np.zeros(int(tinit/delta_tr)),np.tile(aux, nCSA)])

    mt       = TimedArray(m_array, dt=delta_tr*ms)
    stimulus = TimedArray(m_array*fCS*Hz, dt=delta_tr*ms)

    new_input_vars={
                'cs_rate'  : 'stimulus(t)',
                'ctxA_rate': '((t>='+str(tinit)+'*ms)*(t<='+str(tinit+tCTXA_dur)+'*ms))*'+str(fCTX)+'*Hz',
                }
    input_vars.update(new_input_vars)

    tsim = tinit+tCTXA_dur
    tstim= np.arange(0.0, tsim, delta_tr)			# Times discretized

    pii_variation    = np.arange(0.1,0.95,0.1)#np.arange(0,0.95,0.05)
    wii_variation    = [1.0, 2.0, 3.0]
    sdelii_variation = ['(rand() + 1.0)*ms','(rand()*0.8 + 0.2)*ms']
    sdelii_label     = ['high', 'low']
    n_simulations    = 5

    sync_index = []
    try:
        os.makedirs('./synchrony/data')
    except:
        None

    def synchrony_simulations(pii):
        for wii in wii_variation:
            for sdelii in range(len(sdelii_variation)):
                for l in range(n_simulations):
                    seed(100+l)
                    print("Running simulations: please wait")
                    print('pii: ' + str(pii) + ', wii: ' + str(wii) + ', sdel: ' + sdelii_variation[sdelii] + ', n: ' + str(l))
                    start_scope()

                    # default case
                    wsyn = [[1.25*nS,  #wee
                            1.25*nS], #wei
                            [2.50*nS,  #wie
                            wii*nS]]  #wii

                    pcon =  [[0.01,   # excitatory to excitatory
                            0.50],    # excitatory to inhibitory
                            [0.50,    # inhibitory to excitatory
                            pii]]     # inhibitory to inhibitory

                    sdelay = [['(randn()*0.1 + 2.0)*ms',    # excitatory to excitatory
                            '(randn()*0.1 + 2.0)*ms'],   # excitatory to inhibitory
                            ['(randn()*0.1 + 2.0)*ms',    # inhibitory to excitatory
                                sdelii_variation[sdelii]]]  # inhibitory to inhibitory

                    net, neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B = amygdala_net(input=True, input_vars=input_vars, pcon=pcon, wsyn=wsyn, sdel=sdelay)
                    # net.run(tsim*ms, report='stdout')
                    net.run(tsim*ms)

                    spikemon_ne, spikemon_ni, spikemon_A, spikemon_B = [spikemon[0],spikemon[1],spikemon[2],spikemon[3]]
                    # statemon_CS, statemon_CTX_A, statemon_CTX_B = [statemon[0], statemon[1], statemon[2]]

                    np.save('synchrony/data/pii' + ("{:.2f}".format(pii)) + '_wii' + ("{:.2f}".format(wii)) + '_sdelii_' + sdelii_label[sdelii] + '_n_{:}'.format(l) + '.npy', \
                        {'spk_ni_t': spikemon_ni.t/ms,
                        'spk_ne_t': spikemon_ne.t/ms,
                        'tstim': tstim, 'nCSA': nCSA, 'm': m_array,
                        'pii': pii, 'sdel': sdelii_variation[sdelii], 'wii': wii})
                
        return(pii)

    if __name__ == "__main__":
        results = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(synchrony_simulations)(pii) for pii in pii_variation )

    ###########################################################################
    # Synchrony index
    ###########################################################################

    #Understanding the meaning of our synchrony index
    wii = 2.0
    sdelii = 0
    l = 0
    pii = [0.1,0.6,0.7]
    panel = ['A', 'B', 'C']

    fig, axs = plt.subplots(3, 1, figsize=(14,9),sharex=True, sharey=True)
    for j in range(3):
        data = np.load('synchrony/data/pii' + ("{:.2f}".format(pii[j])) + '_wii' + ("{:.2f}".format(wii)) + '_sdelii_' + sdelii_label[sdelii] + '_n_{:}'.format(l) + '.npy', allow_pickle=True)
        #pii0.10_wii1.00_sdelii_low_n_3
        spk_ni  = data.item().get('spk_ni_t')
        m_array = data.item().get('m')
        tstim   = data.item().get('tstim')
        nCSA    = data.item().get('nCSA')

        nonzero_id = np.nonzero(m_array)
        winsize  = int(tCS_dur/delta_tr)
        ind  = 0
        cs_intervals = []
        for i in range(nCSA):
            cs_intervals.append([tstim[nonzero_id][ind],tstim[nonzero_id][ind+winsize-1]])
            ind+=winsize

        inh_activity,_ =  np.histogram(spk_ni, bins=np.arange(cs_intervals[-4][0],tstim[-1],1.0))

        ax = axs[j]
        ax.plot(_[:-1],matlab_smooth(inh_activity, 5)*1.5,'k-',lw=1.5)
        for i in range(1,5):
            ax.plot([cs_intervals[-i][0],cs_intervals[-i][1]],[-5,-5],'k-',lw = 3)
        ax.text(0.8,0.8,'Synchrony index: {:.2f}'.format(np.var(inh_activity)/np.mean(inh_activity)),horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, size=15)
        ax.set_xlim(1300,2100)
        ax.text(1315, 42,"{}".format(panel[j]), weight="bold", fontsize=25)
        #f, Pxx_den = signal.welch(inh_activity, fs=1000) #PSD

        
        #plt.plot(f, Pxx_den,lw=2)
    ax.set_xlabel("Time (ms)", fontsize=22)
    ax = axs[1]
    ax.set_ylabel("Activity", fontsize=22)
    plt.tight_layout(h_pad=0)
    plt.savefig('synchrony/examples_synchrony_index.png', dpi = 200)
    
    #complete figure for all simulations
    plt.figure()
    graph_color = ['gray', 'lightgreen', 'darkgreen']

    for sdelii in range(len(sdelii_variation)):
        for wii in wii_variation:
            sync_index = []
            for pii in pii_variation:
                n_sync_index = []
                for l in range(n_simulations):
                    data = np.load('synchrony/data/pii' + ("{:.2f}".format(pii)) + '_wii' + ("{:.2f}".format(wii)) + '_sdelii_' + sdelii_label[sdelii] + '_n_{:}'.format(l) + '.npy', allow_pickle=True)

                    spk_ni  = data.item().get('spk_ni_t')
                    m_array = data.item().get('m')
                    tstim   = data.item().get('tstim')
                    nCSA    = data.item().get('nCSA')

                    nonzero_id = np.nonzero(m_array)
                    winsize  = int(tCS_dur/delta_tr)
                    ind  = 0
                    cs_intervals = []
                    for i in range(nCSA):
                        cs_intervals.append([tstim[nonzero_id][ind],tstim[nonzero_id][ind+winsize-1]])
                        ind+=winsize

                    inh_activity,_ =  np.histogram(spk_ni, bins=np.arange(cs_intervals[-4][0],tstim[-1],1.0))
                    n_sync_index.append(np.var(inh_activity)/np.mean(inh_activity))
                sync_index.append(np.mean(n_sync_index))

            if sdelii_label[sdelii]=='high':
                plt.plot(pii_variation, sync_index, color=graph_color[int(wii)-1], label=r'$w_{ii}$: ' + ("{:.0f}".format(wii)) + ' nS' + ' for "higher"', lw=2)
            else:
                plt.plot(pii_variation, sync_index, color=graph_color[int(wii)-1], linestyle='--', label=r'$w_{ii}$: ' + ("{:.0f}".format(wii)) + ' nS'  + ' for "lower"',lw=2)
    plt.axhline(y=4.5, lw=1.0, color='purple',ls='--')
    plt.xlabel(r'$p_{ii}$', fontsize=22)
    plt.ylabel('Synchrony index',fontsize=22)
    legend = plt.legend(fontsize=15)
    legend.get_frame().set_alpha(0)
    plt.tight_layout()
    plt.savefig('synchrony/synchrony_index.png', dpi = 200)
    plt.show()

#protocol 5: GABA blockage experiment, objective: to qualitatively reproduce Figure 8C of the original paper
elif protocol == 5:
    os.system('mkdir GABA_blockage')
    inh_deactivate = [0.0, 0.5, 0.9]

    def GABA_block(deactivation):

        t1 = tinit + tCTXA_dur + tCTX_off
        t2 = tCTXB_dur

        aux	= np.zeros(int((tCS_dur+tCS_off)/delta_tr))
        aux[:int(tCS_dur/delta_tr)] = 1
        m_array = np.concatenate([np.zeros(int(tinit/delta_tr)),np.tile(aux, nCSA),\
                            np.zeros(int(tinit/delta_tr)),np.tile(aux, nCSB)])

        mt       = TimedArray(m_array, dt=delta_tr*ms)
        stimulus = TimedArray(m_array*fCS*Hz, dt=delta_tr*ms)

        new_input_vars={
                    'cs_rate'  : 'stimulus(t)',
                    'ctxA_rate': '(t<=' + str(tinit+tCTXA_dur) + '*ms)*' + str(fCTX) + '*Hz',
                    'ctxB_rate': '(t>=' + str(tinit+tCTXA_dur+tinit) + '*ms)*' + str(fCTX) + '*Hz',
                    }
        input_vars.update(new_input_vars)


        last_fr_A = []
        last_fr_B = []

        n_simulations = 30


        for l in range(n_simulations):
            seed(100+l)
            print("Running simulations: {} of {} for deactivation {:.0f}%".format(l+1,n_simulations,deactivation*100))

            start_scope()
            net, neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B = amygdala_net(input=True, input_vars=input_vars, record_weights=False)
            net.run(t1*ms, report='stdout')
            conn[2].w['i<(NI*deactivation)'] = 0.0*nS
            conn[3].w['i<(NI*deactivation)'] = 0.0*nS
            net.run(t2*ms, report='stdout')

            spikemon_ne, spikemon_ni, spikemon_A, spikemon_B = [spikemon[0],spikemon[1],spikemon[2],spikemon[3]]
            statemon_CS, statemon_CTX_A, statemon_CTX_B = [statemon[0], statemon[1], statemon[2]]

            ###########################################################################
            # Results
            ###########################################################################

            #Figure 5 a)
            nonzero_id = np.nonzero(m_array)
            winsize  = int(tCS_dur/delta_tr)

            ind  = 0
            cs_intervals = []
            for m in range(nCSA+nCSB):
                cs_intervals.append([tstim[nonzero_id][ind],tstim[nonzero_id][ind+winsize-1]])
                ind+=winsize

            ###########################################################################
            # Average firing rate for each subpopulation
            ###########################################################################
            print("Calculating the average firing rate for each subpopulation")

            timesA = spikemon_A.t/ms
            timesB = spikemon_B.t/ms

            fr_A = []
            fr_B = []

            fr_A.append(np.sum((timesA>=cs_intervals[-1][0]) & (timesA<=cs_intervals[-1][1]))/(NA*tCS_dur/1000.0))
            fr_B.append(np.sum((timesB>=cs_intervals[-1][0]) & (timesB<=cs_intervals[-1][1]))/(NB*tCS_dur/1000.0))

            last_fr_A.append(fr_A[-1])
            last_fr_B.append(fr_B[-1])
            
            if(l==(n_simulations-1)):
                np.save('GABA_blockage/raster_blockage_{}.npy'.format(deactivation*100),\
                        {'spk_ni_t': spikemon_ni.t/ms,
                        'spk_ne_t': spikemon_ne.t/ms,
                        'spk_A_t': spikemon_A.t/ms,
                        'spk_B_t': spikemon_B.t/ms,
                        'spk_ni_id': np.array(spikemon_ni.i),
                        'spk_ne_id': np.array(spikemon_ne.i),
                        'spk_A_id': np.array(spikemon_A.i),
                        'spk_B_id': np.array(spikemon_B.i),
                        'cs_time': np.array(cs_intervals)})

        return(np.mean(last_fr_A), np.std(last_fr_A),np.mean(last_fr_B), np.std(last_fr_B))

    processing_blockage = mp.Pool(3)
    results = processing_blockage.map(GABA_block, inh_deactivate)

    for l in inh_deactivate:
        data = np.load('GABA_blockage/raster_blockage_{}.npy'.format(l*100),allow_pickle=True)
        spk_ne_t = data.item().get('spk_ne_t')
        spk_ni_t = data.item().get('spk_ni_t')
        spk_A_t = data.item().get('spk_A_t')
        spk_B_t = data.item().get('spk_B_t')
        spk_ne_id = data.item().get('spk_ne_id')
        spk_ni_id = data.item().get('spk_ni_id')
        spk_A_id = data.item().get('spk_A_id')
        spk_B_id = data.item().get('spk_B_id')
        cs_time = data.item().get('cs_time')

        plt.figure(figsize=(15,7))
        plt.plot(spk_ne_t, spk_ne_id, 'k.', ms=2) #all excitatory exneurons
        plt.plot(spk_ni_t, spk_ni_id + 3400, 'r.', ms=2) #inhibitory neurons
        plt.plot(spk_A_t, spk_A_id, '.', ms=4, color="tab:orange") #sub-pop A - fear neurons
        plt.plot(spk_B_t, spk_B_id + 2720 , '.', color="tab:blue", ms=4) #sub-pop B - extinction neurons

        for m,n in cs_time:
            plt.plot([m,n],[-100,-100],'k-', lw = 3)
	    
        plt.text(350, -350, 'CONDITIONING')
        plt.text(1650, -350, 'EXTINCTION')
        plt.axvline(tinit + tCTXA_dur, ls='--', lw=2, color='black')
        plt.ylim(-400,4000)
        plt.xlim(-50,tsim + 50)
        plt.ylabel('# Neuron')
        plt.xlabel('Time (ms)')
        plt.tight_layout()
        plt.savefig('GABA_blockage/raster_blockage_{}.png'.format(l*100), dpi = 200)
	#plt.show()

    n_fr_A = []
    error_fr_A = []
    n_fr_B = []
    error_fr_B = []

    for k in range(len(inh_deactivate)):
        n_fr_A.append(results[k][0])
        error_fr_A.append(results[k][1])
        n_fr_B.append(results[k][2])
        error_fr_B.append(results[k][3])

    plt.figure(figsize=(5,6))
    GABA_blockage = ['0%','50%','90%']
    plt.xticks([0,50,90], GABA_blockage)

    plt.errorbar([0,50,90],n_fr_B, yerr=error_fr_B, fmt='o', ms=10, capsize=3)
    plt.errorbar([0,50,90],n_fr_A, yerr=error_fr_A, fmt='s', ms=10, capsize=3)

    plt.xlim(-10,100)
    plt.xlabel("GABA blockage")
    plt.ylabel("Firing Rate (Hz)")
    plt.tight_layout()
    plt.savefig('GABA_blockage/gaba_block.png', dpi = 200)
    plt.show()
