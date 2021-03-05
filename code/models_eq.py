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
# Equations for the neuron, synapse and plasticity models.
# ----------------------------------------------------------------------------


#############################################################################
# Neuron model equations
#############################################################################
eq_LIF = '''
        dv/dt = (Gl * (E0 - v) + Gexc * (Eexc - v) + Ginh * (Einh - v)) / Cm : volt (unless refractory)

        dGexc/dt = Gexc_aux - Gexc/tauexc_rise : siemens
        dGexc_aux/dt = -Gexc_aux/tauexc_decay : siemens/second

        dGinh/dt = Ginh_aux - Ginh/tauinh_rise : siemens
        dGinh_aux/dt = -Ginh_aux/tauinh_decay : siemens/second

        dh/dt = -h/tauh : 1
        dc/dt = -c/tauc : 1

        tcs : second
        tctx: second
        wcs : siemens
        wctx: siemens
        '''

reset_LIF = '''
            v = Ek
            '''

#############################################################################
# Synapse model equations
#############################################################################
syn_model = ''' w   : siemens'''
pre_exc   = '''
            Gexc_aux_post += w*Gexc_0
            '''

pre_inh   = '''
            Ginh_aux_post += w*Ginh_0
            '''
pre_eq    = [pre_exc, pre_inh]

#############################################################################
# Synaptic plasticity model equations
#############################################################################
syn_plast  =''' delta_t : second
            '''
pre_cs     ='''
            tcs_post = t
            c_post += c_u
            delta_t = abs(tcs_post - tctx_post)
            wcs_post += mt(t)*alpha*h_post*abs(w_max-wcs_post)*c_post*(delta_t<100.0*ms) - mt(t)*alpha*abs(w_min-wcs_post)*c_post*(delta_t>100.0*ms)
            wctx_post += mt(t)*alpha*h_post*abs(w_max-wctx_post)*c_post*(delta_t<100.0*ms) - mt(t)*alpha*abs(w_min-wctx_post)*c_post*(delta_t>100.0*ms)
            Gexc_aux_post += wcs_post * Gexc_0
            '''
pre_ctx    ='''
            tctx_post = t
            h_post+= h_u
            delta_t = abs(tcs_post - tctx_post)
            wcs_post+= mt(t)*alpha*h_post*abs(w_max-wcs_post)*c_post*(delta_t<100.0*ms) - mt(t)*alpha*abs(w_min-wcs_post)*c_post*(delta_t>100.0*ms)
            wctx_post+= mt(t)*alpha*h_post*abs(w_max-wctx_post)*c_post*(delta_t<100.0*ms) - mt(t)*alpha*abs(w_min-wctx_post)*c_post*(delta_t>100.0*ms)
            Gexc_aux_post += wctx_post * Gexc_0
            '''
 
