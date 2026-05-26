# Roll coefficient build-up from:
# Dydek Z., Annaswamy A., and Lavretsky E., Adaptive Control and the NASA X-15-3 Flight Revisited, 
# IEEE Control Systems Magazine, June 2010.
#
def Cl_X15(ClBeta_prad, ClP_prps, ClR_prps, ClBetadot_pdps, Cldela_pdeg, Cldelr_pdeg, Beta_rad, Betadot_dps, P_rps, r_rps, \
     dela_deg, delr_deg, true_airspeed_mps, b_m):

     Cl = ClBeta_prad*Beta_rad + ClP_prps*P_rps*b_m/(2*true_airspeed_mps) + ClR_prps*r_rps*b_m/(2*true_airspeed_mps) + \
     ClBetadot_pdps*Betadot_dps*b_m/(2*true_airspeed_mps) + Cldela_pdeg*dela_deg + Cldelr_pdeg*delr_deg

     return Cl