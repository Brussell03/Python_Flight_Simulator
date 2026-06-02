# Yaw coefficient build-up from:
# Dydek Z., Annaswamy A., and Lavretsky E., Adaptive Control and the NASA X-15-3 Flight Revisited, 
# IEEE Control Systems Magazine, June 2010.
#
def Cn_X15(CnBeta_prad, CnP_prps, CnR_prps, CnBetadot_pdps, Cndela_pdeg, Cndelr_pdeg, Beta_rad, Betadot_dps, P_rps, R_rps, \
    dela_deg, delr_deg, true_airspeed_mps, b_m):

    Cn = CnBeta_prad*Beta_rad + CnP_prps*P_rps*b_m/(2*true_airspeed_mps) + CnR_prps*R_rps*b_m/(2*true_airspeed_mps) + \
    CnBetadot_pdps*Betadot_dps*b_m/(2*true_airspeed_mps) + Cndela_pdeg*dela_deg + Cndelr_pdeg*delr_deg

    return Cn