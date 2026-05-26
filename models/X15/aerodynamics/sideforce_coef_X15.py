def CY_X15(CYBeta_prad, CYP_prps, CYR_prps, CYBetadot_pdps, CYdela_pdeg, CYdelr_pdeg, Beta_rad, P_rps, R_rps, \
    Betadot_dps, dela_deg, delr_deg, true_airspeed_mps, b_m):

    CY = CYBeta_prad*Beta_rad + CYP_prps*P_rps*b_m/(2*true_airspeed_mps) + CYR_prps*R_rps*b_m/(2*true_airspeed_mps) + \
    CYBetadot_pdps*Betadot_dps*b_m/(2*true_airspeed_mps) + CYdela_pdeg*dela_deg + CYdelr_pdeg*delr_deg
    
    return CY