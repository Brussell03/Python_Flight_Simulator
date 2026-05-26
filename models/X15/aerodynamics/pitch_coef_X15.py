# Pitch coefficient build-up from:
# Dydek Z., Annaswamy A., and Lavretsky E., Adaptive Control and the NASA X-15-3 Flight Revisited, 
# IEEE Control Systems Magazine, June 2010.
#
def Cm_X15(Cmwb, Cmalpha_pdeg, Cmq_prps, Cmalphadot_prps, Cmdele_pdeg, Cmdelsb_pdeg, alpha_deg, \
        alphadot_rps, q_rps, dele_deg, delsb_deg, true_airspeed_mps, c_m):

    Cm = Cmwb + \
        Cmalpha_pdeg*alpha_deg + \
        Cmq_prps*q_rps*c_m/(2*true_airspeed_mps) + \
        Cmalphadot_prps*alphadot_rps*c_m/(2*true_airspeed_mps) + \
        Cmdele_pdeg*dele_deg + \
        Cmdelsb_pdeg*delsb_deg

    return Cm