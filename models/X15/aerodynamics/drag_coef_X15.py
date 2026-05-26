# Lift coefficient build-up from:
# Dydek Z., Annaswamy A., and Lavretsky E., Adaptive Control and the NASA X-15-3 Flight Revisited, 
# IEEE Control Systems Magazine, June 2010.
#
def CD_X15(CDwb, CDdele_pdeg, CDdelsb_pdeg, dele_deg, delsb_deg):

    CD = CDwb + CDdele_pdeg*dele_deg + CDdelsb_pdeg*delsb_deg # speed brake is provided as delta CDSB not a deritvative
    
    return CD