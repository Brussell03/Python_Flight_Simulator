# Lift coefficient build-up from:
# Dydek Z., Annaswamy A., and Lavretsky E., Adaptive Control and the NASA X-15-3 Flight Revisited, 
# IEEE Control Systems Magazine, June 2010.
#
# Note, CLdele_pdeg is a function of Mach and (AoA+dele) per the source, Walker60.
#
def CL_X15(CLwb, CLalpha_pdeg, CLdele_pdeg, alpha_deg, dele_deg):

    #CL = CLwb + CLalpha_pdeg*alpha_deg + CLdele_pdeg*(alpha_deg + dele_deg)
    CL = CLwb + CLalpha_pdeg*alpha_deg + CLdele_pdeg*dele_deg
    
    return CL