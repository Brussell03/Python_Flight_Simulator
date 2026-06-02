from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the project root folder to sys.path
root_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(root_dir))

from src.utils.interpolators import fastInterp1, fastInterp2
'''build_X15_database.py imports csv data to create an aerodynamic database for the 
X-15 space plane. 

Imported data is extracted from various NASA reports using a plot digitizer. The data
is then imported here, interpolated to a set of common specified breakpoints,
collected in a dictionary, and saved.

In many cases, the data in the reports exists only for positive angle of attack. As a 
result, here the data is reflected to negative angle of attack values, assuming 
symmetry. The canopy, anhedral of the horizontal stabilizers, and asymmetry between 
dorsal and ventral vertical stabilizers so that the vehicle is not symmetric and 
the angle of attack reflection will introduce errors.

To populate the aerodynamic database, the data derived from the reports is a mix of 
computational and experminentally derived values. Data digitization involves uncertainty. 
The original data also has unknown uncertainly levels. Therefore, the actual accuracy 
of this aerodynamic data is uncertain and may not accuratly capture the actual flight 
dynamics of the X-15.'''

# Select if plot is made
CL_vs_alpha_Mach_plot           = 'on'
Cmalpha_vs_Mach_plot            = 'on'
Cmq_pdeg_vs_Mach_plot           = 'on'
Cmdele_vs_Mach_plot             = 'on'
CLdele_vs_Mach_AoA_p_dele_plot  = 'on'
Clbeta_vs_Mach_alpha_plot       = 'on'
CL_vs_alpha_plot                = 'on'
CD_vs_alpha_plot                = 'on'
CYbeta_pdeg_vs_Mach_plot        = 'on'
Cnbeta_pdeg_vs_Mach_plot        = 'on'
CYdelr_pdeg_vs_Mach_plot        = 'on'
CYp_prps_vs_Mach_plot           = 'on'
CYr_prps_vs_Mach_plot           = 'on'
CYdela_pdeg_vs_Mach_plot        = 'on'
Clp_vs_Mach_alpha_plot          = 'on'
Clr_vs_Mach_alpha_plot          = 'on'
Cldela_pdeg_vs_Mach_plot        = 'on'
Cldelr_pdeg_vs_Mach_plot        = 'on'
Cm_vs_alpha_plot                = 'on'
Cnp_vs_Mach_alpha_plot          = 'on'
Cnr_vs_Mach_alpha_plot          = 'on'
Cndela_pdeg_vs_Mach_plot        = 'on'
Cndelr_pdeg_vs_Mach_plot        = 'on'
Jxx_Jxz_vs_Weight_plot          = 'on'
Jyy_vs_Weight_plot              = 'on'
Jzz_vs_Weight_plot              = 'on'

# Select if database is saved (overwrites previous database)
save_database = 'on'

# Breakpoints for angle of attack and Mach numbers
alpha_bps_deg  = np.arange(-25, 26, 1, dtype=float)
nalpha_bps_deg = len(alpha_bps_deg)

Mach_bps = np.arange(2, 8.5, 0.5, dtype=float)
nMach_bps = len(Mach_bps)

# 1.    Coefficient: Lift, CL = f(alpha_bps_deg, Mach_bps)
#       Method of determination: Computational
#       Details: Total body, no surface deflections
#       Source: Walker 1960, page 66. 
#-------------------------------------------------------------------------------

# CL Table from comma separated variables exported from plot digitizer
CL_for_AoA_Mach = pd.read_csv('models/X15/aerodynamic_model/Walker60/data/CL_v_alpha_Walker60.csv', header = 0)

# Get data sequences from the pandas DataFrame
alpha_sample_deg   = CL_for_AoA_Mach['x'].values
CL_vs_alpha_deg_M2 = CL_for_AoA_Mach['CL_vs_alpha_deg_M2_Walker60'].values
CL_vs_alpha_deg_M3 = CL_for_AoA_Mach['CL_vs_alpha_deg_M3_Walker60'].values
CL_vs_alpha_deg_M4 = CL_for_AoA_Mach['CL_vs_alpha_deg_M4_Walker60'].values
CL_vs_alpha_deg_M6 = CL_for_AoA_Mach['CL_vs_alpha_deg_M6_Walker60'].values
CL_vs_alpha_deg_M8 = CL_for_AoA_Mach['CL_vs_alpha_deg_M8_Walker60'].values

# Reflect CL data about x-axis (CL = 0)
alpha_sample_deg   = -alpha_sample_deg[::-1] + alpha_sample_deg
CL_vs_alpha_deg_M2 = -CL_vs_alpha_deg_M2[::-1] + CL_vs_alpha_deg_M2
CL_vs_alpha_deg_M3 = -CL_vs_alpha_deg_M3[::-1] + CL_vs_alpha_deg_M3
CL_vs_alpha_deg_M4 = -CL_vs_alpha_deg_M4[::-1] + CL_vs_alpha_deg_M4
CL_vs_alpha_deg_M6 = -CL_vs_alpha_deg_M6[::-1] + CL_vs_alpha_deg_M6
CL_vs_alpha_deg_M8 = -CL_vs_alpha_deg_M8[::-1] + CL_vs_alpha_deg_M8

# We will populate CL vectors for each Mach over the alpha_deg values
CL_v_alpha_deg_M2 = np.zeros(nalpha_bps_deg)
CL_v_alpha_deg_M3 = np.zeros(nalpha_bps_deg)
CL_v_alpha_deg_M4 = np.zeros(nalpha_bps_deg)
CL_v_alpha_deg_M6 = np.zeros(nalpha_bps_deg)
CL_v_alpha_deg_M8 = np.zeros(nalpha_bps_deg)

# Interpolate to a regularly spaced database
for ii in range(0, nalpha_bps_deg, 1):
    CL_v_alpha_deg_M2[ii] = fastInterp1(alpha_sample_deg, CL_vs_alpha_deg_M2, alpha_bps_deg[ii])
    CL_v_alpha_deg_M3[ii] = fastInterp1(alpha_sample_deg, CL_vs_alpha_deg_M3, alpha_bps_deg[ii])
    CL_v_alpha_deg_M4[ii] = fastInterp1(alpha_sample_deg, CL_vs_alpha_deg_M4, alpha_bps_deg[ii])
    CL_v_alpha_deg_M6[ii] = fastInterp1(alpha_sample_deg, CL_vs_alpha_deg_M6, alpha_bps_deg[ii])
    CL_v_alpha_deg_M8[ii] = fastInterp1(alpha_sample_deg, CL_vs_alpha_deg_M8, alpha_bps_deg[ii])
    
# Combine data into a single 2D array
CL_vs_alpha_deg_Mach = np.column_stack((CL_v_alpha_deg_M2, CL_v_alpha_deg_M3, \
    CL_v_alpha_deg_M4, CL_v_alpha_deg_M6, CL_v_alpha_deg_M8))

# Interpolate to all specified Mach breakpoints
CL_table_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
Mach_presently = np.array([2, 3, 4, 6, 8], dtype=float)

for jj in range(nMach_bps):
    for ii in range(nalpha_bps_deg):
        CL_table_alpha_deg_Mach[ii, jj] = fastInterp2(alpha_bps_deg, Mach_presently, CL_vs_alpha_deg_Mach, alpha_bps_deg[ii], Mach_bps[jj])

# Compare the results on a single plot
if CL_vs_alpha_Mach_plot == 'on':
    plt.figure(1)
    plt.plot(alpha_sample_deg, CL_vs_alpha_deg_M2, color='black',  linestyle='solid', label="Mach 2 - Extracted")
    plt.plot(alpha_sample_deg, CL_vs_alpha_deg_M3, color='blue',   linestyle='solid', label="Mach 3 - Extracted")
    plt.plot(alpha_sample_deg, CL_vs_alpha_deg_M4, color='green',  linestyle='solid', label="Mach 4 - Extracted")
    plt.plot(alpha_sample_deg, CL_vs_alpha_deg_M6, color='red',    linestyle='solid', label="Mach 6 - Extracted")
    plt.plot(alpha_sample_deg, CL_vs_alpha_deg_M8, color='orange', linestyle='solid', label="Mach 8 - Extracted")
    plt.plot(alpha_bps_deg, CL_table_alpha_deg_Mach[:,0], color='grey', linestyle='dashed', label="Mach 2 - Interpolated", marker='o', markerfacecolor='none')
    plt.plot(alpha_bps_deg, CL_table_alpha_deg_Mach[:,2], color='grey', linestyle='dashed', label="Mach 3 - Interpolated", marker='x')
    plt.plot(alpha_bps_deg, CL_table_alpha_deg_Mach[:,4], color='grey', linestyle='dashed', label="Mach 4 - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(alpha_bps_deg, CL_table_alpha_deg_Mach[:,6], color='grey', linestyle='dashed', label="Mach 5 - Interpolated", marker='+')
    plt.plot(alpha_bps_deg, CL_table_alpha_deg_Mach[:,8], color='grey', linestyle='dashed', label="Mach 6 - Interpolated", marker='D', markerfacecolor='none')
    plt.plot(alpha_bps_deg, CL_table_alpha_deg_Mach[:,10], color='grey', linestyle='dashed', label="Mach 7 - Interpolated", marker='*')
    plt.plot(alpha_bps_deg, CL_table_alpha_deg_Mach[:,12], color='grey', linestyle='dashed', label="Mach 8 - Interpolated", marker='h', markerfacecolor='none')
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("CL")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 2.    Coefficient: CLdele [1/deg] = f(AoA + dele (deg), Mach)
#       Method of determination: Computational
#       Details: Change in lift due to change in incidence on elevator (angle of attack + elevator deflection)
#       Source: Walker 60, report page 77
#------------------------------------------------------------------------------------------

# CLdele Table from comma separated variables exported from plot digitizer
CLdele_for_AoA_p_dele_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/CLdele_v_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample = CLdele_for_AoA_p_dele_Mach['x'].values
CLdele_vs_Mach_pdeg_AoA_p_dele_0_deg  = CLdele_for_AoA_p_dele_Mach['CLdele_v_Mach_alpha_0deg_Walker60'].values
CLdele_vs_Mach_pdeg_AoA_p_dele_8_deg  = CLdele_for_AoA_p_dele_Mach['CLdele_v_Mach_alpha_8deg_Walker60'].values
CLdele_vs_Mach_pdeg_AoA_p_dele_16_deg = CLdele_for_AoA_p_dele_Mach['CLdele_v_Mach_alpha_16deg_Walker60'].values
CLdele_vs_Mach_pdeg_AoA_p_dele_24_deg = CLdele_for_AoA_p_dele_Mach['CLdele_v_Mach_alpha_24deg_Walker60'].values

# We will populate CLdele vectors for each AoA over the Mach values
CLdele_pdeg_v_Mach_AoA_p_dele_0  = np.zeros(nMach_bps)
CLdele_pdeg_v_Mach_AoA_p_dele_8  = np.zeros(nMach_bps)
CLdele_pdeg_v_Mach_AoA_p_dele_16 = np.zeros(nMach_bps)
CLdele_pdeg_v_Mach_AoA_p_dele_24 = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    CLdele_pdeg_v_Mach_AoA_p_dele_0[ii]  = fastInterp1(Mach_sample, CLdele_vs_Mach_pdeg_AoA_p_dele_0_deg,  Mach_bps[ii])
    CLdele_pdeg_v_Mach_AoA_p_dele_8[ii]  = fastInterp1(Mach_sample, CLdele_vs_Mach_pdeg_AoA_p_dele_8_deg,  Mach_bps[ii])
    CLdele_pdeg_v_Mach_AoA_p_dele_16[ii] = fastInterp1(Mach_sample, CLdele_vs_Mach_pdeg_AoA_p_dele_16_deg, Mach_bps[ii])
    CLdele_pdeg_v_Mach_AoA_p_dele_24[ii] = fastInterp1(Mach_sample, CLdele_vs_Mach_pdeg_AoA_p_dele_24_deg, Mach_bps[ii])
    
# Reflect CLdele to negative AoA
CLdele_pdeg_v_Mach_AoA_p_dele_m8  = -CLdele_pdeg_v_Mach_AoA_p_dele_8
CLdele_pdeg_v_Mach_AoA_p_dele_m16 = -CLdele_pdeg_v_Mach_AoA_p_dele_16
CLdele_pdeg_v_Mach_AoA_p_dele_m24 = -CLdele_pdeg_v_Mach_AoA_p_dele_24

# Combine data into a single 2D array
CLdele_pdeg_v_Mach_AoA_p_dele = np.column_stack((CLdele_pdeg_v_Mach_AoA_p_dele_m24, \
                                                    CLdele_pdeg_v_Mach_AoA_p_dele_m16, \
                                                    CLdele_pdeg_v_Mach_AoA_p_dele_m8, \
                                                    CLdele_pdeg_v_Mach_AoA_p_dele_0, \
                                                    CLdele_pdeg_v_Mach_AoA_p_dele_8, \
                                                    CLdele_pdeg_v_Mach_AoA_p_dele_16, \
                                                    CLdele_pdeg_v_Mach_AoA_p_dele_24))

# Interpolate table CLdele_pdeg_v_Mach_AoA_p_dele to breakpoints for Mach and alpha
# Below, alpha is used in place of alpha+dele for convenience. The data is in terms of
# alpha+dele [deg]. We use AoA_p_dele to indicate this but alpha_bps_deg is used here
# for convenience because it shares the same desired breakpoint values.
alphapdele_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
CLdele_table_pdeg_v_Mach_AoA_p_dele = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        CLdele_table_pdeg_v_Mach_AoA_p_dele[jj,ii]  = fastInterp2(Mach_bps, alphapdele_deg_sample, \
                                    CLdele_pdeg_v_Mach_AoA_p_dele,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if CLdele_vs_Mach_AoA_p_dele_plot == 'on':
    plt.figure(3)
    plt.plot(Mach_sample, CLdele_vs_Mach_pdeg_AoA_p_dele_0_deg,  color='black',  linestyle='solid', label="AoA + dele 0 [deg] - Extracted")
    plt.plot(Mach_sample, CLdele_vs_Mach_pdeg_AoA_p_dele_8_deg,  color='blue',   linestyle='solid', label="AoA + dele 8 [deg] - Extracted")
    plt.plot(Mach_sample, CLdele_vs_Mach_pdeg_AoA_p_dele_16_deg, color='green',  linestyle='solid', label="AoA + dele 16 [deg] - Extracted")
    plt.plot(Mach_sample, CLdele_vs_Mach_pdeg_AoA_p_dele_24_deg, color='red',    linestyle='solid', label="AoA + dele 24 [deg] - Extracted")   
    plt.plot(Mach_bps, CLdele_table_pdeg_v_Mach_AoA_p_dele[1,:],  color='grey', linestyle='dashed', label="AoA + dele -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, CLdele_table_pdeg_v_Mach_AoA_p_dele[9,:],  color='grey', linestyle='dashed', label="AoA + dele -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, CLdele_table_pdeg_v_Mach_AoA_p_dele[17,:], color='grey', linestyle='dashed', label="AoA + dele -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, CLdele_table_pdeg_v_Mach_AoA_p_dele[25,:], color='grey', linestyle='dashed', label="AoA + dele 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, CLdele_table_pdeg_v_Mach_AoA_p_dele[33,:], color='grey', linestyle='dashed', label="AoA + dele 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, CLdele_table_pdeg_v_Mach_AoA_p_dele[41,:], color='grey', linestyle='dashed', label="AoA + dele 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, CLdele_table_pdeg_v_Mach_AoA_p_dele[49,:], color='grey', linestyle='dashed', label="AoA + dele 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("CLdele_pdeg")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()


# 3.    Coefficient: Cmalpha_pdeg = f(alpha_bps_deg, Mach_bps)
#       Method of determination: Computational
#       Details: Change in pitch moment due to change in angle of attack
#       Source: Walker 60, report page 75
#-------------------------------------------------------------------------------

# Cmalpha table from comma separated variables exported from plot digitizer
Cmalpha_for_AoA_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/Cmalpha_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample                    = Cmalpha_for_AoA_Mach['x'].values
Cmalpha_vs_Mach_pdeg_AoA0_deg  = Cmalpha_for_AoA_Mach['Cmalpha_pdeg_vs_Mach_AoA0deg_Walker60'].values
Cmalpha_vs_Mach_pdeg_AoA8_deg  = Cmalpha_for_AoA_Mach['Cmalpha_pdeg_vs_Mach_AoA8deg_Walker60'].values
Cmalpha_vs_Mach_pdeg_AoA16_deg = Cmalpha_for_AoA_Mach['Cmalpha_pdeg_vs_Mach_AoA16deg_Walker60'].values
Cmalpha_vs_Mach_pdeg_AoA24_deg = Cmalpha_for_AoA_Mach['Cmalpha_pdeg_vs_Mach_AoA24deg_Walker60'].values

# Populate Cmalpha vectors for each AoA over the Mach values
Cmalpha_vs_Mach_AoA0_deg  = np.zeros(nMach_bps)
Cmalpha_vs_Mach_AoA8_deg  = np.zeros(nMach_bps)
Cmalpha_vs_Mach_AoA16_deg = np.zeros(nMach_bps)
Cmalpha_vs_Mach_AoA24_deg = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Cmalpha_vs_Mach_AoA0_deg[ii]  = fastInterp1(Mach_sample, Cmalpha_vs_Mach_pdeg_AoA0_deg,  Mach_bps[ii])
    Cmalpha_vs_Mach_AoA8_deg[ii]  = fastInterp1(Mach_sample, Cmalpha_vs_Mach_pdeg_AoA8_deg,  Mach_bps[ii])
    Cmalpha_vs_Mach_AoA16_deg[ii] = fastInterp1(Mach_sample, Cmalpha_vs_Mach_pdeg_AoA16_deg, Mach_bps[ii])
    Cmalpha_vs_Mach_AoA24_deg[ii] = fastInterp1(Mach_sample, Cmalpha_vs_Mach_pdeg_AoA24_deg, Mach_bps[ii])

# Reflect Cmalpha to negative AoA
Cmalpha_vs_Mach_AoAm8_deg  = -Cmalpha_vs_Mach_AoA8_deg
Cmalpha_vs_Mach_AoAm16_deg = -Cmalpha_vs_Mach_AoA16_deg
Cmalpha_vs_Mach_AoAm24_deg = -Cmalpha_vs_Mach_AoA24_deg
    
# Combine data into a single 2D array 
Cmalpha_combined_pdeg_vs_Mach_alpha_deg = np.column_stack((Cmalpha_vs_Mach_AoAm24_deg, Cmalpha_vs_Mach_AoAm16_deg, Cmalpha_vs_Mach_AoAm8_deg, \
                                                            Cmalpha_vs_Mach_AoA0_deg,  Cmalpha_vs_Mach_AoA8_deg,   Cmalpha_vs_Mach_AoA16_deg,  \
                                                            Cmalpha_vs_Mach_AoA24_deg))

# Interpolate table Cmalpha_pdeg_vs_Mach_alpha_deg to final aerodynamic database breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Cmalpha_pdeg_table_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Cmalpha_pdeg_table_alpha_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                            Cmalpha_combined_pdeg_vs_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Cmalpha_vs_Mach_plot == 'on':
    plt.figure(3)
    plt.plot(Mach_sample, Cmalpha_vs_Mach_pdeg_AoA0_deg,  color='black',   linestyle='solid', label="AoA 0 [deg] - Extracted")
    plt.plot(Mach_sample, Cmalpha_vs_Mach_pdeg_AoA8_deg,  color='blue',    linestyle='solid', label="AoA 8 [deg] - Extracted")
    plt.plot(Mach_sample, Cmalpha_vs_Mach_pdeg_AoA16_deg, color='green',   linestyle='solid', label="AoA 16 [deg] - Extracted")
    plt.plot(Mach_sample, Cmalpha_vs_Mach_pdeg_AoA24_deg, color='red',     linestyle='solid', label="AoA 24 [deg] - Extracted")
    plt.plot(Mach_bps, Cmalpha_pdeg_table_alpha_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cmalpha_pdeg_table_alpha_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cmalpha_pdeg_table_alpha_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Cmalpha_pdeg_table_alpha_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Cmalpha_pdeg_table_alpha_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cmalpha_pdeg_table_alpha_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cmalpha_pdeg_table_alpha_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("Cmalpha_pdeg")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 4.    Coefficient: Cmq_pdps = f(alpha_bps_deg, Mach_bps)
#       Method of determination: Computational
#       Details: Change in pitch moment due to change in pitch rate (rate damping)
#       Source: Walker 60, report page 84
#-------------------------------------------------------------------------------

# Cmalpha table from comma separated variables exported from plot digitizer
Cmq_pdps_vs_Mach_alpha_deg_0  = pd.read_csv('models/X15/aerodynamic_model/Walker60/data/Cmq_prps_vs_Mach_alpha_deg_0_Walker60.csv',  header = 0)
Cmq_pdps_vs_Mach_alpha_deg_8  = pd.read_csv('models/X15/aerodynamic_model/Walker60/data/Cmq_prps_vs_Mach_alpha_deg_8_Walker60.csv',  header = 0)
Cmq_pdps_vs_Mach_alpha_deg_16 = pd.read_csv('models/X15/aerodynamic_model/Walker60/data/Cmq_prps_vs_Mach_alpha_deg_16_Walker60.csv', header = 0)
Cmq_pdps_vs_Mach_alpha_deg_24 = pd.read_csv('models/X15/aerodynamic_model/Walker60/data/Cmq_prps_vs_Mach_alpha_deg_24_Walker60.csv', header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample_for_alpha_deg_0   = Cmq_pdps_vs_Mach_alpha_deg_0['x'].values
Mach_sample_for_alpha_deg_8   = Cmq_pdps_vs_Mach_alpha_deg_8['x'].values
Mach_sample_for_alpha_deg_16  = Cmq_pdps_vs_Mach_alpha_deg_16['x'].values
Mach_sample_for_alpha_deg_24  = Cmq_pdps_vs_Mach_alpha_deg_24['x'].values
Cmq_pdps_vs_Mach_alpha_deg_0  = Cmq_pdps_vs_Mach_alpha_deg_0['alpha_deg_0'].values
Cmq_pdps_vs_Mach_alpha_deg_8  = Cmq_pdps_vs_Mach_alpha_deg_8['alpha_deg_8'].values
Cmq_pdps_vs_Mach_alpha_deg_16 = Cmq_pdps_vs_Mach_alpha_deg_16['alpha_deg_16'].values
Cmq_pdps_vs_Mach_alpha_deg_24 = Cmq_pdps_vs_Mach_alpha_deg_24['alpha_deg_24'].values

# Populate Cmalpha vectors for each AoA over the Mach values
Cmq_pdps_vs_Mach_AoA0_deg  = np.zeros(nMach_bps)
Cmq_pdps_vs_Mach_AoA8_deg  = np.zeros(nMach_bps)
Cmq_pdps_vs_Mach_AoA16_deg = np.zeros(nMach_bps)
Cmq_pdps_vs_Mach_AoA24_deg = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Cmq_pdps_vs_Mach_AoA0_deg[ii]  = fastInterp1(Mach_sample_for_alpha_deg_0,  Cmq_pdps_vs_Mach_alpha_deg_0,  Mach_bps[ii])
    Cmq_pdps_vs_Mach_AoA8_deg[ii]  = fastInterp1(Mach_sample_for_alpha_deg_8,  Cmq_pdps_vs_Mach_alpha_deg_8,  Mach_bps[ii])
    Cmq_pdps_vs_Mach_AoA16_deg[ii] = fastInterp1(Mach_sample_for_alpha_deg_16, Cmq_pdps_vs_Mach_alpha_deg_16, Mach_bps[ii])
    Cmq_pdps_vs_Mach_AoA24_deg[ii] = fastInterp1(Mach_sample_for_alpha_deg_24, Cmq_pdps_vs_Mach_alpha_deg_24, Mach_bps[ii])

# Reflect Cmq to negative AoA
Cmq_pdps_vs_Mach_AoAm8_deg  = Cmq_pdps_vs_Mach_AoA8_deg
Cmq_pdps_vs_Mach_AoAm16_deg = Cmq_pdps_vs_Mach_AoA16_deg
Cmq_pdps_vs_Mach_AoAm24_deg = Cmq_pdps_vs_Mach_AoA24_deg
    
# Combine data into a single 2D array 
Cmq_combined_pdps_vs_alpha_Mach_deg = np.column_stack((Cmq_pdps_vs_Mach_AoAm24_deg, Cmq_pdps_vs_Mach_AoAm16_deg, Cmq_pdps_vs_Mach_AoAm8_deg, \
                                                        Cmq_pdps_vs_Mach_AoA0_deg,  Cmq_pdps_vs_Mach_AoA8_deg,   Cmq_pdps_vs_Mach_AoA16_deg,  \
                                                        Cmq_pdps_vs_Mach_AoA24_deg))

# Interpolate table Cmalpha_pdeg_vs_Mach_alpha_deg to final aerodynamic database breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Cmq_pdps_table_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Cmq_pdps_table_alpha_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, Cmq_combined_pdps_vs_alpha_Mach_deg, Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Cmq_pdeg_vs_Mach_plot == 'on':
    plt.figure("Pitch Damping - Cmq [1/dps]")
    plt.plot(Mach_sample_for_alpha_deg_0,  Cmq_pdps_vs_Mach_alpha_deg_0,  color='black',   linestyle='solid', label="AoA 0 [deg] - Extracted")
    plt.plot(Mach_sample_for_alpha_deg_8,  Cmq_pdps_vs_Mach_alpha_deg_8,  color='blue',    linestyle='solid', label="AoA 8 [deg] - Extracted")
    plt.plot(Mach_sample_for_alpha_deg_16, Cmq_pdps_vs_Mach_alpha_deg_16, color='green',   linestyle='solid', label="AoA 16 [deg] - Extracted")
    plt.plot(Mach_sample_for_alpha_deg_24, Cmq_pdps_vs_Mach_alpha_deg_24, color='red',     linestyle='solid', label="AoA 24 [deg] - Extracted")
    plt.plot(Mach_bps, Cmq_pdps_table_alpha_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cmq_pdps_table_alpha_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cmq_pdps_table_alpha_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Cmq_pdps_table_alpha_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Cmq_pdps_table_alpha_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA 8 [deg] - Interpolated",  marker='*')
    plt.plot(Mach_bps, Cmq_pdps_table_alpha_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA 16 [deg] - Interpolated", marker='D', markerfacecolor='none')
    plt.plot(Mach_bps, Cmq_pdps_table_alpha_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA 24 [deg] - Interpolated", marker='.')
    plt.xlabel("Mach")
    plt.ylabel("Cmq_pdps")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 5.    Coefficient: Cmdele_pdeg = f(alpha_bps_deg, Mach_bps)
#       Method of determination: Computational
#       Details: Change in pitch moment due to change in incidence on elevator (angle of attack + elevator deflection)
#       Source: Walker 60, report page 77
#-------------------------------------------------------------------------------

# Cmalpha table from comma separated variables exported from plot digitizer
Cmdele_for_AoApdele_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/Cmdele_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample                        = Cmdele_for_AoApdele_Mach['x'].values
Cmdele_vs_Mach_pdeg_AoApdele0_deg  = Cmdele_for_AoApdele_Mach['alpha_p_dele_0'].values
Cmdele_vs_Mach_pdeg_AoApdele8_deg  = Cmdele_for_AoApdele_Mach['alpha_p_dele_8'].values
Cmdele_vs_Mach_pdeg_AoApdele16_deg = Cmdele_for_AoApdele_Mach['alpha_p_dele_16'].values
Cmdele_vs_Mach_pdeg_AoApdele24_deg = Cmdele_for_AoApdele_Mach['alpha_p_dele_24'].values

# Populate Cmalpha vectors for each AoA over the Mach values
Cmdele_vs_Mach_AoApdele0_deg  = np.zeros(nMach_bps)
Cmdele_vs_Mach_AoApdele8_deg  = np.zeros(nMach_bps)
Cmdele_vs_Mach_AoApdele16_deg = np.zeros(nMach_bps)
Cmdele_vs_Mach_AoApdele24_deg = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Cmdele_vs_Mach_AoApdele0_deg[ii]  = fastInterp1(Mach_sample, Cmdele_vs_Mach_pdeg_AoApdele0_deg,  Mach_bps[ii])
    Cmdele_vs_Mach_AoApdele8_deg[ii]  = fastInterp1(Mach_sample, Cmdele_vs_Mach_pdeg_AoApdele8_deg,  Mach_bps[ii])
    Cmdele_vs_Mach_AoApdele16_deg[ii] = fastInterp1(Mach_sample, Cmdele_vs_Mach_pdeg_AoApdele16_deg, Mach_bps[ii])
    Cmdele_vs_Mach_AoApdele24_deg[ii] = fastInterp1(Mach_sample, Cmdele_vs_Mach_pdeg_AoApdele24_deg, Mach_bps[ii])

# Reflect Cmalpha to negative AoA
Cmdele_vs_Mach_AoApdelem8_deg  = -Cmdele_vs_Mach_AoApdele8_deg
Cmdele_vs_Mach_AoApdelem16_deg = -Cmdele_vs_Mach_AoApdele16_deg
Cmdele_vs_Mach_AoApdelem24_deg = -Cmdele_vs_Mach_AoApdele24_deg
    
# Combine data into a single 2D array 
Cmdele_combined_pdeg_vs_Mach_AoApdele_deg = np.column_stack((Cmdele_vs_Mach_AoApdelem24_deg, Cmdele_vs_Mach_AoApdelem16_deg, Cmdele_vs_Mach_AoApdelem8_deg, \
                                                            Cmdele_vs_Mach_AoApdele0_deg,  Cmdele_vs_Mach_AoApdele8_deg,   Cmdele_vs_Mach_AoApdele16_deg,  \
                                                            Cmdele_vs_Mach_AoApdele24_deg))

# Interpolate table Cmalpha_pdeg_vs_Mach_alpha_deg to final aerodynamic database breakpoints for Mach and alpha
alphapdele_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Cmdele_pdeg_table_AoApdele_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Cmdele_pdeg_table_AoApdele_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                            Cmdele_combined_pdeg_vs_Mach_AoApdele_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Cmdele_vs_Mach_plot == 'on':
    plt.figure(4)
    plt.plot(Mach_sample, Cmdele_vs_Mach_pdeg_AoApdele0_deg,  color='black',   linestyle='solid', label="AoA + dele 0 [deg] - Extracted")
    plt.plot(Mach_sample, Cmdele_vs_Mach_pdeg_AoApdele8_deg,  color='blue',    linestyle='solid', label="AoA + dele 8 [deg] - Extracted")
    plt.plot(Mach_sample, Cmdele_vs_Mach_pdeg_AoApdele16_deg, color='green',   linestyle='solid', label="AoA + dele 16 [deg] - Extracted")
    plt.plot(Mach_sample, Cmdele_vs_Mach_pdeg_AoApdele24_deg, color='red',     linestyle='solid', label="AoA + dele 24 [deg] - Extracted")
    plt.plot(Mach_bps, Cmdele_pdeg_table_AoApdele_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA + dele -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cmdele_pdeg_table_AoApdele_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA + dele -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cmdele_pdeg_table_AoApdele_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA + dele -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Cmdele_pdeg_table_AoApdele_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA + dele 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Cmdele_pdeg_table_AoApdele_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA + dele 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cmdele_pdeg_table_AoApdele_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA + dele 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cmdele_pdeg_table_AoApdele_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA + dele 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("Cmdele_pdeg")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 6.    Coefficient: Clbeta_pdeg = f(alpha_bps_deg, Mach_bps)
#       Method of determination: Computational
#       Details: Change in roll moment due to change in sideslip
#       Source: Walker 60, report page 92
#-------------------------------------------------------------------------------

# Clbeta Table from comma separated variables exported from plot digitizer
Clbeta_for_AoA_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/Clbeta_v_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample                  = Clbeta_for_AoA_Mach['x'].values
Clbeta_vs_Mach_deg_AoA0_deg  = Clbeta_for_AoA_Mach['alpha_deg_0'].values
Clbeta_vs_Mach_deg_AoA8_deg  = Clbeta_for_AoA_Mach['alpha_deg_8'].values
Clbeta_vs_Mach_deg_AoA16_deg = Clbeta_for_AoA_Mach['alpha_deg_16'].values
Clbeta_vs_Mach_deg_AoA24_deg = Clbeta_for_AoA_Mach['alpha_deg_24'].values

# Populate Clbeta vectors for each AoA over the Mach values
Clbeta_v_Mach_AoA0_deg  = np.zeros(nMach_bps)
Clbeta_v_Mach_AoA8_deg  = np.zeros(nMach_bps)
Clbeta_v_Mach_AoA16_deg = np.zeros(nMach_bps)
Clbeta_v_Mach_AoA24_deg = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Clbeta_v_Mach_AoA0_deg[ii]  = fastInterp1(Mach_sample, Clbeta_vs_Mach_deg_AoA0_deg,  Mach_bps[ii])
    Clbeta_v_Mach_AoA8_deg[ii]  = fastInterp1(Mach_sample, Clbeta_vs_Mach_deg_AoA8_deg,  Mach_bps[ii])
    Clbeta_v_Mach_AoA16_deg[ii] = fastInterp1(Mach_sample, Clbeta_vs_Mach_deg_AoA16_deg, Mach_bps[ii])
    Clbeta_v_Mach_AoA24_deg[ii] = fastInterp1(Mach_sample, Clbeta_vs_Mach_deg_AoA24_deg, Mach_bps[ii])

# Reflect Clbeta to negative AoA
Clbeta_v_Mach_AoAm8_deg  = -Clbeta_v_Mach_AoA8_deg
Clbeta_v_Mach_AoAm16_deg = -Clbeta_v_Mach_AoA16_deg
Clbeta_v_Mach_AoAm24_deg = -Clbeta_v_Mach_AoA24_deg
    
# Combine data into a single 2D array
Clbeta_pdeg_vs_Mach_alpha_deg = np.column_stack((Clbeta_v_Mach_AoAm24_deg, Clbeta_v_Mach_AoAm16_deg, Clbeta_v_Mach_AoAm8_deg, \
                                                    Clbeta_v_Mach_AoA0_deg, Clbeta_v_Mach_AoA8_deg, Clbeta_v_Mach_AoA16_deg,  \
                                                    Clbeta_v_Mach_AoA24_deg))

# Interpolate table Clbeta_pdeg_vs_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Clbeta_pdeg_table_Mach_alpha_deg = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Clbeta_pdeg_table_Mach_alpha_deg[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                            Clbeta_pdeg_vs_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Clbeta_vs_Mach_alpha_plot == 'on':
    plt.figure(5)
    plt.plot(Mach_sample, Clbeta_vs_Mach_deg_AoA0_deg,  color='black',   linestyle='solid', label="AoA 0 [deg] - Extracted")
    plt.plot(Mach_sample, Clbeta_vs_Mach_deg_AoA8_deg,  color='blue',    linestyle='solid', label="AoA 8 [deg] - Extracted")
    plt.plot(Mach_sample, Clbeta_vs_Mach_deg_AoA16_deg, color='green',   linestyle='solid', label="AoA 16 [deg] - Extracted")
    plt.plot(Mach_sample, Clbeta_vs_Mach_deg_AoA24_deg, color='red',     linestyle='solid', label="AoA 24 [deg] - Extracted")
    plt.plot(Mach_bps, Clbeta_pdeg_table_Mach_alpha_deg[1,:],  color='grey', linestyle='dashed', label="AoA -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Clbeta_pdeg_table_Mach_alpha_deg[9,:],  color='grey', linestyle='dashed', label="AoA -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Clbeta_pdeg_table_Mach_alpha_deg[17,:], color='grey', linestyle='dashed', label="AoA -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Clbeta_pdeg_table_Mach_alpha_deg[25,:], color='grey', linestyle='dashed', label="AoA 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Clbeta_pdeg_table_Mach_alpha_deg[33,:], color='grey', linestyle='dashed', label="AoA 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Clbeta_pdeg_table_Mach_alpha_deg[41,:], color='grey', linestyle='dashed', label="AoA 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Clbeta_pdeg_table_Mach_alpha_deg[49,:], color='grey', linestyle='dashed', label="AoA 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("Clbeta_pdeg")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        
# 7. CD versus AoA for various Mach numbers. There are a few steps given how the data is provided
# in Saltzmann 1966. At this time, a better means of drag data for the X15 is not known. 
#
#   a. Get CL versus AoA. 
#   b. Get CL versus CD. 
#   c. Note that CL v AoA and CL v CD are connected via plot in the report so that CD = f(CL(AoA)) = f(AoA) can be obtained.
#   d. Interpoloate CD to AoA and Mach numbers over desired breakpoints. Export as a table.
#   e. Now you can pick up anything.
#
#-----------------------------------------------------------------------------------------------

# a. CL v AoA from comma separated variables exported from plot digitizer
CL_v_alpha_M1p1 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_alpha_deg_M1p1_Saltzmann66.csv', header = 0)
CL_v_alpha_M1p7 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_alpha_deg_M1p7_Saltzmann66.csv', header = 0)
CL_v_alpha_M1p9 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_alpha_deg_M1p9_Saltzmann66.csv', header = 0)
CL_v_alpha_M2p4 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_alpha_deg_M2p4_Saltzmann66.csv', header = 0)
CL_v_alpha_M2p6 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_alpha_deg_M2p6_Saltzmann66.csv', header = 0)
CL_v_alpha_M2p9 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_alpha_deg_M2p9_Saltzmann66.csv', header = 0)
CL_v_alpha_M3p1 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_alpha_deg_M3p1_Saltzmann66.csv', header = 0)
CL_v_alpha_M3p3 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_alpha_deg_M3p3_Saltzmann66.csv', header = 0)
CL_v_alpha_M3p7 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_alpha_deg_M3p7_Saltzmann66.csv', header = 0)
CL_v_alpha_M4p1 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_alpha_deg_M4p1_Saltzmann66.csv', header = 0)
CL_v_alpha_M5p5 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_alpha_deg_M5p5_Saltzmann66.csv', header = 0)
CL_v_alpha_M6p0 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_alpha_deg_M6p0_Saltzmann66.csv', header = 0)

# AoA bps for CD (points to interpolate to, what the final table is in terms of)
alpha_bps_for_CD_deg  = np.arange(0, 25, 1, dtype = float)
nalpha_bps_for_CD_deg = len(alpha_bps_for_CD_deg)

# Mach bps for CD (points to interpolate to, what the final table is in terms of)
Mach_bps_for_CD = np.arange(1, 8, 1, dtype = float)
nMach_bps_for_CD = len(Mach_bps_for_CD)

# Mach data points from extracted data (needed for interpolation)
Mach_data_for_CD = np.array([1.1, 1.7, 1.9, 2.4, 2.6, 2.9, 3.1, 3.3, 3.7, 4.1, 5.5, 6.0])
nMach_data_for_CD = len(Mach_data_for_CD)

# Get data sequences from the pandas DataFrame. Each data sequence has its own AoA points
# from the digitization process. 
alpha_sample_M1p1 = CL_v_alpha_M1p1['x'].values
CL_v_alpha_M1p1_Saltzmann66 = CL_v_alpha_M1p1['CL_vs_alpha_deg_M1p1_Saltzmann66'].values
alpha_sample_M1p7 = CL_v_alpha_M1p7['x'].values
CL_v_alpha_M1p7_Saltzmann66 = CL_v_alpha_M1p7['CL_vs_alpha_deg_M1p7_Saltzmann66'].values
alpha_sample_M1p9 = CL_v_alpha_M1p9['x'].values
CL_v_alpha_M1p9_Saltzmann66 = CL_v_alpha_M1p9['CL_vs_alpha_deg_M1p9_Saltzmann66'].values
alpha_sample_M2p4 = CL_v_alpha_M2p4['x'].values
CL_v_alpha_M2p4_Saltzmann66 = CL_v_alpha_M2p4['CL_vs_alpha_deg_M2p4_Saltzmann66'].values
alpha_sample_M2p6 = CL_v_alpha_M2p6['x'].values
CL_v_alpha_M2p6_Saltzmann66 = CL_v_alpha_M2p6['CL_vs_alpha_deg_M2p6_Saltzmann66'].values
alpha_sample_M2p9 = CL_v_alpha_M2p9['x'].values
CL_v_alpha_M2p9_Saltzmann66 = CL_v_alpha_M2p9['CL_vs_alpha_deg_M2p9_Saltzmann66'].values
alpha_sample_M3p1 = CL_v_alpha_M3p1['x'].values
CL_v_alpha_M3p1_Saltzmann66 = CL_v_alpha_M3p1['CL_vs_alpha_deg_M3p1_Saltzmann66'].values
alpha_sample_M3p3 = CL_v_alpha_M3p3['x'].values
CL_v_alpha_M3p3_Saltzmann66 = CL_v_alpha_M3p3['CL_vs_alpha_deg_M3p3_Saltzmann66'].values
alpha_sample_M3p7 = CL_v_alpha_M3p7['x'].values
CL_v_alpha_M3p7_Saltzmann66 = CL_v_alpha_M3p7['CL_vs_alpha_deg_M3p7_Saltzmann66'].values
alpha_sample_M4p1 = CL_v_alpha_M4p1['x'].values
CL_v_alpha_M4p1_Saltzmann66 = CL_v_alpha_M4p1['CL_vs_alpha_deg_M4p1_Saltzmann66'].values
alpha_sample_M5p5 = CL_v_alpha_M5p5['x'].values
CL_v_alpha_M5p5_Saltzmann66 = CL_v_alpha_M5p5['CL_vs_alpha_deg_M5p5_Saltzmann66'].values
alpha_sample_M6p0 = CL_v_alpha_M6p0['x'].values
CL_v_alpha_M6p0_Saltzmann66 = CL_v_alpha_M6p0['CL_vs_alpha_deg_M6p0_Saltzmann66'].values

# We preallocate CL v AoA vectors in preparation to fill them with a loop
CL_v_alpha_M1p1_interp = np.zeros(nalpha_bps_for_CD_deg)
CL_v_alpha_M1p7_interp = np.zeros(nalpha_bps_for_CD_deg) 
CL_v_alpha_M1p9_interp = np.zeros(nalpha_bps_for_CD_deg)
CL_v_alpha_M2p4_interp = np.zeros(nalpha_bps_for_CD_deg)
CL_v_alpha_M2p6_interp = np.zeros(nalpha_bps_for_CD_deg)
CL_v_alpha_M2p9_interp = np.zeros(nalpha_bps_for_CD_deg)
CL_v_alpha_M3p1_interp = np.zeros(nalpha_bps_for_CD_deg)
CL_v_alpha_M3p3_interp = np.zeros(nalpha_bps_for_CD_deg)
CL_v_alpha_M3p7_interp = np.zeros(nalpha_bps_for_CD_deg)
CL_v_alpha_M4p1_interp = np.zeros(nalpha_bps_for_CD_deg)
CL_v_alpha_M5p5_interp = np.zeros(nalpha_bps_for_CD_deg)
CL_v_alpha_M6p0_interp = np.zeros(nalpha_bps_for_CD_deg)

# Interpolate digitized data to alpha breakpoints 0 to 24
for ii in range(0, nalpha_bps_for_CD_deg, 1):
    CL_v_alpha_M1p1_interp[ii] = fastInterp1(alpha_sample_M1p1, CL_v_alpha_M1p1_Saltzmann66, alpha_bps_for_CD_deg[ii])
    CL_v_alpha_M1p7_interp[ii] = fastInterp1(alpha_sample_M1p7, CL_v_alpha_M1p7_Saltzmann66, alpha_bps_for_CD_deg[ii])
    CL_v_alpha_M1p9_interp[ii] = fastInterp1(alpha_sample_M1p9, CL_v_alpha_M1p9_Saltzmann66, alpha_bps_for_CD_deg[ii])
    CL_v_alpha_M2p4_interp[ii] = fastInterp1(alpha_sample_M2p4, CL_v_alpha_M2p4_Saltzmann66, alpha_bps_for_CD_deg[ii])
    CL_v_alpha_M2p6_interp[ii] = fastInterp1(alpha_sample_M2p6, CL_v_alpha_M2p6_Saltzmann66, alpha_bps_for_CD_deg[ii])
    CL_v_alpha_M2p9_interp[ii] = fastInterp1(alpha_sample_M2p9, CL_v_alpha_M2p9_Saltzmann66, alpha_bps_for_CD_deg[ii])
    CL_v_alpha_M3p1_interp[ii] = fastInterp1(alpha_sample_M3p1, CL_v_alpha_M3p1_Saltzmann66, alpha_bps_for_CD_deg[ii])
    CL_v_alpha_M3p3_interp[ii] = fastInterp1(alpha_sample_M3p3, CL_v_alpha_M3p3_Saltzmann66, alpha_bps_for_CD_deg[ii])
    CL_v_alpha_M3p7_interp[ii] = fastInterp1(alpha_sample_M3p7, CL_v_alpha_M3p7_Saltzmann66, alpha_bps_for_CD_deg[ii])
    CL_v_alpha_M4p1_interp[ii] = fastInterp1(alpha_sample_M4p1, CL_v_alpha_M4p1_Saltzmann66, alpha_bps_for_CD_deg[ii])
    CL_v_alpha_M5p5_interp[ii] = fastInterp1(alpha_sample_M5p5, CL_v_alpha_M5p5_Saltzmann66, alpha_bps_for_CD_deg[ii])
    CL_v_alpha_M6p0_interp[ii] = fastInterp1(alpha_sample_M6p0, CL_v_alpha_M6p0_Saltzmann66, alpha_bps_for_CD_deg[ii])
    
# 2. Import CL versus CD data that's extracted from the plot digitizer
CL_v_CD_M1p1 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_CD_M1p1_Saltzmann66.csv', header = 0)
CL_v_CD_M1p7 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_CD_M1p7_Saltzmann66.csv', header = 0)
CL_v_CD_M1p9 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_CD_M1p9_Saltzmann66.csv', header = 0)
CL_v_CD_M2p4 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_CD_M2p4_Saltzmann66.csv', header = 0)
CL_v_CD_M2p6 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_CD_M2p6_Saltzmann66.csv', header = 0)
CL_v_CD_M2p9 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_CD_M2p9_Saltzmann66.csv', header = 0)
CL_v_CD_M3p1 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_CD_M3p1_Saltzmann66.csv', header = 0)
CL_v_CD_M3p3 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_CD_M3p3_Saltzmann66.csv', header = 0)
CL_v_CD_M3p7 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_CD_M3p7_Saltzmann66.csv', header = 0)
CL_v_CD_M4p1 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_CD_M4p1_Saltzmann66.csv', header = 0)
CL_v_CD_M5p5 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_CD_M5p5_Saltzmann66.csv', header = 0)
CL_v_CD_M6p0 = pd.read_csv('models/X15/aerodynamic_model/Saltzman66/data/CL_v_CD_M6p0_Saltzmann66.csv', header = 0)

# Get data sequences from the pandas DataFrame
CD_sample_M1p1 = CL_v_CD_M1p1['x'].values
CL_v_CD_M1p1_Saltzmann66  = CL_v_CD_M1p1['CL_vs_CD_M1p1_Saltzmann66'].values
CD_sample_M1p7 = CL_v_CD_M1p7['x'].values
CL_v_CD_M1p7_Saltzmann66  = CL_v_CD_M1p7['CL_vs_CD_M1p7_Saltzmann66'].values
CD_sample_M1p9 = CL_v_CD_M1p9['x'].values
CL_v_CD_M1p9_Saltzmann66  = CL_v_CD_M1p9['CL_vs_CD_M1p9_Saltzmann66'].values
CD_sample_M2p4 = CL_v_CD_M2p4['x'].values
CL_v_CD_M2p4_Saltzmann66  = CL_v_CD_M2p4['CL_vs_CD_M2p4_Saltzmann66'].values
CD_sample_M2p6 = CL_v_CD_M2p6['x'].values
CL_v_CD_M2p6_Saltzmann66  = CL_v_CD_M2p6['CL_vs_CD_M2p6_Saltzmann66'].values
CD_sample_M2p9 = CL_v_CD_M2p9['x'].values
CL_v_CD_M2p9_Saltzmann66  = CL_v_CD_M2p9['CL_vs_CD_M2p9_Saltzmann66'].values
CD_sample_M3p1 = CL_v_CD_M3p1['x'].values
CL_v_CD_M3p1_Saltzmann66  = CL_v_CD_M3p1['CL_vs_CD_M3p1_Saltzmann66'].values
CD_sample_M3p3 = CL_v_CD_M3p3['x'].values
CL_v_CD_M3p3_Saltzmann66  = CL_v_CD_M3p3['CL_vs_CD_M3p3_Saltzmann66'].values
CD_sample_M3p7 = CL_v_CD_M3p7['x'].values
CL_v_CD_M3p7_Saltzmann66  = CL_v_CD_M3p7['CL_vs_CD_M3p7_Saltzmann66'].values
CD_sample_M4p1 = CL_v_CD_M4p1['x'].values
CL_v_CD_M4p1_Saltzmann66  = CL_v_CD_M4p1['CL_vs_CD_M4p1_Saltzmann66'].values
CD_sample_M5p5 = CL_v_CD_M5p5['x'].values
CL_v_CD_M5p5_Saltzmann66  = CL_v_CD_M5p5['CL_vs_CD_M5p5_Saltzmann66'].values
CD_sample_M6p0 = CL_v_CD_M6p0['x'].values
CL_v_CD_M6p0_Saltzmann66  = CL_v_CD_M6p0['CL_vs_CD_M6p0_Saltzmann66'].values

# Preallocate the CD v AoA vectors 
CD_v_alpha_M1p1_interp = np.zeros(nalpha_bps_for_CD_deg)
CD_v_alpha_M1p7_interp = np.zeros(nalpha_bps_for_CD_deg)
CD_v_alpha_M1p9_interp = np.zeros(nalpha_bps_for_CD_deg)
CD_v_alpha_M2p4_interp = np.zeros(nalpha_bps_for_CD_deg)
CD_v_alpha_M2p6_interp = np.zeros(nalpha_bps_for_CD_deg)
CD_v_alpha_M2p9_interp = np.zeros(nalpha_bps_for_CD_deg)
CD_v_alpha_M3p1_interp = np.zeros(nalpha_bps_for_CD_deg)
CD_v_alpha_M3p3_interp = np.zeros(nalpha_bps_for_CD_deg)
CD_v_alpha_M3p7_interp = np.zeros(nalpha_bps_for_CD_deg)
CD_v_alpha_M4p1_interp = np.zeros(nalpha_bps_for_CD_deg)
CD_v_alpha_M5p5_interp = np.zeros(nalpha_bps_for_CD_deg)
CD_v_alpha_M6p0_interp = np.zeros(nalpha_bps_for_CD_deg)

# Interpolate CD to AoA breakpoints 0 to 24 degrees - this means that CD and CL will have the 
# same AoA breakpoints so that CD versus alpha can be determined from CL versus CD.
for ii in range(0, nalpha_bps_for_CD_deg, 1):
    CD_v_alpha_M1p1_interp[ii] = fastInterp1(CL_v_CD_M1p1_Saltzmann66, CD_sample_M1p1, CL_v_alpha_M1p1_interp[ii])
    CD_v_alpha_M1p7_interp[ii] = fastInterp1(CL_v_CD_M1p7_Saltzmann66, CD_sample_M1p7, CL_v_alpha_M1p7_interp[ii])
    CD_v_alpha_M1p9_interp[ii] = fastInterp1(CL_v_CD_M1p9_Saltzmann66, CD_sample_M1p9, CL_v_alpha_M1p9_interp[ii])
    CD_v_alpha_M2p4_interp[ii] = fastInterp1(CL_v_CD_M2p4_Saltzmann66, CD_sample_M2p4, CL_v_alpha_M2p4_interp[ii])
    CD_v_alpha_M2p6_interp[ii] = fastInterp1(CL_v_CD_M2p6_Saltzmann66, CD_sample_M2p6, CL_v_alpha_M2p6_interp[ii])
    CD_v_alpha_M2p9_interp[ii] = fastInterp1(CL_v_CD_M2p9_Saltzmann66, CD_sample_M2p9, CL_v_alpha_M2p9_interp[ii])
    CD_v_alpha_M3p1_interp[ii] = fastInterp1(CL_v_CD_M3p1_Saltzmann66, CD_sample_M3p1, CL_v_alpha_M3p1_interp[ii])
    CD_v_alpha_M3p3_interp[ii] = fastInterp1(CL_v_CD_M3p3_Saltzmann66, CD_sample_M3p3, CL_v_alpha_M3p3_interp[ii])
    CD_v_alpha_M3p7_interp[ii] = fastInterp1(CL_v_CD_M3p7_Saltzmann66, CD_sample_M3p7, CL_v_alpha_M3p7_interp[ii])
    CD_v_alpha_M4p1_interp[ii] = fastInterp1(CL_v_CD_M4p1_Saltzmann66, CD_sample_M4p1, CL_v_alpha_M4p1_interp[ii])
    CD_v_alpha_M5p5_interp[ii] = fastInterp1(CL_v_CD_M5p5_Saltzmann66, CD_sample_M5p5, CL_v_alpha_M5p5_interp[ii])
    CD_v_alpha_M6p0_interp[ii] = fastInterp1(CL_v_CD_M6p0_Saltzmann66, CD_sample_M6p0, CL_v_alpha_M6p0_interp[ii])

# Reflect CD_v_alpha to negative angles of attack, now the range is -24 to 24 degrees
CD_v_alpha_M1p1_interp = np.concatenate((CD_v_alpha_M1p1_interp[::-1][:-1], CD_v_alpha_M1p1_interp))
CD_v_alpha_M1p7_interp = np.concatenate((CD_v_alpha_M1p7_interp[::-1][:-1], CD_v_alpha_M1p7_interp))
CD_v_alpha_M1p9_interp = np.concatenate((CD_v_alpha_M1p9_interp[::-1][:-1], CD_v_alpha_M1p9_interp))
CD_v_alpha_M2p4_interp = np.concatenate((CD_v_alpha_M2p4_interp[::-1][:-1], CD_v_alpha_M2p4_interp))
CD_v_alpha_M2p6_interp = np.concatenate((CD_v_alpha_M2p6_interp[::-1][:-1], CD_v_alpha_M2p6_interp))
CD_v_alpha_M2p9_interp = np.concatenate((CD_v_alpha_M2p9_interp[::-1][:-1], CD_v_alpha_M2p9_interp))
CD_v_alpha_M3p1_interp = np.concatenate((CD_v_alpha_M3p1_interp[::-1][:-1], CD_v_alpha_M3p1_interp))
CD_v_alpha_M3p3_interp = np.concatenate((CD_v_alpha_M3p3_interp[::-1][:-1], CD_v_alpha_M3p3_interp))
CD_v_alpha_M3p7_interp = np.concatenate((CD_v_alpha_M3p7_interp[::-1][:-1], CD_v_alpha_M3p7_interp))
CD_v_alpha_M4p1_interp = np.concatenate((CD_v_alpha_M4p1_interp[::-1][:-1], CD_v_alpha_M4p1_interp))
CD_v_alpha_M5p5_interp = np.concatenate((CD_v_alpha_M5p5_interp[::-1][:-1], CD_v_alpha_M5p5_interp))
CD_v_alpha_M6p0_interp = np.concatenate((CD_v_alpha_M6p0_interp[::-1][:-1], CD_v_alpha_M6p0_interp))

# Combine data into a single 2D array
CD_v_alpha_Mach = np.column_stack((CD_v_alpha_M1p1_interp, \
                                    CD_v_alpha_M1p7_interp, \
                                    CD_v_alpha_M1p9_interp, \
                                    CD_v_alpha_M2p4_interp, \
                                    CD_v_alpha_M2p6_interp, \
                                    CD_v_alpha_M2p9_interp, \
                                    CD_v_alpha_M3p1_interp, \
                                    CD_v_alpha_M3p3_interp, \
                                    CD_v_alpha_M3p7_interp, \
                                    CD_v_alpha_M4p1_interp, \
                                    CD_v_alpha_M5p5_interp, \
                                    CD_v_alpha_M6p0_interp, 
                                    ))

# 4. Interpolate table CD_table_v_alpha_Mach to breakpoints for Mach and alpha
CD_table_v_alpha_Mach = np.zeros((nalpha_bps_deg, nMach_bps))

alpha_deg_reflected = np.arange(-24, 25, 1)
for ii in range(0, nalpha_bps_deg, 1):
    for jj in range(0, nMach_bps, 1):
        CD_table_v_alpha_Mach[ii, jj]  = fastInterp2(alpha_deg_reflected, Mach_data_for_CD, CD_v_alpha_Mach, alpha_bps_deg[ii], Mach_bps[jj])
        
# Compare the results on a single plot
if CL_vs_alpha_plot == 'on':
    plt.figure(6)
    plt.plot(alpha_sample_M1p9, CL_v_alpha_M1p9_Saltzmann66, color='black',  linestyle='solid', label="Mach 1.9 - Extracted")
    plt.plot(alpha_sample_M2p4, CL_v_alpha_M2p4_Saltzmann66, color='blue',   linestyle='solid', label="Mach 2.4 - Extracted")
    plt.plot(alpha_sample_M2p6, CL_v_alpha_M2p6_Saltzmann66, color='green',  linestyle='solid', label="Mach 2.6 - Extracted")
    plt.plot(alpha_sample_M2p9, CL_v_alpha_M2p9_Saltzmann66, color='red',    linestyle='solid', label="Mach 2.9 - Extracted")   
    plt.plot(alpha_sample_M3p1, CL_v_alpha_M3p1_Saltzmann66, color='m',      linestyle='solid', label="Mach 3.1 - Extracted") 
    plt.plot(alpha_bps_for_CD_deg, CL_v_alpha_M1p9_interp, color='grey', linestyle='dashed', label="Mach 1.9 - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(alpha_bps_for_CD_deg, CL_v_alpha_M2p4_interp, color='grey', linestyle='dashed', label="Mach 2.4 - Interpolated",  marker='x')
    plt.plot(alpha_bps_for_CD_deg, CL_v_alpha_M2p6_interp, color='grey', linestyle='dashed', label="Mach 2.6 - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(alpha_bps_for_CD_deg, CL_v_alpha_M2p9_interp, color='grey', linestyle='dashed', label="Mach 2.9 - Interpolated", marker='+')
    plt.plot(alpha_bps_for_CD_deg, CL_v_alpha_M3p1_interp, color='grey', linestyle='dashed', label="Mach 3.1 - Interpolated", marker='D', markerfacecolor='none')
    plt.xlabel("AoA [deg]")
    plt.ylabel("CL")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    
if CD_vs_alpha_plot == 'on':
    plt.figure(7)
    plt.plot(alpha_bps_deg, CD_table_v_alpha_Mach[:,0], color='black', linestyle='dashed', label=f"Mach {Mach_bps_for_CD[0]} - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(alpha_bps_deg, CD_table_v_alpha_Mach[:,1], color='blue',  linestyle='dashed', label=f"Mach {Mach_bps_for_CD[1]} - Interpolated",  marker='x')
    plt.plot(alpha_bps_deg, CD_table_v_alpha_Mach[:,2], color='green', linestyle='dashed', label=f"Mach {Mach_bps_for_CD[2]} - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(alpha_bps_deg, CD_table_v_alpha_Mach[:,3], color='red',   linestyle='dashed', label=f"Mach {Mach_bps_for_CD[3]} - Interpolated", marker='+')
    plt.plot(alpha_bps_deg, CD_table_v_alpha_Mach[:,4], color='m',     linestyle='dashed', label=f"Mach {Mach_bps_for_CD[4]} - Interpolated", marker='D', markerfacecolor='none')
    plt.plot(alpha_bps_deg, CD_table_v_alpha_Mach[:,5], color='grey',  linestyle='dashed', label=f"Mach {Mach_bps_for_CD[5]} - Interpolated",  marker='.')
    plt.plot(alpha_bps_deg, CD_table_v_alpha_Mach[:,6], color='black', linestyle='dashed', label=f"Mach {Mach_bps_for_CD[6]} - Interpolated", marker='^', markerfacecolor='none')
    plt.xlabel("AoA [deg]")
    plt.ylabel("CD")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 8.    Coefficient: CYbeta [1/deg] = f(alpha_deg, Mach)
#       Method of determination: Computational
#       Details: Change in sideforce due to change in sideslip, directional stability, 
#                positive for stability. System is directionally unstable. 
#       Source: Walker 60, report page 92
#------------------------------------------------------------------------------------------

# CYbeta table from comma separated variables exported from plot digitizer
CYbeta_pdeg_vs_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/CYbeta_pdeg_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample = CYbeta_pdeg_vs_Mach['x'].values
CYbeta_pdeg_vs_Mach_alpha_deg_0  = CYbeta_pdeg_vs_Mach['alpha_deg_0'].values
CYbeta_pdeg_vs_Mach_alpha_deg_8  = CYbeta_pdeg_vs_Mach['alpha_deg_8'].values
CYbeta_pdeg_vs_Mach_alpha_deg_16 = CYbeta_pdeg_vs_Mach['alpha_deg_16'].values
CYbeta_pdeg_vs_Mach_alpha_deg_24 = CYbeta_pdeg_vs_Mach['alpha_deg_24'].values

# We will populate CYbeta vectors for each AoA over the Mach values
CYbeta_pdeg_v_Mach_alpha_deg_0  = np.zeros(nMach_bps)
CYbeta_pdeg_v_Mach_alpha_deg_8  = np.zeros(nMach_bps)
CYbeta_pdeg_v_Mach_alpha_deg_16 = np.zeros(nMach_bps)
CYbeta_pdeg_v_Mach_alpha_deg_24 = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    CYbeta_pdeg_v_Mach_alpha_deg_0[ii]  = fastInterp1(Mach_sample, CYbeta_pdeg_vs_Mach_alpha_deg_0,  Mach_bps[ii])
    CYbeta_pdeg_v_Mach_alpha_deg_8[ii]  = fastInterp1(Mach_sample, CYbeta_pdeg_vs_Mach_alpha_deg_8,  Mach_bps[ii])
    CYbeta_pdeg_v_Mach_alpha_deg_16[ii] = fastInterp1(Mach_sample, CYbeta_pdeg_vs_Mach_alpha_deg_16, Mach_bps[ii])
    CYbeta_pdeg_v_Mach_alpha_deg_24[ii] = fastInterp1(Mach_sample, CYbeta_pdeg_vs_Mach_alpha_deg_24, Mach_bps[ii])
    
# Reflect CYbeta to negative AoA
CYbeta_pdeg_v_Mach_alpha_deg_m8  = CYbeta_pdeg_v_Mach_alpha_deg_8
CYbeta_pdeg_v_Mach_alpha_deg_m16 = CYbeta_pdeg_v_Mach_alpha_deg_16
CYbeta_pdeg_v_Mach_alpha_deg_m24 = CYbeta_pdeg_v_Mach_alpha_deg_24

# Combine data into a single 2D array
CYbeta_pdeg_v_Mach_alpha_deg = np.column_stack((CYbeta_pdeg_v_Mach_alpha_deg_m24, \
                                                    CYbeta_pdeg_v_Mach_alpha_deg_m16, \
                                                    CYbeta_pdeg_v_Mach_alpha_deg_m8, \
                                                    CYbeta_pdeg_v_Mach_alpha_deg_0, \
                                                    CYbeta_pdeg_v_Mach_alpha_deg_8, \
                                                    CYbeta_pdeg_v_Mach_alpha_deg_16, \
                                                    CYbeta_pdeg_v_Mach_alpha_deg_24))

# Interpolate table CYbeta_pdeg_v_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
CYbeta_table_pdeg_v_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        CYbeta_table_pdeg_v_alpha_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                    CYbeta_pdeg_v_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if CYbeta_pdeg_vs_Mach_plot == 'on':
    plt.figure('CYbeta [1/deg] - Stability Derivative')
    plt.plot(Mach_sample, CYbeta_pdeg_vs_Mach_alpha_deg_0,  color='black',  linestyle='solid', label="AoA + dele 0 [deg] - Extracted")
    plt.plot(Mach_sample, CYbeta_pdeg_vs_Mach_alpha_deg_8,  color='blue',   linestyle='solid', label="AoA + dele 8 [deg] - Extracted")
    plt.plot(Mach_sample, CYbeta_pdeg_vs_Mach_alpha_deg_16, color='green',  linestyle='solid', label="AoA + dele 16 [deg] - Extracted")
    plt.plot(Mach_sample, CYbeta_pdeg_vs_Mach_alpha_deg_24, color='red',    linestyle='solid', label="AoA + dele 24 [deg] - Extracted")   
    plt.plot(Mach_bps, CYbeta_table_pdeg_v_alpha_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA + dele -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, CYbeta_table_pdeg_v_alpha_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA + dele -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, CYbeta_table_pdeg_v_alpha_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA + dele -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, CYbeta_table_pdeg_v_alpha_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA + dele 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, CYbeta_table_pdeg_v_alpha_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA + dele 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, CYbeta_table_pdeg_v_alpha_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA + dele 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, CYbeta_table_pdeg_v_alpha_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA + dele 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("CYbeta_pdeg")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 9.    Coefficient: Cnbeta [1/deg] = f(alpha_deg, Mach)
#       Method of determination: Computational
#       Details: Change in yaw moment coefficient due to change in sideslip, positive 
#                valued for stability, primarily influenced by vertical tail, analagous 
#                to Cmalpha in vertical plane.
#       Source: Walker 60, report page 89
#------------------------------------------------------------------------------------------

# CYbeta table from comma separated variables exported from plot digitizer
Cnbeta_pdeg_vs_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/Cnbeta_pdeg_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample = Cnbeta_pdeg_vs_Mach['x'].values
Cnbeta_pdeg_vs_Mach_alpha_deg_0  = Cnbeta_pdeg_vs_Mach['alpha_deg_0'].values
Cnbeta_pdeg_vs_Mach_alpha_deg_8  = Cnbeta_pdeg_vs_Mach['alpha_deg_8'].values
Cnbeta_pdeg_vs_Mach_alpha_deg_16 = Cnbeta_pdeg_vs_Mach['alpha_deg_16'].values
Cnbeta_pdeg_vs_Mach_alpha_deg_24 = Cnbeta_pdeg_vs_Mach['alpha_deg_24'].values

# We will populate Cnbeta vectors for each AoA over the Mach values
Cnbeta_pdeg_v_Mach_alpha_deg_0  = np.zeros(nMach_bps)
Cnbeta_pdeg_v_Mach_alpha_deg_8  = np.zeros(nMach_bps)
Cnbeta_pdeg_v_Mach_alpha_deg_16 = np.zeros(nMach_bps)
Cnbeta_pdeg_v_Mach_alpha_deg_24 = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Cnbeta_pdeg_v_Mach_alpha_deg_0[ii]  = fastInterp1(Mach_sample, Cnbeta_pdeg_vs_Mach_alpha_deg_0,  Mach_bps[ii])
    Cnbeta_pdeg_v_Mach_alpha_deg_8[ii]  = fastInterp1(Mach_sample, Cnbeta_pdeg_vs_Mach_alpha_deg_8,  Mach_bps[ii])
    Cnbeta_pdeg_v_Mach_alpha_deg_16[ii] = fastInterp1(Mach_sample, Cnbeta_pdeg_vs_Mach_alpha_deg_16, Mach_bps[ii])
    Cnbeta_pdeg_v_Mach_alpha_deg_24[ii] = fastInterp1(Mach_sample, Cnbeta_pdeg_vs_Mach_alpha_deg_24, Mach_bps[ii])
    
# Reflect Cnbeta to negative AoA
Cnbeta_pdeg_v_Mach_alpha_deg_m8  = Cnbeta_pdeg_v_Mach_alpha_deg_8
Cnbeta_pdeg_v_Mach_alpha_deg_m16 = Cnbeta_pdeg_v_Mach_alpha_deg_16
Cnbeta_pdeg_v_Mach_alpha_deg_m24 = Cnbeta_pdeg_v_Mach_alpha_deg_24

# Combine data into a single 2D array
Cnbeta_pdeg_v_Mach_alpha_deg = np.column_stack((Cnbeta_pdeg_v_Mach_alpha_deg_m24, \
                                                    Cnbeta_pdeg_v_Mach_alpha_deg_m16, \
                                                    Cnbeta_pdeg_v_Mach_alpha_deg_m8, \
                                                    Cnbeta_pdeg_v_Mach_alpha_deg_0, \
                                                    Cnbeta_pdeg_v_Mach_alpha_deg_8, \
                                                    Cnbeta_pdeg_v_Mach_alpha_deg_16, \
                                                    Cnbeta_pdeg_v_Mach_alpha_deg_24))

# Interpolate table Cnbeta_pdeg_v_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Cnbeta_table_pdeg_v_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Cnbeta_table_pdeg_v_alpha_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                    Cnbeta_pdeg_v_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Cnbeta_pdeg_vs_Mach_plot == 'on':
    plt.figure('Cnbeta [1/deg] - Stability Derivative')
    plt.plot(Mach_sample, Cnbeta_pdeg_vs_Mach_alpha_deg_0,  color='black',  linestyle='solid', label="AoA + dele 0 [deg] - Extracted")
    plt.plot(Mach_sample, Cnbeta_pdeg_vs_Mach_alpha_deg_8,  color='blue',   linestyle='solid', label="AoA + dele 8 [deg] - Extracted")
    plt.plot(Mach_sample, Cnbeta_pdeg_vs_Mach_alpha_deg_16, color='green',  linestyle='solid', label="AoA + dele 16 [deg] - Extracted")
    plt.plot(Mach_sample, Cnbeta_pdeg_vs_Mach_alpha_deg_24, color='red',    linestyle='solid', label="AoA + dele 24 [deg] - Extracted")   
    plt.plot(Mach_bps, Cnbeta_table_pdeg_v_alpha_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA + dele -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cnbeta_table_pdeg_v_alpha_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA + dele -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cnbeta_table_pdeg_v_alpha_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA + dele -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Cnbeta_table_pdeg_v_alpha_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA + dele 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Cnbeta_table_pdeg_v_alpha_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA + dele 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cnbeta_table_pdeg_v_alpha_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA + dele 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cnbeta_table_pdeg_v_alpha_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA + dele 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("Cnbeta_pdeg")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 10.    Coefficient: CYdelr [1/deg] = f(alpha_deg, Mach)
#       Method of determination: Computational
#       Details: Change in sideforce due to change in rudder angle
#       Source: Walker 60, report page 113
#------------------------------------------------------------------------------------------

# CYdelr table from comma separated variables exported from plot digitizer
CYdelr_pdeg_vs_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/CYdelr_pdeg_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample = CYdelr_pdeg_vs_Mach['x'].values
CYdelr_pdeg_vs_Mach_alpha_deg_0  = CYdelr_pdeg_vs_Mach['alpha_deg_0'].values
CYdelr_pdeg_vs_Mach_alpha_deg_8  = CYdelr_pdeg_vs_Mach['alpha_deg_8'].values
CYdelr_pdeg_vs_Mach_alpha_deg_16 = CYdelr_pdeg_vs_Mach['alpha_deg_16'].values
CYdelr_pdeg_vs_Mach_alpha_deg_24 = CYdelr_pdeg_vs_Mach['alpha_deg_24'].values

# We will populate CYdelr vectors for each AoA over the Mach values
CYdelr_pdeg_v_Mach_alpha_deg_0  = np.zeros(nMach_bps)
CYdelr_pdeg_v_Mach_alpha_deg_8  = np.zeros(nMach_bps)
CYdelr_pdeg_v_Mach_alpha_deg_16 = np.zeros(nMach_bps)
CYdelr_pdeg_v_Mach_alpha_deg_24 = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    CYdelr_pdeg_v_Mach_alpha_deg_0[ii]  = fastInterp1(Mach_sample, CYdelr_pdeg_vs_Mach_alpha_deg_0,  Mach_bps[ii])
    CYdelr_pdeg_v_Mach_alpha_deg_8[ii]  = fastInterp1(Mach_sample, CYdelr_pdeg_vs_Mach_alpha_deg_8,  Mach_bps[ii])
    CYdelr_pdeg_v_Mach_alpha_deg_16[ii] = fastInterp1(Mach_sample, CYdelr_pdeg_vs_Mach_alpha_deg_16, Mach_bps[ii])
    CYdelr_pdeg_v_Mach_alpha_deg_24[ii] = fastInterp1(Mach_sample, CYdelr_pdeg_vs_Mach_alpha_deg_24, Mach_bps[ii])
    
# Reflect CYdelr to negative AoA
CYdelr_pdeg_v_Mach_alpha_deg_m8  = CYdelr_pdeg_v_Mach_alpha_deg_8
CYdelr_pdeg_v_Mach_alpha_deg_m16 = CYdelr_pdeg_v_Mach_alpha_deg_16
CYdelr_pdeg_v_Mach_alpha_deg_m24 = CYdelr_pdeg_v_Mach_alpha_deg_24

# Combine data into a single 2D array
CYdelr_pdeg_v_Mach_alpha_deg = np.column_stack((CYdelr_pdeg_v_Mach_alpha_deg_m24, \
                                                    CYdelr_pdeg_v_Mach_alpha_deg_m16, \
                                                    CYdelr_pdeg_v_Mach_alpha_deg_m8, \
                                                    CYdelr_pdeg_v_Mach_alpha_deg_0, \
                                                    CYdelr_pdeg_v_Mach_alpha_deg_8, \
                                                    CYdelr_pdeg_v_Mach_alpha_deg_16, \
                                                    CYdelr_pdeg_v_Mach_alpha_deg_24))

# Interpolate table CYdelr_pdeg_v_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
CYdelr_table_pdeg_v_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        CYdelr_table_pdeg_v_alpha_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                    CYdelr_pdeg_v_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if CYdelr_pdeg_vs_Mach_plot == 'on':
    plt.figure('CYdelr [1/deg] - Stability Derivative')
    plt.plot(Mach_sample, CYdelr_pdeg_vs_Mach_alpha_deg_0,  color='black',  linestyle='solid', label="AoA + dele 0 [deg] - Extracted")
    plt.plot(Mach_sample, CYdelr_pdeg_vs_Mach_alpha_deg_8,  color='blue',   linestyle='solid', label="AoA + dele 8 [deg] - Extracted")
    plt.plot(Mach_sample, CYdelr_pdeg_vs_Mach_alpha_deg_16, color='green',  linestyle='solid', label="AoA + dele 16 [deg] - Extracted")
    plt.plot(Mach_sample, CYdelr_pdeg_vs_Mach_alpha_deg_24, color='red',    linestyle='solid', label="AoA + dele 24 [deg] - Extracted")   
    plt.plot(Mach_bps, CYdelr_table_pdeg_v_alpha_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA + dele -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, CYdelr_table_pdeg_v_alpha_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA + dele -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, CYdelr_table_pdeg_v_alpha_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA + dele -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, CYdelr_table_pdeg_v_alpha_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA + dele 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, CYdelr_table_pdeg_v_alpha_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA + dele 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, CYdelr_table_pdeg_v_alpha_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA + dele 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, CYdelr_table_pdeg_v_alpha_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA + dele 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("CYdelr_pdeg")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 11.    Coefficient: CYp [1/(rad/s)] = f(alpha_deg, Mach)
#       Method of determination: Computational
#       Details: Change in sideforce due to change in roll rate - how much side force is 
#                generated when the aircraft rolls? Values are positive, therefore rolling
#                to the right creates positive side force. 
#       Source: Walker 60, report page 113
#------------------------------------------------------------------------------------------

# CYp table from comma separated variables exported from plot digitizer
CYp_prps_vs_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/CYp_prps_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample = CYp_prps_vs_Mach['x'].values
CYp_prps_vs_Mach_alpha_deg_0  = CYp_prps_vs_Mach['alpha_deg_0'].values
CYp_prps_vs_Mach_alpha_deg_8  = CYp_prps_vs_Mach['alpha_deg_8'].values
CYp_prps_vs_Mach_alpha_deg_16 = CYp_prps_vs_Mach['alpha_deg_16'].values
CYp_prps_vs_Mach_alpha_deg_24 = CYp_prps_vs_Mach['alpha_deg_24'].values

# We will populate CYp vectors for each AoA over the Mach values
CYp_prps_v_Mach_alpha_deg_0  = np.zeros(nMach_bps)
CYp_prps_v_Mach_alpha_deg_8  = np.zeros(nMach_bps)
CYp_prps_v_Mach_alpha_deg_16 = np.zeros(nMach_bps)
CYp_prps_v_Mach_alpha_deg_24 = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    CYp_prps_v_Mach_alpha_deg_0[ii]  = fastInterp1(Mach_sample, CYp_prps_vs_Mach_alpha_deg_0,  Mach_bps[ii])
    CYp_prps_v_Mach_alpha_deg_8[ii]  = fastInterp1(Mach_sample, CYp_prps_vs_Mach_alpha_deg_8,  Mach_bps[ii])
    CYp_prps_v_Mach_alpha_deg_16[ii] = fastInterp1(Mach_sample, CYp_prps_vs_Mach_alpha_deg_16, Mach_bps[ii])
    CYp_prps_v_Mach_alpha_deg_24[ii] = fastInterp1(Mach_sample, CYp_prps_vs_Mach_alpha_deg_24, Mach_bps[ii])
    
# Reflect CYp to negative AoA
CYp_prps_v_Mach_alpha_deg_m8  = CYp_prps_v_Mach_alpha_deg_8
CYp_prps_v_Mach_alpha_deg_m16 = CYp_prps_v_Mach_alpha_deg_16
CYp_prps_v_Mach_alpha_deg_m24 = CYp_prps_v_Mach_alpha_deg_24

# Combine data into a single 2D array
CYp_prps_v_Mach_alpha_deg = np.column_stack((CYp_prps_v_Mach_alpha_deg_m24, \
                                                    CYp_prps_v_Mach_alpha_deg_m16, \
                                                    CYp_prps_v_Mach_alpha_deg_m8, \
                                                    CYp_prps_v_Mach_alpha_deg_0, \
                                                    CYp_prps_v_Mach_alpha_deg_8, \
                                                    CYp_prps_v_Mach_alpha_deg_16, \
                                                    CYp_prps_v_Mach_alpha_deg_24))

# Interpolate table CYp_prps_v_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
CYp_table_prps_v_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        CYp_table_prps_v_alpha_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                    CYp_prps_v_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if CYp_prps_vs_Mach_plot == 'on':
    plt.figure('CYp [1/(rad/s)] - Stability Derivative')
    plt.plot(Mach_sample, CYp_prps_vs_Mach_alpha_deg_0,  color='black',  linestyle='solid', label="AoA + dele 0 [deg] - Extracted")
    plt.plot(Mach_sample, CYp_prps_vs_Mach_alpha_deg_8,  color='blue',   linestyle='solid', label="AoA + dele 8 [deg] - Extracted")
    plt.plot(Mach_sample, CYp_prps_vs_Mach_alpha_deg_16, color='green',  linestyle='solid', label="AoA + dele 16 [deg] - Extracted")
    plt.plot(Mach_sample, CYp_prps_vs_Mach_alpha_deg_24, color='red',    linestyle='solid', label="AoA + dele 24 [deg] - Extracted")   
    plt.plot(Mach_bps, CYp_table_prps_v_alpha_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA + dele -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, CYp_table_prps_v_alpha_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA + dele -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, CYp_table_prps_v_alpha_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA + dele -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, CYp_table_prps_v_alpha_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA + dele 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, CYp_table_prps_v_alpha_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA + dele 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, CYp_table_prps_v_alpha_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA + dele 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, CYp_table_prps_v_alpha_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA + dele 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("CYp_prps")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 12.    Coefficient: CYr [1/(rad/s)] = f(alpha_deg, Mach)
#       Method of determination: Computational
#       Details: Change in sideforce due to change in yaw rate - how much side force is 
#                generated when the aircraft yaws? Values are negative, therefore negative 
#                sideforce is generated when the aircraft yaws right. This is desirable
#                for stability. The aircraft becomes more stable at higher angles of attack
#                due to the presentation of the ventral rudder into the flow.
#       Source: Walker 60, page 100
#------------------------------------------------------------------------------------------

# CYr table from comma separated variables exported from plot digitizer
CYr_prps_vs_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/CYr_prps_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample = CYr_prps_vs_Mach['x'].values
CYr_prps_vs_Mach_alpha_deg_0  = CYr_prps_vs_Mach['alpha_deg_0'].values
CYr_prps_vs_Mach_alpha_deg_8  = CYr_prps_vs_Mach['alpha_deg_8'].values
CYr_prps_vs_Mach_alpha_deg_16 = CYr_prps_vs_Mach['alpha_deg_16'].values
CYr_prps_vs_Mach_alpha_deg_24 = CYr_prps_vs_Mach['alpha_deg_24'].values

# We will populate CYr vectors for each AoA over the Mach values
CYr_prps_v_Mach_alpha_deg_0  = np.zeros(nMach_bps)
CYr_prps_v_Mach_alpha_deg_8  = np.zeros(nMach_bps)
CYr_prps_v_Mach_alpha_deg_16 = np.zeros(nMach_bps)
CYr_prps_v_Mach_alpha_deg_24 = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    CYr_prps_v_Mach_alpha_deg_0[ii]  = fastInterp1(Mach_sample, CYr_prps_vs_Mach_alpha_deg_0,  Mach_bps[ii])
    CYr_prps_v_Mach_alpha_deg_8[ii]  = fastInterp1(Mach_sample, CYr_prps_vs_Mach_alpha_deg_8,  Mach_bps[ii])
    CYr_prps_v_Mach_alpha_deg_16[ii] = fastInterp1(Mach_sample, CYr_prps_vs_Mach_alpha_deg_16, Mach_bps[ii])
    CYr_prps_v_Mach_alpha_deg_24[ii] = fastInterp1(Mach_sample, CYr_prps_vs_Mach_alpha_deg_24, Mach_bps[ii])
    
# Reflect CYr to negative AoA
CYr_prps_v_Mach_alpha_deg_m8  = CYr_prps_v_Mach_alpha_deg_8
CYr_prps_v_Mach_alpha_deg_m16 = CYr_prps_v_Mach_alpha_deg_16
CYr_prps_v_Mach_alpha_deg_m24 = CYr_prps_v_Mach_alpha_deg_24

# Combine data into a single 2D array
CYr_prps_v_Mach_alpha_deg = np.column_stack((CYr_prps_v_Mach_alpha_deg_m24, \
                                                    CYr_prps_v_Mach_alpha_deg_m16, \
                                                    CYr_prps_v_Mach_alpha_deg_m8, \
                                                    CYr_prps_v_Mach_alpha_deg_0, \
                                                    CYr_prps_v_Mach_alpha_deg_8, \
                                                    CYr_prps_v_Mach_alpha_deg_16, \
                                                    CYr_prps_v_Mach_alpha_deg_24))

# Interpolate table CYr_prps_v_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
CYr_table_prps_v_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        CYr_table_prps_v_alpha_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                    CYr_prps_v_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if CYr_prps_vs_Mach_plot == 'on':
    plt.figure('CYr [1/(rad/s)] - Stability Derivative')
    plt.plot(Mach_sample, CYr_prps_vs_Mach_alpha_deg_0,  color='black',  linestyle='solid', label="AoA + dele 0 [deg] - Extracted")
    plt.plot(Mach_sample, CYr_prps_vs_Mach_alpha_deg_8,  color='blue',   linestyle='solid', label="AoA + dele 8 [deg] - Extracted")
    plt.plot(Mach_sample, CYr_prps_vs_Mach_alpha_deg_16, color='green',  linestyle='solid', label="AoA + dele 16 [deg] - Extracted")
    plt.plot(Mach_sample, CYr_prps_vs_Mach_alpha_deg_24, color='red',    linestyle='solid', label="AoA + dele 24 [deg] - Extracted")   
    plt.plot(Mach_bps, CYr_table_prps_v_alpha_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA + dele -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, CYr_table_prps_v_alpha_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA + dele -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, CYr_table_prps_v_alpha_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA + dele -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, CYr_table_prps_v_alpha_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA + dele 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, CYr_table_prps_v_alpha_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA + dele 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, CYr_table_prps_v_alpha_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA + dele 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, CYr_table_prps_v_alpha_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA + dele 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("CYr_prps")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 13.   Coefficient: CYdela [1/deg] = f(alpha_deg, Mach)
#       Method of determination: Computational
#       Details: Change in sideforce due to change in rudder angle
#       Source: Walker 60, report page 113
#------------------------------------------------------------------------------------------

# CYdela table from comma separated variables exported from plot digitizer
CYdela_pdeg_vs_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/CYdela_pdeg_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample = CYdela_pdeg_vs_Mach['x'].values
CYdela_pdeg_vs_Mach_alpha_deg_0  = CYdela_pdeg_vs_Mach['alpha_deg_0'].values
CYdela_pdeg_vs_Mach_alpha_deg_8  = CYdela_pdeg_vs_Mach['alpha_deg_8'].values
CYdela_pdeg_vs_Mach_alpha_deg_16 = CYdela_pdeg_vs_Mach['alpha_deg_16'].values
CYdela_pdeg_vs_Mach_alpha_deg_24 = CYdela_pdeg_vs_Mach['alpha_deg_24'].values

# We will populate CYdela vectors for each AoA over the Mach values
CYdela_pdeg_v_Mach_alpha_deg_0  = np.zeros(nMach_bps)
CYdela_pdeg_v_Mach_alpha_deg_8  = np.zeros(nMach_bps)
CYdela_pdeg_v_Mach_alpha_deg_16 = np.zeros(nMach_bps)
CYdela_pdeg_v_Mach_alpha_deg_24 = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    CYdela_pdeg_v_Mach_alpha_deg_0[ii]  = fastInterp1(Mach_sample, CYdela_pdeg_vs_Mach_alpha_deg_0,  Mach_bps[ii])
    CYdela_pdeg_v_Mach_alpha_deg_8[ii]  = fastInterp1(Mach_sample, CYdela_pdeg_vs_Mach_alpha_deg_8,  Mach_bps[ii])
    CYdela_pdeg_v_Mach_alpha_deg_16[ii] = fastInterp1(Mach_sample, CYdela_pdeg_vs_Mach_alpha_deg_16, Mach_bps[ii])
    CYdela_pdeg_v_Mach_alpha_deg_24[ii] = fastInterp1(Mach_sample, CYdela_pdeg_vs_Mach_alpha_deg_24, Mach_bps[ii])
    
# Reflect CYdela to negative AoA
CYdela_pdeg_v_Mach_alpha_deg_m8  = CYdela_pdeg_v_Mach_alpha_deg_8
CYdela_pdeg_v_Mach_alpha_deg_m16 = CYdela_pdeg_v_Mach_alpha_deg_16
CYdela_pdeg_v_Mach_alpha_deg_m24 = CYdela_pdeg_v_Mach_alpha_deg_24

# Combine data into a single 2D array
CYdela_pdeg_v_Mach_alpha_deg = np.column_stack((CYdela_pdeg_v_Mach_alpha_deg_m24, \
                                                    CYdela_pdeg_v_Mach_alpha_deg_m16, \
                                                    CYdela_pdeg_v_Mach_alpha_deg_m8, \
                                                    CYdela_pdeg_v_Mach_alpha_deg_0, \
                                                    CYdela_pdeg_v_Mach_alpha_deg_8, \
                                                    CYdela_pdeg_v_Mach_alpha_deg_16, \
                                                    CYdela_pdeg_v_Mach_alpha_deg_24))

# Interpolate table CYdela_pdeg_v_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
CYdela_table_pdeg_v_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        CYdela_table_pdeg_v_alpha_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                    CYdela_pdeg_v_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if CYdela_pdeg_vs_Mach_plot == 'on':
    plt.figure('CYdela [1/deg] - Control Derivative')
    plt.plot(Mach_sample, CYdela_pdeg_vs_Mach_alpha_deg_0,  color='black',  linestyle='solid', label="AoA + dele 0 [deg] - Extracted")
    plt.plot(Mach_sample, CYdela_pdeg_vs_Mach_alpha_deg_8,  color='blue',   linestyle='solid', label="AoA + dele 8 [deg] - Extracted")
    plt.plot(Mach_sample, CYdela_pdeg_vs_Mach_alpha_deg_16, color='green',  linestyle='solid', label="AoA + dele 16 [deg] - Extracted")
    plt.plot(Mach_sample, CYdela_pdeg_vs_Mach_alpha_deg_24, color='red',    linestyle='solid', label="AoA + dele 24 [deg] - Extracted")   
    plt.plot(Mach_bps, CYdela_table_pdeg_v_alpha_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA + dele -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, CYdela_table_pdeg_v_alpha_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA + dele -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, CYdela_table_pdeg_v_alpha_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA + dele -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, CYdela_table_pdeg_v_alpha_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA + dele 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, CYdela_table_pdeg_v_alpha_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA + dele 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, CYdela_table_pdeg_v_alpha_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA + dele 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, CYdela_table_pdeg_v_alpha_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA + dele 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("CYdela_pdeg")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 14.   Coefficient: Clp_pdeg = f(alpha_bps_deg, Mach_bps)
#       Method of determination: Computational
#       Details: Change in roll moment due to change in roll rate. Negative 
#                valued indicating that a positive roll rate (rotating right
#                wing down) produces a negative roll moment, tending to rotate
#                the airframe right wing up. Here, the derivative is negative
#                valued, indicating stability. This will help dampen roll 
#                oscillations.
#       Source: Walker 60
#-------------------------------------------------------------------------------

# Clp Table from comma separated variables exported from plot digitizer
Clp_for_AoA_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/Clp_v_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample               = Clp_for_AoA_Mach['x'].values
Clp_vs_Mach_deg_AoA0_deg  = Clp_for_AoA_Mach['alpha_deg_0'].values
Clp_vs_Mach_deg_AoA8_deg  = Clp_for_AoA_Mach['alpha_deg_8'].values
Clp_vs_Mach_deg_AoA16_deg = Clp_for_AoA_Mach['alpha_deg_16'].values
Clp_vs_Mach_deg_AoA24_deg = Clp_for_AoA_Mach['alpha_deg_24'].values

# Populate Clp vectors for each AoA over the Mach values
Clp_v_Mach_AoA0_deg  = np.zeros(nMach_bps)
Clp_v_Mach_AoA8_deg  = np.zeros(nMach_bps)
Clp_v_Mach_AoA16_deg = np.zeros(nMach_bps)
Clp_v_Mach_AoA24_deg = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Clp_v_Mach_AoA0_deg[ii]  = fastInterp1(Mach_sample, Clp_vs_Mach_deg_AoA0_deg,  Mach_bps[ii])
    Clp_v_Mach_AoA8_deg[ii]  = fastInterp1(Mach_sample, Clp_vs_Mach_deg_AoA8_deg,  Mach_bps[ii])
    Clp_v_Mach_AoA16_deg[ii] = fastInterp1(Mach_sample, Clp_vs_Mach_deg_AoA16_deg, Mach_bps[ii])
    Clp_v_Mach_AoA24_deg[ii] = fastInterp1(Mach_sample, Clp_vs_Mach_deg_AoA24_deg, Mach_bps[ii])

# Reflect Clp to negative AoA
Clp_v_Mach_AoAm8_deg  = Clp_v_Mach_AoA8_deg
Clp_v_Mach_AoAm16_deg = Clp_v_Mach_AoA16_deg
Clp_v_Mach_AoAm24_deg = Clp_v_Mach_AoA24_deg
    
# Combine data into a single 2D array
Clp_prps_vs_Mach_alpha_deg = np.column_stack((Clp_v_Mach_AoAm24_deg, Clp_v_Mach_AoAm16_deg, Clp_v_Mach_AoAm8_deg, \
                                                    Clp_v_Mach_AoA0_deg, Clp_v_Mach_AoA8_deg, Clp_v_Mach_AoA16_deg,  \
                                                    Clp_v_Mach_AoA24_deg))

# Interpolate table Clp_prps_vs_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Clp_prps_table_Mach_alpha_deg = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Clp_prps_table_Mach_alpha_deg[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                            Clp_prps_vs_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Clp_vs_Mach_alpha_plot == 'on':
    plt.figure('Clp [1/(rad/s)] - Stability Derivative')
    plt.plot(Mach_sample, Clp_vs_Mach_deg_AoA0_deg,  color='black',   linestyle='solid', label="AoA 0 [deg] - Extracted")
    plt.plot(Mach_sample, Clp_vs_Mach_deg_AoA8_deg,  color='blue',    linestyle='solid', label="AoA 8 [deg] - Extracted")
    plt.plot(Mach_sample, Clp_vs_Mach_deg_AoA16_deg, color='green',   linestyle='solid', label="AoA 16 [deg] - Extracted")
    plt.plot(Mach_sample, Clp_vs_Mach_deg_AoA24_deg, color='red',     linestyle='solid', label="AoA 24 [deg] - Extracted")
    plt.plot(Mach_bps, Clp_prps_table_Mach_alpha_deg[1,:],  color='grey', linestyle='dashed', label="AoA -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Clp_prps_table_Mach_alpha_deg[9,:],  color='grey', linestyle='dashed', label="AoA -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Clp_prps_table_Mach_alpha_deg[17,:], color='grey', linestyle='dashed', label="AoA -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Clp_prps_table_Mach_alpha_deg[25,:], color='grey', linestyle='dashed', label="AoA 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Clp_prps_table_Mach_alpha_deg[33,:], color='grey', linestyle='dashed', label="AoA 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Clp_prps_table_Mach_alpha_deg[41,:], color='grey', linestyle='dashed', label="AoA 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Clp_prps_table_Mach_alpha_deg[49,:], color='grey', linestyle='dashed', label="AoA 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("Clp_prps")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 15.   Coefficient: Clr_prps = f(alpha_bps_deg, Mach_bps)
#       Method of determination: Computational
#       Details: Change in roll moment due to change in yaw rate. The derivative
#                is negative so that if the aircraft yaws right it will roll left
#                (left wing down). 
#       Source: Walker 60, page 100
#-------------------------------------------------------------------------------

# Clr Table from comma separated variables exported from plot digitizer
Clr_for_AoA_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/Clr_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample               = Clr_for_AoA_Mach['x'].values
Clr_vs_Mach_deg_AoA0_deg  = Clr_for_AoA_Mach['alpha_deg_0'].values
Clr_vs_Mach_deg_AoA8_deg  = Clr_for_AoA_Mach['alpha_deg_8'].values
Clr_vs_Mach_deg_AoA16_deg = Clr_for_AoA_Mach['alpha_deg_16'].values
Clr_vs_Mach_deg_AoA24_deg = Clr_for_AoA_Mach['alpha_deg_24'].values

# Populate Clr vectors for each AoA over the Mach values
Clr_v_Mach_AoA0_deg  = np.zeros(nMach_bps)
Clr_v_Mach_AoA8_deg  = np.zeros(nMach_bps)
Clr_v_Mach_AoA16_deg = np.zeros(nMach_bps)
Clr_v_Mach_AoA24_deg = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Clr_v_Mach_AoA0_deg[ii]  = fastInterp1(Mach_sample, Clr_vs_Mach_deg_AoA0_deg,  Mach_bps[ii])
    Clr_v_Mach_AoA8_deg[ii]  = fastInterp1(Mach_sample, Clr_vs_Mach_deg_AoA8_deg,  Mach_bps[ii])
    Clr_v_Mach_AoA16_deg[ii] = fastInterp1(Mach_sample, Clr_vs_Mach_deg_AoA16_deg, Mach_bps[ii])
    Clr_v_Mach_AoA24_deg[ii] = fastInterp1(Mach_sample, Clr_vs_Mach_deg_AoA24_deg, Mach_bps[ii])

# Reflect Clr to negative AoA
Clr_v_Mach_AoAm8_deg  = Clr_v_Mach_AoA8_deg
Clr_v_Mach_AoAm16_deg = Clr_v_Mach_AoA16_deg
Clr_v_Mach_AoAm24_deg = Clr_v_Mach_AoA24_deg
    
# Combine data into a single 2D array
Clr_prps_vs_Mach_alpha_deg = np.column_stack((Clr_v_Mach_AoAm24_deg, Clr_v_Mach_AoAm16_deg, Clr_v_Mach_AoAm8_deg, \
                                                    Clr_v_Mach_AoA0_deg, Clr_v_Mach_AoA8_deg, Clr_v_Mach_AoA16_deg,  \
                                                    Clr_v_Mach_AoA24_deg))

# Interpolate table Clr_prps_vs_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Clr_prps_table_Mach_alpha_deg = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Clr_prps_table_Mach_alpha_deg[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                            Clr_prps_vs_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Clr_vs_Mach_alpha_plot == 'on':
    plt.figure('Clr [1/(rad/s)] - Stability Derivative')
    plt.plot(Mach_sample, Clr_vs_Mach_deg_AoA0_deg,  color='black',   linestyle='solid', label="AoA 0 [deg] - Extracted")
    plt.plot(Mach_sample, Clr_vs_Mach_deg_AoA8_deg,  color='blue',    linestyle='solid', label="AoA 8 [deg] - Extracted")
    plt.plot(Mach_sample, Clr_vs_Mach_deg_AoA16_deg, color='green',   linestyle='solid', label="AoA 16 [deg] - Extracted")
    plt.plot(Mach_sample, Clr_vs_Mach_deg_AoA24_deg, color='red',     linestyle='solid', label="AoA 24 [deg] - Extracted")
    plt.plot(Mach_bps, Clr_prps_table_Mach_alpha_deg[1,:],  color='grey', linestyle='dashed', label="AoA -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Clr_prps_table_Mach_alpha_deg[9,:],  color='grey', linestyle='dashed', label="AoA -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Clr_prps_table_Mach_alpha_deg[17,:], color='grey', linestyle='dashed', label="AoA -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Clr_prps_table_Mach_alpha_deg[25,:], color='grey', linestyle='dashed', label="AoA 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Clr_prps_table_Mach_alpha_deg[33,:], color='grey', linestyle='dashed', label="AoA 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Clr_prps_table_Mach_alpha_deg[41,:], color='grey', linestyle='dashed', label="AoA 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Clr_prps_table_Mach_alpha_deg[49,:], color='grey', linestyle='dashed', label="AoA 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("Clr_prps")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 16.   Coefficient: Cldela [1/deg] = f(alpha_deg, Mach)
#       Method of determination: Computational
#       Details: Change in roll moment due to aileron
#       Source: Walker 60, report page 112
#------------------------------------------------------------------------------------------

# Cldela table from comma separated variables exported from plot digitizer
Cldela_pdeg_vs_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/Cldela_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample = Cldela_pdeg_vs_Mach['x'].values
Cldela_pdeg_vs_Mach_alpha_deg_0  = Cldela_pdeg_vs_Mach['alpha_deg_0'].values
Cldela_pdeg_vs_Mach_alpha_deg_8  = Cldela_pdeg_vs_Mach['alpha_deg_8'].values
Cldela_pdeg_vs_Mach_alpha_deg_16 = Cldela_pdeg_vs_Mach['alpha_deg_16'].values
Cldela_pdeg_vs_Mach_alpha_deg_24 = Cldela_pdeg_vs_Mach['alpha_deg_24'].values

# We will populate Cldela vectors for each AoA over the Mach values
Cldela_pdeg_v_Mach_alpha_deg_0  = np.zeros(nMach_bps)
Cldela_pdeg_v_Mach_alpha_deg_8  = np.zeros(nMach_bps)
Cldela_pdeg_v_Mach_alpha_deg_16 = np.zeros(nMach_bps)
Cldela_pdeg_v_Mach_alpha_deg_24 = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Cldela_pdeg_v_Mach_alpha_deg_0[ii]  = fastInterp1(Mach_sample, Cldela_pdeg_vs_Mach_alpha_deg_0,  Mach_bps[ii])
    Cldela_pdeg_v_Mach_alpha_deg_8[ii]  = fastInterp1(Mach_sample, Cldela_pdeg_vs_Mach_alpha_deg_8,  Mach_bps[ii])
    Cldela_pdeg_v_Mach_alpha_deg_16[ii] = fastInterp1(Mach_sample, Cldela_pdeg_vs_Mach_alpha_deg_16, Mach_bps[ii])
    Cldela_pdeg_v_Mach_alpha_deg_24[ii] = fastInterp1(Mach_sample, Cldela_pdeg_vs_Mach_alpha_deg_24, Mach_bps[ii])
    
# Reflect Cldela to negative AoA
Cldela_pdeg_v_Mach_alpha_deg_m8  = Cldela_pdeg_v_Mach_alpha_deg_8
Cldela_pdeg_v_Mach_alpha_deg_m16 = Cldela_pdeg_v_Mach_alpha_deg_16
Cldela_pdeg_v_Mach_alpha_deg_m24 = Cldela_pdeg_v_Mach_alpha_deg_24

# Combine data into a single 2D array
Cldela_pdeg_v_Mach_alpha_deg = np.column_stack((Cldela_pdeg_v_Mach_alpha_deg_m24, \
                                                    Cldela_pdeg_v_Mach_alpha_deg_m16, \
                                                    Cldela_pdeg_v_Mach_alpha_deg_m8, \
                                                    Cldela_pdeg_v_Mach_alpha_deg_0, \
                                                    Cldela_pdeg_v_Mach_alpha_deg_8, \
                                                    Cldela_pdeg_v_Mach_alpha_deg_16, \
                                                    Cldela_pdeg_v_Mach_alpha_deg_24))

# Interpolate table Cldela_pdeg_v_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Cldela_table_pdeg_v_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Cldela_table_pdeg_v_alpha_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                    Cldela_pdeg_v_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Cldela_pdeg_vs_Mach_plot == 'on':
    plt.figure('Cldela [1/deg] - Control Derivative')
    plt.plot(Mach_sample, Cldela_pdeg_vs_Mach_alpha_deg_0,  color='black',  linestyle='solid', label="AoA + dele 0 [deg] - Extracted")
    plt.plot(Mach_sample, Cldela_pdeg_vs_Mach_alpha_deg_8,  color='blue',   linestyle='solid', label="AoA + dele 8 [deg] - Extracted")
    plt.plot(Mach_sample, Cldela_pdeg_vs_Mach_alpha_deg_16, color='green',  linestyle='solid', label="AoA + dele 16 [deg] - Extracted")
    plt.plot(Mach_sample, Cldela_pdeg_vs_Mach_alpha_deg_24, color='red',    linestyle='solid', label="AoA + dele 24 [deg] - Extracted")   
    plt.plot(Mach_bps, Cldela_table_pdeg_v_alpha_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA + dele -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cldela_table_pdeg_v_alpha_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA + dele -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cldela_table_pdeg_v_alpha_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA + dele -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Cldela_table_pdeg_v_alpha_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA + dele 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Cldela_table_pdeg_v_alpha_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA + dele 8 [deg] - Interpolated",  marker='.')
    plt.plot(Mach_bps, Cldela_table_pdeg_v_alpha_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA + dele 16 [deg] - Interpolated", marker='^', markerfacecolor='none')
    plt.plot(Mach_bps, Cldela_table_pdeg_v_alpha_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA + dele 24 [deg] - Interpolated", marker='*')
    plt.xlabel("Mach")
    plt.ylabel("Cldela_pdeg")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 17.   Coefficient: Cldelr [1/deg] = f(alpha_deg, Mach)
#       Method of determination: Computational
#       Details: Change in roll moment due to change in rudder angle
#       Source: Walker 60, report page 110
#------------------------------------------------------------------------------------------

# Cldelr table from comma separated variables exported from plot digitizer
Cldelr_pdeg_vs_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/Cldelr_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample = Cldelr_pdeg_vs_Mach['x'].values
Cldelr_pdeg_vs_Mach_alpha_deg_0  = Cldelr_pdeg_vs_Mach['alpha_deg_0'].values
Cldelr_pdeg_vs_Mach_alpha_deg_8  = Cldelr_pdeg_vs_Mach['alpha_deg_8'].values
Cldelr_pdeg_vs_Mach_alpha_deg_16 = Cldelr_pdeg_vs_Mach['alpha_deg_16'].values
Cldelr_pdeg_vs_Mach_alpha_deg_24 = Cldelr_pdeg_vs_Mach['alpha_deg_24'].values

# We will populate Cldelr vectors for each AoA over the Mach values
Cldelr_pdeg_v_Mach_alpha_deg_0  = np.zeros(nMach_bps)
Cldelr_pdeg_v_Mach_alpha_deg_8  = np.zeros(nMach_bps)
Cldelr_pdeg_v_Mach_alpha_deg_16 = np.zeros(nMach_bps)
Cldelr_pdeg_v_Mach_alpha_deg_24 = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Cldelr_pdeg_v_Mach_alpha_deg_0[ii]  = fastInterp1(Mach_sample, Cldelr_pdeg_vs_Mach_alpha_deg_0,  Mach_bps[ii])
    Cldelr_pdeg_v_Mach_alpha_deg_8[ii]  = fastInterp1(Mach_sample, Cldelr_pdeg_vs_Mach_alpha_deg_8,  Mach_bps[ii])
    Cldelr_pdeg_v_Mach_alpha_deg_16[ii] = fastInterp1(Mach_sample, Cldelr_pdeg_vs_Mach_alpha_deg_16, Mach_bps[ii])
    Cldelr_pdeg_v_Mach_alpha_deg_24[ii] = fastInterp1(Mach_sample, Cldelr_pdeg_vs_Mach_alpha_deg_24, Mach_bps[ii])
    
# Reflect Cldelr to negative AoA
Cldelr_pdeg_v_Mach_alpha_deg_m8  = Cldelr_pdeg_v_Mach_alpha_deg_8
Cldelr_pdeg_v_Mach_alpha_deg_m16 = Cldelr_pdeg_v_Mach_alpha_deg_16
Cldelr_pdeg_v_Mach_alpha_deg_m24 = Cldelr_pdeg_v_Mach_alpha_deg_24

# Combine data into a single 2D array
Cldelr_pdeg_v_Mach_alpha_deg = np.column_stack((Cldelr_pdeg_v_Mach_alpha_deg_m24, \
                                                    Cldelr_pdeg_v_Mach_alpha_deg_m16, \
                                                    Cldelr_pdeg_v_Mach_alpha_deg_m8, \
                                                    Cldelr_pdeg_v_Mach_alpha_deg_0, \
                                                    Cldelr_pdeg_v_Mach_alpha_deg_8, \
                                                    Cldelr_pdeg_v_Mach_alpha_deg_16, \
                                                    Cldelr_pdeg_v_Mach_alpha_deg_24))

# Interpolate table Cldelr_pdeg_v_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Cldelr_table_pdeg_v_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Cldelr_table_pdeg_v_alpha_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                    Cldelr_pdeg_v_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Cldelr_pdeg_vs_Mach_plot == 'on':
    plt.figure('Cldelr [1/deg] - Control Derivative')
    plt.plot(Mach_sample, Cldelr_pdeg_vs_Mach_alpha_deg_0,  color='black',  linestyle='solid', label="AoA + dele 0 [deg] - Extracted")
    plt.plot(Mach_sample, Cldelr_pdeg_vs_Mach_alpha_deg_8,  color='blue',   linestyle='solid', label="AoA + dele 8 [deg] - Extracted")
    plt.plot(Mach_sample, Cldelr_pdeg_vs_Mach_alpha_deg_16, color='green',  linestyle='solid', label="AoA + dele 16 [deg] - Extracted")
    plt.plot(Mach_sample, Cldelr_pdeg_vs_Mach_alpha_deg_24, color='red',    linestyle='solid', label="AoA + dele 24 [deg] - Extracted")   
    plt.plot(Mach_bps, Cldelr_table_pdeg_v_alpha_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA + dele -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cldelr_table_pdeg_v_alpha_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA + dele -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cldelr_table_pdeg_v_alpha_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA + dele -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Cldelr_table_pdeg_v_alpha_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA + dele 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Cldelr_table_pdeg_v_alpha_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA + dele 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cldelr_table_pdeg_v_alpha_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA + dele 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cldelr_table_pdeg_v_alpha_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA + dele 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("Cldelr_pdeg")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 18.   Coefficient: Pitch moment, Cm = f(alpha_bps_deg, Mach_bps)
#       Method of determination: Computational
#       Details: Total body, no surface deflections
#       Source: Walker 1960, page 72. 
#-------------------------------------------------------------------------------

# Cm Table from comma separated variables exported from plot digitizer
Cm_for_AoA_Mach = pd.read_csv('models/X15/aerodynamic_model/Walker60/data/Cm_vs_alpha_Walker60.csv', header = 0)

# Get data sequences from the pandas DataFrame
alpha_sample_deg   = Cm_for_AoA_Mach['x'].values
Cm_vs_alpha_deg_M2 = Cm_for_AoA_Mach['Mach_2'].values
Cm_vs_alpha_deg_M3 = Cm_for_AoA_Mach['Mach_3'].values
Cm_vs_alpha_deg_M4 = Cm_for_AoA_Mach['Mach_4'].values
Cm_vs_alpha_deg_M6 = Cm_for_AoA_Mach['Mach_6'].values
Cm_vs_alpha_deg_M8 = Cm_for_AoA_Mach['Mach_8'].values

# Reflect Cm data about x-axis (Cm = 0)
alpha_sample_deg   = -alpha_sample_deg[::-1] + alpha_sample_deg
Cm_vs_alpha_deg_M2 = -Cm_vs_alpha_deg_M2[::-1] + Cm_vs_alpha_deg_M2
Cm_vs_alpha_deg_M3 = -Cm_vs_alpha_deg_M3[::-1] + Cm_vs_alpha_deg_M3
Cm_vs_alpha_deg_M4 = -Cm_vs_alpha_deg_M4[::-1] + Cm_vs_alpha_deg_M4
Cm_vs_alpha_deg_M6 = -Cm_vs_alpha_deg_M6[::-1] + Cm_vs_alpha_deg_M6
Cm_vs_alpha_deg_M8 = -Cm_vs_alpha_deg_M8[::-1] + Cm_vs_alpha_deg_M8

# We will populate Cm vectors for each Mach over the alpha_deg values
Cm_v_alpha_deg_M2 = np.zeros(nalpha_bps_deg)
Cm_v_alpha_deg_M3 = np.zeros(nalpha_bps_deg)
Cm_v_alpha_deg_M4 = np.zeros(nalpha_bps_deg)
Cm_v_alpha_deg_M6 = np.zeros(nalpha_bps_deg)
Cm_v_alpha_deg_M8 = np.zeros(nalpha_bps_deg)

# Interpolate to a regularly spaced database
for ii in range(0, nalpha_bps_deg, 1):
    Cm_v_alpha_deg_M2[ii] = fastInterp1(alpha_sample_deg, Cm_vs_alpha_deg_M2, alpha_bps_deg[ii])
    Cm_v_alpha_deg_M3[ii] = fastInterp1(alpha_sample_deg, Cm_vs_alpha_deg_M3, alpha_bps_deg[ii])
    Cm_v_alpha_deg_M4[ii] = fastInterp1(alpha_sample_deg, Cm_vs_alpha_deg_M4, alpha_bps_deg[ii])
    Cm_v_alpha_deg_M6[ii] = fastInterp1(alpha_sample_deg, Cm_vs_alpha_deg_M6, alpha_bps_deg[ii])
    Cm_v_alpha_deg_M8[ii] = fastInterp1(alpha_sample_deg, Cm_vs_alpha_deg_M8, alpha_bps_deg[ii])
    
# Combine data into a single 2D array
Cm_vs_alpha_deg_Mach = np.column_stack((Cm_v_alpha_deg_M2, Cm_v_alpha_deg_M3, \
    Cm_v_alpha_deg_M4, Cm_v_alpha_deg_M6, Cm_v_alpha_deg_M8))

# Interpolate to all specified Mach breakpoints
Cm_table_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
Mach_presently = np.array([2, 3, 4, 6, 8], dtype=float)

for jj in range(nMach_bps):
    for ii in range(nalpha_bps_deg):
        Cm_table_alpha_deg_Mach[ii, jj] = fastInterp2(alpha_bps_deg, Mach_presently, Cm_vs_alpha_deg_Mach, alpha_bps_deg[ii], Mach_bps[jj])

# Compare the results on a single plot
if Cm_vs_alpha_plot == 'on':
    plt.figure('Cm - Moment Coefficient for Wings and Body')
    plt.plot(alpha_sample_deg, Cm_vs_alpha_deg_M2, color='black',  linestyle='solid', label="Mach 2 - Extracted")
    plt.plot(alpha_sample_deg, Cm_vs_alpha_deg_M3, color='blue',   linestyle='solid', label="Mach 3 - Extracted")
    plt.plot(alpha_sample_deg, Cm_vs_alpha_deg_M4, color='green',  linestyle='solid', label="Mach 4 - Extracted")
    plt.plot(alpha_sample_deg, Cm_vs_alpha_deg_M6, color='red',    linestyle='solid', label="Mach 6 - Extracted")
    plt.plot(alpha_sample_deg, Cm_vs_alpha_deg_M8, color='orange', linestyle='solid', label="Mach 8 - Extracted")
    plt.plot(alpha_bps_deg, Cm_table_alpha_deg_Mach[:,0], color='grey', linestyle='dashed', label="Mach 2 - Interpolated", marker='o', markerfacecolor='none')
    plt.plot(alpha_bps_deg, Cm_table_alpha_deg_Mach[:,2], color='grey', linestyle='dashed', label="Mach 3 - Interpolated", marker='x')
    plt.plot(alpha_bps_deg, Cm_table_alpha_deg_Mach[:,4], color='grey', linestyle='dashed', label="Mach 4 - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(alpha_bps_deg, Cm_table_alpha_deg_Mach[:,6], color='grey', linestyle='dashed', label="Mach 5 - Interpolated", marker='+')
    plt.plot(alpha_bps_deg, Cm_table_alpha_deg_Mach[:,8], color='grey', linestyle='dashed', label="Mach 6 - Interpolated", marker='D', markerfacecolor='none')
    plt.plot(alpha_bps_deg, Cm_table_alpha_deg_Mach[:,10], color='grey', linestyle='dashed', label="Mach 7 - Interpolated", marker='*')
    plt.plot(alpha_bps_deg, Cm_table_alpha_deg_Mach[:,12], color='grey', linestyle='dashed', label="Mach 8 - Interpolated", marker='h', markerfacecolor='none')
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Cm")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 19.   Coefficient: Cnp_prps = f(alpha_bps_deg, Mach_bps)
#       Method of determination: Computational
#       Details: Change in yaw moment due to change in roll rate. This is roll
#                damping in yaw. A rolling moment can influence yaw rate and vice
#                versa. Rolling right wing down means yawing to the left.
#       Source: Walker 60
#-------------------------------------------------------------------------------

# Cnp Table from comma separated variables exported from plot digitizer
Cnp_for_AoA_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/Cnp_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample               = Cnp_for_AoA_Mach['x'].values
Cnp_vs_Mach_deg_AoA0_deg  = Cnp_for_AoA_Mach['alpha_deg_0'].values
Cnp_vs_Mach_deg_AoA8_deg  = Cnp_for_AoA_Mach['alpha_deg_8'].values
Cnp_vs_Mach_deg_AoA16_deg = Cnp_for_AoA_Mach['alpha_deg_16'].values
Cnp_vs_Mach_deg_AoA24_deg = Cnp_for_AoA_Mach['alpha_deg_24'].values

# Populate Cnp vectors for each AoA over the Mach values
Cnp_v_Mach_AoA0_deg  = np.zeros(nMach_bps)
Cnp_v_Mach_AoA8_deg  = np.zeros(nMach_bps)
Cnp_v_Mach_AoA16_deg = np.zeros(nMach_bps)
Cnp_v_Mach_AoA24_deg = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Cnp_v_Mach_AoA0_deg[ii]  = fastInterp1(Mach_sample, Cnp_vs_Mach_deg_AoA0_deg,  Mach_bps[ii])
    Cnp_v_Mach_AoA8_deg[ii]  = fastInterp1(Mach_sample, Cnp_vs_Mach_deg_AoA8_deg,  Mach_bps[ii])
    Cnp_v_Mach_AoA16_deg[ii] = fastInterp1(Mach_sample, Cnp_vs_Mach_deg_AoA16_deg, Mach_bps[ii])
    Cnp_v_Mach_AoA24_deg[ii] = fastInterp1(Mach_sample, Cnp_vs_Mach_deg_AoA24_deg, Mach_bps[ii])

# Reflect Cnp to negative AoA
Cnp_v_Mach_AoAm8_deg  = Cnp_v_Mach_AoA8_deg
Cnp_v_Mach_AoAm16_deg = Cnp_v_Mach_AoA16_deg
Cnp_v_Mach_AoAm24_deg = Cnp_v_Mach_AoA24_deg
    
# Combine data into a single 2D array
Cnp_prps_vs_Mach_alpha_deg = np.column_stack((Cnp_v_Mach_AoAm24_deg, Cnp_v_Mach_AoAm16_deg, Cnp_v_Mach_AoAm8_deg, \
                                                    Cnp_v_Mach_AoA0_deg, Cnp_v_Mach_AoA8_deg, Cnp_v_Mach_AoA16_deg,  \
                                                    Cnp_v_Mach_AoA24_deg))

# Interpolate table Cnp_prps_vs_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Cnp_prps_table_Mach_alpha_deg = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Cnp_prps_table_Mach_alpha_deg[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                            Cnp_prps_vs_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Cnp_vs_Mach_alpha_plot == 'on':
    plt.figure('Cnp [1/(rad/s)] - Stability Derivative')
    plt.plot(Mach_sample, Cnp_vs_Mach_deg_AoA0_deg,  color='black',   linestyle='solid', label="AoA 0 [deg] - Extracted")
    plt.plot(Mach_sample, Cnp_vs_Mach_deg_AoA8_deg,  color='blue',    linestyle='solid', label="AoA 8 [deg] - Extracted")
    plt.plot(Mach_sample, Cnp_vs_Mach_deg_AoA16_deg, color='green',   linestyle='solid', label="AoA 16 [deg] - Extracted")
    plt.plot(Mach_sample, Cnp_vs_Mach_deg_AoA24_deg, color='red',     linestyle='solid', label="AoA 24 [deg] - Extracted")
    plt.plot(Mach_bps, Cnp_prps_table_Mach_alpha_deg[1,:],  color='grey', linestyle='dashed', label="AoA -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cnp_prps_table_Mach_alpha_deg[9,:],  color='grey', linestyle='dashed', label="AoA -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cnp_prps_table_Mach_alpha_deg[17,:], color='grey', linestyle='dashed', label="AoA -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Cnp_prps_table_Mach_alpha_deg[25,:], color='grey', linestyle='dashed', label="AoA 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Cnp_prps_table_Mach_alpha_deg[33,:], color='grey', linestyle='dashed', label="AoA 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cnp_prps_table_Mach_alpha_deg[41,:], color='grey', linestyle='dashed', label="AoA 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cnp_prps_table_Mach_alpha_deg[49,:], color='grey', linestyle='dashed', label="AoA 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("Cnp_prps")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 20.   Coefficient: Cnr_prps = f(alpha_bps_deg, Mach_bps)
#       Method of determination: Computational
#       Details: Change in yaw moment due to change in yaw rate. This is
#                damping in yaw. 
#       Source: Walker 60
#-------------------------------------------------------------------------------

# Cnr Table from comma separated variables exported from plot digitizer
Cnr_for_AoA_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/Cnr_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample               = Cnr_for_AoA_Mach['x'].values
Cnr_vs_Mach_deg_AoA0_deg  = Cnr_for_AoA_Mach['alpha_deg_0'].values
Cnr_vs_Mach_deg_AoA8_deg  = Cnr_for_AoA_Mach['alpha_deg_8'].values
Cnr_vs_Mach_deg_AoA16_deg = Cnr_for_AoA_Mach['alpha_deg_16'].values
Cnr_vs_Mach_deg_AoA24_deg = Cnr_for_AoA_Mach['alpha_deg_24'].values

# Populate Cnr vectors for each AoA over the Mach values
Cnr_v_Mach_AoA0_deg  = np.zeros(nMach_bps)
Cnr_v_Mach_AoA8_deg  = np.zeros(nMach_bps)
Cnr_v_Mach_AoA16_deg = np.zeros(nMach_bps)
Cnr_v_Mach_AoA24_deg = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Cnr_v_Mach_AoA0_deg[ii]  = fastInterp1(Mach_sample, Cnr_vs_Mach_deg_AoA0_deg,  Mach_bps[ii])
    Cnr_v_Mach_AoA8_deg[ii]  = fastInterp1(Mach_sample, Cnr_vs_Mach_deg_AoA8_deg,  Mach_bps[ii])
    Cnr_v_Mach_AoA16_deg[ii] = fastInterp1(Mach_sample, Cnr_vs_Mach_deg_AoA16_deg, Mach_bps[ii])
    Cnr_v_Mach_AoA24_deg[ii] = fastInterp1(Mach_sample, Cnr_vs_Mach_deg_AoA24_deg, Mach_bps[ii])

# Reflect Cnr to negative AoA
Cnr_v_Mach_AoAm8_deg  = Cnr_v_Mach_AoA8_deg
Cnr_v_Mach_AoAm16_deg = Cnr_v_Mach_AoA16_deg
Cnr_v_Mach_AoAm24_deg = Cnr_v_Mach_AoA24_deg
    
# Combine data into a single 2D array
Cnr_prps_vs_Mach_alpha_deg = np.column_stack((Cnr_v_Mach_AoAm24_deg, Cnr_v_Mach_AoAm16_deg, Cnr_v_Mach_AoAm8_deg, \
                                                    Cnr_v_Mach_AoA0_deg, Cnr_v_Mach_AoA8_deg, Cnr_v_Mach_AoA16_deg,  \
                                                    Cnr_v_Mach_AoA24_deg))

# Interpolate table Cnr_prps_vs_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Cnr_prps_table_Mach_alpha_deg = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Cnr_prps_table_Mach_alpha_deg[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                            Cnr_prps_vs_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Cnr_vs_Mach_alpha_plot == 'on':
    plt.figure('Cnr [1/(rad/s)] - Stability Derivative')
    plt.plot(Mach_sample, Cnr_vs_Mach_deg_AoA0_deg,  color='black',   linestyle='solid', label="AoA 0 [deg] - Extracted")
    plt.plot(Mach_sample, Cnr_vs_Mach_deg_AoA8_deg,  color='blue',    linestyle='solid', label="AoA 8 [deg] - Extracted")
    plt.plot(Mach_sample, Cnr_vs_Mach_deg_AoA16_deg, color='green',   linestyle='solid', label="AoA 16 [deg] - Extracted")
    plt.plot(Mach_sample, Cnr_vs_Mach_deg_AoA24_deg, color='red',     linestyle='solid', label="AoA 24 [deg] - Extracted")
    plt.plot(Mach_bps, Cnr_prps_table_Mach_alpha_deg[1,:],  color='grey', linestyle='dashed', label="AoA -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cnr_prps_table_Mach_alpha_deg[9,:],  color='grey', linestyle='dashed', label="AoA -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cnr_prps_table_Mach_alpha_deg[17,:], color='grey', linestyle='dashed', label="AoA -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Cnr_prps_table_Mach_alpha_deg[25,:], color='grey', linestyle='dashed', label="AoA 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Cnr_prps_table_Mach_alpha_deg[33,:], color='grey', linestyle='dashed', label="AoA 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cnr_prps_table_Mach_alpha_deg[41,:], color='grey', linestyle='dashed', label="AoA 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cnr_prps_table_Mach_alpha_deg[49,:], color='grey', linestyle='dashed', label="AoA 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("Cnr_prps")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 21.   Coefficient: Cndela [1/deg] = f(alpha_deg, Mach)
#       Method of determination: Computational
#       Details: Change in yaw moment due to change in aileron
#       Source: Walker 60
#------------------------------------------------------------------------------------------

# Cndela table from comma separated variables exported from plot digitizer
Cndela_pdeg_vs_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/Cndela_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample = Cndela_pdeg_vs_Mach['x'].values
Cndela_pdeg_vs_Mach_alpha_deg_0  = Cndela_pdeg_vs_Mach['alpha_deg_0'].values
Cndela_pdeg_vs_Mach_alpha_deg_8  = Cndela_pdeg_vs_Mach['alpha_deg_8'].values
Cndela_pdeg_vs_Mach_alpha_deg_16 = Cndela_pdeg_vs_Mach['alpha_deg_16'].values
Cndela_pdeg_vs_Mach_alpha_deg_24 = Cndela_pdeg_vs_Mach['alpha_deg_24'].values

# We will populate Cndela vectors for each AoA over the Mach values
Cndela_pdeg_v_Mach_alpha_deg_0  = np.zeros(nMach_bps)
Cndela_pdeg_v_Mach_alpha_deg_8  = np.zeros(nMach_bps)
Cndela_pdeg_v_Mach_alpha_deg_16 = np.zeros(nMach_bps)
Cndela_pdeg_v_Mach_alpha_deg_24 = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Cndela_pdeg_v_Mach_alpha_deg_0[ii]  = fastInterp1(Mach_sample, Cndela_pdeg_vs_Mach_alpha_deg_0,  Mach_bps[ii])
    Cndela_pdeg_v_Mach_alpha_deg_8[ii]  = fastInterp1(Mach_sample, Cndela_pdeg_vs_Mach_alpha_deg_8,  Mach_bps[ii])
    Cndela_pdeg_v_Mach_alpha_deg_16[ii] = fastInterp1(Mach_sample, Cndela_pdeg_vs_Mach_alpha_deg_16, Mach_bps[ii])
    Cndela_pdeg_v_Mach_alpha_deg_24[ii] = fastInterp1(Mach_sample, Cndela_pdeg_vs_Mach_alpha_deg_24, Mach_bps[ii])
    
# Reflect Cndela to negative AoA
Cndela_pdeg_v_Mach_alpha_deg_m8  = Cndela_pdeg_v_Mach_alpha_deg_8
Cndela_pdeg_v_Mach_alpha_deg_m16 = Cndela_pdeg_v_Mach_alpha_deg_16
Cndela_pdeg_v_Mach_alpha_deg_m24 = Cndela_pdeg_v_Mach_alpha_deg_24

# Combine data into a single 2D array
Cndela_pdeg_v_Mach_alpha_deg = np.column_stack((Cndela_pdeg_v_Mach_alpha_deg_m24, \
                                                    Cndela_pdeg_v_Mach_alpha_deg_m16, \
                                                    Cndela_pdeg_v_Mach_alpha_deg_m8, \
                                                    Cndela_pdeg_v_Mach_alpha_deg_0, \
                                                    Cndela_pdeg_v_Mach_alpha_deg_8, \
                                                    Cndela_pdeg_v_Mach_alpha_deg_16, \
                                                    Cndela_pdeg_v_Mach_alpha_deg_24))

# Interpolate table Cndela_pdeg_v_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Cndela_table_pdeg_v_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Cndela_table_pdeg_v_alpha_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                    Cndela_pdeg_v_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Cndela_pdeg_vs_Mach_plot == 'on':
    plt.figure('Cndela [1/deg] - Control Derivative')
    plt.plot(Mach_sample, Cndela_pdeg_vs_Mach_alpha_deg_0,  color='black',  linestyle='solid', label="AoA + dele 0 [deg] - Extracted")
    plt.plot(Mach_sample, Cndela_pdeg_vs_Mach_alpha_deg_8,  color='blue',   linestyle='solid', label="AoA + dele 8 [deg] - Extracted")
    plt.plot(Mach_sample, Cndela_pdeg_vs_Mach_alpha_deg_16, color='green',  linestyle='solid', label="AoA + dele 16 [deg] - Extracted")
    plt.plot(Mach_sample, Cndela_pdeg_vs_Mach_alpha_deg_24, color='red',    linestyle='solid', label="AoA + dele 24 [deg] - Extracted")   
    plt.plot(Mach_bps, Cndela_table_pdeg_v_alpha_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA + dele -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cndela_table_pdeg_v_alpha_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA + dele -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cndela_table_pdeg_v_alpha_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA + dele -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Cndela_table_pdeg_v_alpha_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA + dele 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Cndela_table_pdeg_v_alpha_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA + dele 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cndela_table_pdeg_v_alpha_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA + dele 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cndela_table_pdeg_v_alpha_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA + dele 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("Cndela_pdeg")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 22.   Coefficient: Cndelr [1/deg] = f(alpha_deg, Mach)
#       Method of determination: Computational
#       Details: Change in yaw moment due to change in rudder
#       Source: Walker 60
#------------------------------------------------------------------------------------------

# Cndelr table from comma separated variables exported from plot digitizer
Cndelr_pdeg_vs_Mach = pd.read_csv(
    'models/X15/aerodynamic_model/Walker60/data/Cndelr_vs_Mach_Walker60.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Mach_sample = Cndelr_pdeg_vs_Mach['x'].values
Cndelr_pdeg_vs_Mach_alpha_deg_0  = Cndelr_pdeg_vs_Mach['alpha_deg_0'].values
Cndelr_pdeg_vs_Mach_alpha_deg_8  = Cndelr_pdeg_vs_Mach['alpha_deg_8'].values
Cndelr_pdeg_vs_Mach_alpha_deg_16 = Cndelr_pdeg_vs_Mach['alpha_deg_16'].values
Cndelr_pdeg_vs_Mach_alpha_deg_24 = Cndelr_pdeg_vs_Mach['alpha_deg_24'].values

# We will populate Cndelr vectors for each AoA over the Mach values
Cndelr_pdeg_v_Mach_alpha_deg_0  = np.zeros(nMach_bps)
Cndelr_pdeg_v_Mach_alpha_deg_8  = np.zeros(nMach_bps)
Cndelr_pdeg_v_Mach_alpha_deg_16 = np.zeros(nMach_bps)
Cndelr_pdeg_v_Mach_alpha_deg_24 = np.zeros(nMach_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nMach_bps, 1):
    Cndelr_pdeg_v_Mach_alpha_deg_0[ii]  = fastInterp1(Mach_sample, Cndelr_pdeg_vs_Mach_alpha_deg_0,  Mach_bps[ii])
    Cndelr_pdeg_v_Mach_alpha_deg_8[ii]  = fastInterp1(Mach_sample, Cndelr_pdeg_vs_Mach_alpha_deg_8,  Mach_bps[ii])
    Cndelr_pdeg_v_Mach_alpha_deg_16[ii] = fastInterp1(Mach_sample, Cndelr_pdeg_vs_Mach_alpha_deg_16, Mach_bps[ii])
    Cndelr_pdeg_v_Mach_alpha_deg_24[ii] = fastInterp1(Mach_sample, Cndelr_pdeg_vs_Mach_alpha_deg_24, Mach_bps[ii])
    
# Reflect Cndelr to negative AoA
Cndelr_pdeg_v_Mach_alpha_deg_m8  = Cndelr_pdeg_v_Mach_alpha_deg_8
Cndelr_pdeg_v_Mach_alpha_deg_m16 = Cndelr_pdeg_v_Mach_alpha_deg_16
Cndelr_pdeg_v_Mach_alpha_deg_m24 = Cndelr_pdeg_v_Mach_alpha_deg_24

# Combine data into a single 2D array
Cndelr_pdeg_v_Mach_alpha_deg = np.column_stack((Cndelr_pdeg_v_Mach_alpha_deg_m24, \
                                                    Cndelr_pdeg_v_Mach_alpha_deg_m16, \
                                                    Cndelr_pdeg_v_Mach_alpha_deg_m8, \
                                                    Cndelr_pdeg_v_Mach_alpha_deg_0, \
                                                    Cndelr_pdeg_v_Mach_alpha_deg_8, \
                                                    Cndelr_pdeg_v_Mach_alpha_deg_16, \
                                                    Cndelr_pdeg_v_Mach_alpha_deg_24))

# Interpolate table Cndelr_pdeg_v_Mach_alpha_deg to breakpoints for Mach and alpha
alpha_deg_sample = np.array([-24, -16, -8, 0, 8, 16, 24])
Cndelr_table_pdeg_v_alpha_deg_Mach = np.zeros((nalpha_bps_deg, nMach_bps))
for ii in range(0, nMach_bps, 1):
    for jj in range(0, nalpha_bps_deg, 1):
        Cndelr_table_pdeg_v_alpha_deg_Mach[jj,ii]  = fastInterp2(Mach_bps, alpha_deg_sample, \
                                    Cndelr_pdeg_v_Mach_alpha_deg,  Mach_bps[ii], alpha_bps_deg[jj])

# Compare the results on a single plot
if Cndelr_pdeg_vs_Mach_plot == 'on':
    plt.figure('Cndelr [1/deg] - Control Derivative')
    plt.plot(Mach_sample, Cndelr_pdeg_vs_Mach_alpha_deg_0,  color='black',  linestyle='solid', label="AoA + dele 0 [deg] - Extracted")
    plt.plot(Mach_sample, Cndelr_pdeg_vs_Mach_alpha_deg_8,  color='blue',   linestyle='solid', label="AoA + dele 8 [deg] - Extracted")
    plt.plot(Mach_sample, Cndelr_pdeg_vs_Mach_alpha_deg_16, color='green',  linestyle='solid', label="AoA + dele 16 [deg] - Extracted")
    plt.plot(Mach_sample, Cndelr_pdeg_vs_Mach_alpha_deg_24, color='red',    linestyle='solid', label="AoA + dele 24 [deg] - Extracted")   
    plt.plot(Mach_bps, Cndelr_table_pdeg_v_alpha_deg_Mach[1,:],  color='grey', linestyle='dashed', label="AoA + dele -24 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cndelr_table_pdeg_v_alpha_deg_Mach[9,:],  color='grey', linestyle='dashed', label="AoA + dele -16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cndelr_table_pdeg_v_alpha_deg_Mach[17,:], color='grey', linestyle='dashed', label="AoA + dele -8 [deg] - Interpolated", marker='+')   
    plt.plot(Mach_bps, Cndelr_table_pdeg_v_alpha_deg_Mach[25,:], color='grey', linestyle='dashed', label="AoA + dele 0 [deg] - Interpolated",  marker='o', markerfacecolor='none')
    plt.plot(Mach_bps, Cndelr_table_pdeg_v_alpha_deg_Mach[33,:], color='grey', linestyle='dashed', label="AoA + dele 8 [deg] - Interpolated",  marker='x')
    plt.plot(Mach_bps, Cndelr_table_pdeg_v_alpha_deg_Mach[41,:], color='grey', linestyle='dashed', label="AoA + dele 16 [deg] - Interpolated", marker='s', markerfacecolor='none')
    plt.plot(Mach_bps, Cndelr_table_pdeg_v_alpha_deg_Mach[49,:], color='grey', linestyle='dashed', label="AoA + dele 24 [deg] - Interpolated", marker='+')
    plt.xlabel("Mach")
    plt.ylabel("Cndelr_pdeg")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Constants
lb2kg = 0.453592
slugft2_to_kgm2 = 1.355818

Weight_bps_lb = np.arange(14700, 33200, 200, dtype=float)
Mass_bps_kg = Weight_bps_lb * lb2kg
nWeight_bps = len(Weight_bps_lb)

# 23.   Moments of Inertia: Jxx and Jxz [slug-ft^2] = f(m_lb)
#       Method of determination: Measurement
#       Details: Resistance to angular acceleration around the body X-axis
#       Source: Yancey 64
#------------------------------------------------------------------------------------------

# Jxx and Jxz from comma separated variables exported from plot digitizer
Jxx_and_Jxz_vs_Weight = pd.read_csv(
    'models/X15/aerodynamic_model/Yancey64/data/Jxx_and_Jxz_vs_Weight_Yancey64.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Weight_sample = Jxx_and_Jxz_vs_Weight['x'].values
Jxx_slugft2_vs_Weight_lb  = Jxx_and_Jxz_vs_Weight['Jxx'].values
Jxz_slugft2_vs_Weight_lb  = Jxx_and_Jxz_vs_Weight['Jxz'].values

# We will populate Cndelr vectors for each AoA over the Mach values
Jxx_slugft2_v_Weight_lb  = np.zeros(nWeight_bps)
Jxz_slugft2_v_Weight_lb  = np.zeros(nWeight_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nWeight_bps, 1):
    Jxx_slugft2_v_Weight_lb[ii]  = fastInterp1(Weight_sample, Jxx_slugft2_vs_Weight_lb,  Weight_bps_lb[ii])
    Jxz_slugft2_v_Weight_lb[ii]  = fastInterp1(Weight_sample, Jxz_slugft2_vs_Weight_lb,  Weight_bps_lb[ii])

# Compare the results on a single plot
if Jxx_Jxz_vs_Weight_plot == 'on':
    plt.figure('Jxx and Jxz [slug-ft^2]')
    plt.plot(Weight_sample, Jxx_slugft2_vs_Weight_lb,  color='black',  linestyle='solid', label="Jxx [slug-ft^2] - Extracted")
    plt.plot(Weight_sample, Jxz_slugft2_vs_Weight_lb,  color='blue',   linestyle='solid', label="Jxz [slug-ft^2] - Extracted")
    plt.plot(Weight_bps_lb, Jxx_slugft2_v_Weight_lb,  color='grey', linestyle='dashed', label="Jxx [slug-ft^2] - Interpolated",  marker='x')
    plt.plot(Weight_bps_lb, Jxz_slugft2_v_Weight_lb,  color='grey', linestyle='dashed', label="Jxz [slug-ft^2] - Interpolated", marker='s', markerfacecolor='none')
    plt.xlabel("Weight (lb)")
    plt.xlabel("Jxx and Jxz (slug-ft^2)")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
Jxx_kgm2_v_Mass_kg = Jxx_slugft2_v_Weight_lb * slugft2_to_kgm2
Jxz_kgm2_v_Mass_kg = Jxz_slugft2_v_Weight_lb * slugft2_to_kgm2

# 24.   Moment of Inertia: Jyy [slug-ft^2] = f(m_lb)
#       Method of determination: Measurement
#       Details: Resistance to angular acceleration around the body Y-axis
#       Source: Yancey 64
#------------------------------------------------------------------------------------------

# Jxx and Jxz from comma separated variables exported from plot digitizer
Jyy_vs_Weight = pd.read_csv(
    'models/X15/aerodynamic_model/Yancey64/data/Jyy_vs_Weight_Yancey64.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Weight_sample = Jyy_vs_Weight['x'].values
Jyy_slugft2_vs_Weight_lb  = Jyy_vs_Weight['Jyy'].values

# We will populate Cndelr vectors for each AoA over the Mach values
Jyy_slugft2_v_Weight_lb  = np.zeros(nWeight_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nWeight_bps, 1):
    Jyy_slugft2_v_Weight_lb[ii]  = fastInterp1(Weight_sample, Jyy_slugft2_vs_Weight_lb,  Weight_bps_lb[ii])

# Compare the results on a single plot
if Jyy_vs_Weight_plot == 'on':
    plt.figure('Jyy [slug-ft^2]')
    plt.plot(Weight_sample, Jyy_slugft2_vs_Weight_lb,  color='black',  linestyle='solid', label="Jyy [slug-ft^2] - Extracted")
    plt.plot(Weight_bps_lb, Jyy_slugft2_v_Weight_lb,  color='grey', linestyle='dashed', label="Jyy [slug-ft^2] - Interpolated",  marker='x')
    plt.xlabel("Weight (lb)")
    plt.xlabel("Jyy (slug-ft^2)")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
Jyy_kgm2_v_Mass_kg = Jyy_slugft2_v_Weight_lb * slugft2_to_kgm2

# 25.   Moment of Inertia: Jzz [slug-ft^2] = f(m_lb)
#       Method of determination: Measurement
#       Details: Resistance to angular acceleration around the body Z-axis
#       Source: Yancey 64
#------------------------------------------------------------------------------------------

# Jxx and Jxz from comma separated variables exported from plot digitizer
Jzz_vs_Weight = pd.read_csv(
    'models/X15/aerodynamic_model/Yancey64/data/Jzz_vs_Weight_Yancey64.csv', 
    header = 0)

# Get data sequences from the pandas DataFrame
Weight_sample = Jzz_vs_Weight['x'].values
Jzz_slugft2_vs_Weight_lb  = Jzz_vs_Weight['Jzz'].values

# We will populate Cndelr vectors for each AoA over the Mach values
Jzz_slugft2_v_Weight_lb  = np.zeros(nWeight_bps)

# Interpolate to a regularly spaced database
for ii in range(0, nWeight_bps, 1):
    Jzz_slugft2_v_Weight_lb[ii]  = fastInterp1(Weight_sample, Jzz_slugft2_vs_Weight_lb,  Weight_bps_lb[ii])

# Compare the results on a single plot
if Jyy_vs_Weight_plot == 'on':
    plt.figure('Jzz [slug-ft^2]')
    plt.plot(Weight_sample, Jzz_slugft2_vs_Weight_lb,  color='black',  linestyle='solid', label="Jzz [slug-ft^2] - Extracted")
    plt.plot(Weight_bps_lb, Jzz_slugft2_v_Weight_lb,  color='grey', linestyle='dashed', label="Jzz [slug-ft^2] - Interpolated",  marker='x')
    plt.xlabel("Weight (lb)")
    plt.xlabel("Jzz (slug-ft^2)")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
Jzz_kgm2_v_Mass_kg = Jzz_slugft2_v_Weight_lb * slugft2_to_kgm2

# Build the aerodynamic database as a dictionary, save it
#-----------------------------------------------------------------------------------------------
X15_aerodynamic_database = {
    'alpha_bps_deg'                      : alpha_bps_deg,
    'alpha_p_dele_bps_deg'               : alpha_bps_deg,
    'Mach_bps'                           : Mach_bps,
    'CL_table_alpha_deg_Mach'            : CL_table_alpha_deg_Mach,
    'CLdele_table_pdeg_v_Mach_AoA_p_dele': CLdele_table_pdeg_v_Mach_AoA_p_dele,
    'Cmalpha_pdeg_table_alpha_deg_Mach'  : Cmalpha_pdeg_table_alpha_deg_Mach,
    'Cmdele_pdeg_table_AoApdele_deg_Mach': Cmdele_pdeg_table_AoApdele_deg_Mach,
    'Cmq_pdps_table_alpha_deg_Mach'      : Cmq_pdps_table_alpha_deg_Mach,
    'Clbeta_pdeg_table_Mach_alpha_deg'   : Clbeta_pdeg_table_Mach_alpha_deg,
    'CD_table_v_alpha_Mach'              : CD_table_v_alpha_Mach,
    'CYbeta_table_prad_v_alpha_deg_Mach' : CYbeta_table_pdeg_v_alpha_deg_Mach*57.3,
    'Cnbeta_table_prad_v_alpha_deg_Mach' : Cnbeta_table_pdeg_v_alpha_deg_Mach*57.3,
    'CYdelr_table_pdeg_v_alpha_deg_Mach' : CYdelr_table_pdeg_v_alpha_deg_Mach,
    'CYp_table_prps_v_alpha_deg_Mach'    : CYp_table_prps_v_alpha_deg_Mach,
    'CYr_table_prps_v_alpha_deg_Mach'    : CYr_table_prps_v_alpha_deg_Mach,
    'CYdela_table_pdeg_v_alpha_deg_Mach' : CYdela_table_pdeg_v_alpha_deg_Mach,
    'Clp_prps_table_Mach_alpha_deg'      : Clp_prps_table_Mach_alpha_deg,
    'Clr_prps_table_Mach_alpha_deg'      : Clr_prps_table_Mach_alpha_deg,
    'Cldela_table_pdeg_v_alpha_deg_Mach' : Cldela_table_pdeg_v_alpha_deg_Mach,
    'Cldelr_table_pdeg_v_alpha_deg_Mach' : Cldelr_table_pdeg_v_alpha_deg_Mach,
    'Cm_table_alpha_deg_Mach'            : Cm_table_alpha_deg_Mach,
    'Cnp_prps_table_Mach_alpha_deg'      : Cnp_prps_table_Mach_alpha_deg,
    'Cnr_prps_table_Mach_alpha_deg'      : Cnr_prps_table_Mach_alpha_deg,
    'Cndela_table_pdeg_v_alpha_deg_Mach' : Cndela_table_pdeg_v_alpha_deg_Mach,
    'Cndelr_table_pdeg_v_alpha_deg_Mach' : Cndelr_table_pdeg_v_alpha_deg_Mach,
    
    'mass_bps_kg'                        : Mass_bps_kg,
    'Jxx_kgm2_v_mass_kg'                 : Jxx_kgm2_v_Mass_kg,
    'Jxz_kgm2_v_mass_kg'                 : Jxz_kgm2_v_Mass_kg,
    'Jyy_kgm2_v_mass_kg'                 : Jyy_kgm2_v_Mass_kg,
    'Jzz_kgm2_v_mass_kg'                 : Jzz_kgm2_v_Mass_kg,
}

if save_database == 'on':
    np.savez('models/X15/aerodynamic_model/X15_aerodynamic_database', **X15_aerodynamic_database)
    
print('X15 aerodynamic database build from report data completed.')