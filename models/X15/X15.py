import math
import numpy as np
from models.base import Vehicle

from models.X15.aerodynamics.drag_coef_X15 import CD_X15
from models.X15.aerodynamics.lift_coef_X15 import CL_X15
from models.X15.aerodynamics.sideforce_coef_X15 import CY_X15
from models.X15.aerodynamics.roll_coef_X15 import Cl_X15
from models.X15.aerodynamics.pitch_coef_X15 import Cm_X15
from models.X15.aerodynamics.yaw_coef_X15 import Cn_X15
from models.X15.engine.XLR99 import calculate_fuel_burn_rate

from src.utils.constants import FT2M, LB2KG, R2D
from src.utils.interpolators import fastInterp1, fastInterp2

class X15(Vehicle):
    def __init__(self, data_path='models/X15/aerodynamic_model/X15_aerodynamic_database.npz'):
        # Load database into memory once
        self.db = np.load(data_path)
        
        # Geometry
        self.b_m = 22.36 * FT2M
        self.c_m = 10.27 * FT2M
        self.A_ref_m2 = 18.6
        
        # Mass Bounds
        self.m_dry_kg = 14700 * LB2KG
        self.m_wet_kg = 33000 * LB2KG
        
        # Actuation Time Constants
        self.tau_a_s = 0.1
        self.tau_e_s = 0.1
        self.tau_r_s = 0.1
        
        # Initialize Aerodynamic Derivatives and other tables
        self._init_tables()

    def _init_tables(self):
        """Initializes constant aerodynamic derivatives and other tables."""
        
        self.alpha_bps_deg = self.db["alpha_bps_deg"]
        self.Mach_bps = self.db["Mach_bps"]
        self.alpha_p_dele_bps_deg = self.db["alpha_p_dele_bps_deg"]
        
        self.Jxx_kgm2_v_mass_kg = self.db["Jxx_kgm2_v_mass_kg"]
        self.Jyy_kgm2_v_mass_kg = self.db["Jyy_kgm2_v_mass_kg"]
        self.Jzz_kgm2_v_mass_kg = self.db["Jzz_kgm2_v_mass_kg"]
        self.Jxz_kgm2_v_mass_kg = self.db["Jxz_kgm2_v_mass_kg"]
        
        # Lift derivatives
        self.CLalpha_pdeg = 0.0
        
        # Drag derivatives
        self.CDdele_pdeg = 0.0
        self.CDdelsb_pdeg = 0.0
        
        # Sideforce derivatives
        self.CYbetadot_pdps = 0.0
        
        # Roll derivatives
        self.Clbetadot_pdps = 0.0
        
        # Pitch derivatives
        self.Cmalpha_pdeg = 0.0
        self.Cmalphadot_prps = 0.0
        self.Cmdelsb_pdeg = 0.0
        
        # Yaw derivatives
        self.Cnbetadot_pdps = 0.0
        
        # --- Lift ---
        self.CL_table_alpha_deg_Mach = self.db["CL_table_alpha_deg_Mach"]
        self.CLdele_table_pdeg_v_Mach_AoA_p_dele = self.db["CLdele_table_pdeg_v_Mach_AoA_p_dele"]
        
        # --- Drag ---
        self.CD_table_v_alpha_Mach = self.db["CD_table_v_alpha_Mach"]
        
        # --- Sideforce ---
        self.CYbeta_table_prad_v_alpha_deg_Mach = self.db["CYbeta_table_prad_v_alpha_deg_Mach"]
        self.CYdelr_table_pdeg_v_alpha_deg_Mach = self.db["CYdelr_table_pdeg_v_alpha_deg_Mach"]
        self.CYdela_table_pdeg_v_alpha_deg_Mach = self.db["CYdela_table_pdeg_v_alpha_deg_Mach"]
        self.CYp_table_prps_v_alpha_deg_Mach = self.db["CYp_table_prps_v_alpha_deg_Mach"]
        self.CYr_table_prps_v_alpha_deg_Mach = self.db["CYr_table_prps_v_alpha_deg_Mach"]
        
        # --- Roll Moment ---
        self.Clbeta_pdeg_table_Mach_alpha_deg = self.db["Clbeta_pdeg_table_Mach_alpha_deg"]
        self.Clp_prps_table_Mach_alpha_deg = self.db["Clp_prps_table_Mach_alpha_deg"]
        self.Clr_prps_table_Mach_alpha_deg = self.db["Clr_prps_table_Mach_alpha_deg"]
        self.Cldela_table_pdeg_v_alpha_deg_Mach = self.db["Cldela_table_pdeg_v_alpha_deg_Mach"]
        self.Cldelr_table_pdeg_v_alpha_deg_Mach = self.db["Cldelr_table_pdeg_v_alpha_deg_Mach"]
        
        # --- Pitch Moment ---
        self.Cm_table_alpha_deg_Mach = self.db["Cm_table_alpha_deg_Mach"]
        self.Cmdele_pdeg_table_AoApdele_deg_Mach = self.db["Cmdele_pdeg_table_AoApdele_deg_Mach"]
        self.Cmq_pdps_table_alpha_deg_Mach = self.db["Cmq_pdps_table_alpha_deg_Mach"]
        
        # --- Yaw Moment ---
        self.Cnbeta_table_prad_v_alpha_deg_Mach = self.db["Cnbeta_table_prad_v_alpha_deg_Mach"]
        self.Cnp_prps_table_Mach_alpha_deg = self.db["Cnp_prps_table_Mach_alpha_deg"]
        self.Cnr_prps_table_Mach_alpha_deg = self.db["Cnr_prps_table_Mach_alpha_deg"]
        self.Cndela_table_pdeg_v_alpha_deg_Mach = self.db["Cndela_table_pdeg_v_alpha_deg_Mach"]
        self.Cndelr_table_pdeg_v_alpha_deg_Mach = self.db["Cndelr_table_pdeg_v_alpha_deg_Mach"]

    def get_mass_properties(self, m_total_kg):
        """Interpolates and returns the inertia tensor based on current mass."""
        mass_bps = self.db["mass_bps_kg"]
        
        Jxx_b_kgm2 = fastInterp1(mass_bps, self.Jxx_kgm2_v_mass_kg, m_total_kg)
        Jyy_b_kgm2 = fastInterp1(mass_bps, self.Jyy_kgm2_v_mass_kg, m_total_kg)
        Jzz_b_kgm2 = fastInterp1(mass_bps, self.Jzz_kgm2_v_mass_kg, m_total_kg)
        Jxz_b_kgm2 = fastInterp1(mass_bps, self.Jxz_kgm2_v_mass_kg, m_total_kg)
        
        return Jxx_b_kgm2, Jyy_b_kgm2, Jzz_b_kgm2, Jxz_b_kgm2

    def get_aero_coeffs(self, alpha_deg, Mach, dele_ach_deg):
        """
        Calculates all non-dimensional aerodynamic coefficients for the current state.
        """
        
        # --- Lift ---
        CLwb = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CL_table_alpha_deg_Mach, alpha_deg, Mach)
        CLdele_pdeg = fastInterp2(self.alpha_p_dele_bps_deg, self.Mach_bps, self.CLdele_table_pdeg_v_Mach_AoA_p_dele, alpha_deg + dele_ach_deg, Mach)
        
        # --- Drag ---
        CDwb = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CD_table_v_alpha_Mach, alpha_deg, Mach) + 0.06 # Added speed brake increment
        
        # --- Sideforce ---
        CYbeta_prad = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CYbeta_table_prad_v_alpha_deg_Mach, alpha_deg, Mach)
        CYdelr_pdeg = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CYdelr_table_pdeg_v_alpha_deg_Mach, alpha_deg, Mach)
        CYdela_pdeg = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CYdela_table_pdeg_v_alpha_deg_Mach, alpha_deg, Mach)
        CYp_prps = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CYp_table_prps_v_alpha_deg_Mach, alpha_deg, Mach)
        CYr_prps = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CYr_table_prps_v_alpha_deg_Mach, alpha_deg, Mach)
        
        # --- Roll Moment ---
        Clbeta_prad = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Clbeta_pdeg_table_Mach_alpha_deg, alpha_deg, Mach) * R2D
        Clp_prps = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Clp_prps_table_Mach_alpha_deg, alpha_deg, Mach) * 10 # Correction to get damping right
        Clr_prps = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Clr_prps_table_Mach_alpha_deg, alpha_deg, Mach)
        Cldela_pdeg = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Cldela_table_pdeg_v_alpha_deg_Mach, alpha_deg, Mach) * 0.5 # Correction from WT data
        Cldelr_pdeg = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Cldelr_table_pdeg_v_alpha_deg_Mach, alpha_deg, Mach)
        
        # --- Pitch Moment ---
        Cm = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Cm_table_alpha_deg_Mach, alpha_deg, Mach)
        Cmdele_pdeg = fastInterp2(self.alpha_p_dele_bps_deg, self.Mach_bps, self.Cmdele_pdeg_table_AoApdele_deg_Mach, alpha_deg + dele_ach_deg, Mach)
        Cmq_prps = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Cmq_pdps_table_alpha_deg_Mach, alpha_deg, Mach) * R2D
        
        # --- Yaw Moment ---
        Cnbeta_prad = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Cnbeta_table_prad_v_alpha_deg_Mach, alpha_deg, Mach)
        Cnp_prps = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Cnp_prps_table_Mach_alpha_deg, alpha_deg, Mach)
        Cnr_prps = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Cnr_prps_table_Mach_alpha_deg, alpha_deg, Mach)
        Cndela_pdeg = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Cndela_table_pdeg_v_alpha_deg_Mach, alpha_deg, Mach)
        Cndelr_pdeg = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Cndelr_table_pdeg_v_alpha_deg_Mach, alpha_deg, Mach)

        # Packed tightly into a tuple
        return (CLwb, CLdele_pdeg, CDwb, Cm, Cmdele_pdeg, Cmq_prps,
                CYbeta_prad, CYdelr_pdeg, CYdela_pdeg, CYp_prps, CYr_prps,
                Clbeta_prad, Clp_prps, Clr_prps, Cldela_pdeg, Cldelr_pdeg,
                Cnbeta_prad, Cnp_prps, Cnr_prps, Cndela_pdeg, Cndelr_pdeg)
    
    def get_engine_burn_rate(self, throttle_perc):
        """Passes the throttle command to the XLR99 engine model."""
        return calculate_fuel_burn_rate(throttle_perc)

    def get_forces_and_moments(self, alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                               delr_ach_deg, delsb_deg, throttle_perc, C_w2b):
        """
        Calculates full dimensional aerodynamic forces and moments mapped to the body frame.
        Applies engine jet damping and wind-to-body transformations natively.
        """
        alpha_deg = alpha_rad * R2D
        
        # Angle of attack / sideslip rates (static constraint for rigid body buildup)
        alphadot_rps = 0.0
        betadot_rps = 0.0

        # Unpack tuple from dimensionless coefficient getter
        (CLwb, CLdele_pdeg, CDwb, Cm, Cmdele_pdeg, Cmq_prps,
         CYbeta_prad, CYdelr_pdeg, CYdela_pdeg, CYp_prps, CYr_prps,
         Clbeta_prad, Clp_prps, Clr_prps, Cldela_pdeg, Cldelr_pdeg,
         Cnbeta_prad, Cnp_prps, Cnr_prps, Cndela_pdeg, Cndelr_pdeg) = self.get_aero_coeffs(alpha_deg, Mach, dele_ach_deg)

        # Apply XLR99 Jet Damping Correction directly to dimensionless terms 
        Cmq_prps += -0.05 * throttle_perc
        Cnr_prps += -0.02 * throttle_perc

        # Dimensional Wind-Axis Forces
        drag_kgmps2 = CD_X15(CDwb, self.CDdele_pdeg, self.CDdelsb_pdeg, dele_ach_deg, delsb_deg) * qbar_kgpms2 * self.A_ref_m2
        lift_kgmps2 = CL_X15(CLwb, self.CLalpha_pdeg, CLdele_pdeg, alpha_deg, dele_ach_deg) * qbar_kgpms2 * self.A_ref_m2
        side_kgmps2 = CY_X15(CYbeta_prad, CYp_prps, CYr_prps, self.CYbetadot_pdps, CYdela_pdeg, CYdelr_pdeg, 
                             beta_rad, p_b_rps, r_b_rps, betadot_rps, dela_ach_deg, delr_ach_deg, true_airspeed_mps, self.b_m) * qbar_kgpms2 * self.A_ref_m2
        
        # Transform Forces to Body Frame using provided C_w2b DCM
        Fx_b_kgmps2 = -(C_w2b[0,0]*drag_kgmps2 + C_w2b[0,1]*side_kgmps2 + C_w2b[0,2]*lift_kgmps2)
        Fy_b_kgmps2 = -(C_w2b[1,0]*drag_kgmps2 + C_w2b[1,1]*side_kgmps2 + C_w2b[1,2]*lift_kgmps2)
        Fz_b_kgmps2 = -(C_w2b[2,0]*drag_kgmps2 + C_w2b[2,1]*side_kgmps2 + C_w2b[2,2]*lift_kgmps2)

        # Dimensional Body-Axis Moments
        l_b_kgm2ps2 = Cl_X15(Clbeta_prad, Clp_prps, Clr_prps, self.Clbetadot_pdps, Cldela_pdeg, Cldelr_pdeg, 
                             beta_rad, betadot_rps, p_b_rps, r_b_rps, dela_ach_deg, delr_ach_deg, true_airspeed_mps, self.b_m) * qbar_kgpms2 * self.A_ref_m2 * self.b_m
        
        m_b_kgm2ps2 = Cm_X15(Cm, self.Cmalpha_pdeg, Cmq_prps, self.Cmalphadot_prps, Cmdele_pdeg, self.Cmdelsb_pdeg, 
                             alpha_deg, alphadot_rps, q_b_rps, dele_ach_deg, delsb_deg, true_airspeed_mps, self.c_m) * qbar_kgpms2 * self.A_ref_m2 * self.c_m
        
        n_b_kgm2ps2 = Cn_X15(Cnbeta_prad, Cnp_prps, Cnr_prps, self.Cnbetadot_pdps, Cndela_pdeg, Cndelr_pdeg, 
                             beta_rad, betadot_rps, p_b_rps, r_b_rps, dela_ach_deg, delr_ach_deg, true_airspeed_mps, self.b_m) * qbar_kgpms2 * self.A_ref_m2 * self.b_m

        return Fx_b_kgmps2, Fy_b_kgmps2, Fz_b_kgmps2, l_b_kgm2ps2, m_b_kgm2ps2, n_b_kgm2ps2
    
    def pitch_control(self, t_s, q_b_rps, cmod):
        
        dele_stick_deg = 0.0
        
        # Elevator motion due to pilot stick input
        if cmod["elevator"] and (cmod["t1_s"] <= t_s <= cmod["t3_s"]):
            dele_stick_deg = -cmod["amplitude"] if t_s < cmod["t2_s"] else cmod["amplitude"]
        
        # Elevator action is superposition of pilot input and SAS
        return cmod["Kq"] * q_b_rps + dele_stick_deg

    # Roll control via aileron
    def roll_control(self, t_s, p_b_rps, r_b_rps, cmod):
        
        dela_stick_deg = 0.0
        
        # Aileron motion due to pilot stick input
        if cmod["aileron"] and (cmod["t1_s"] <= t_s <= cmod["t3_s"]):
            dela_stick_deg = -cmod["amplitude"] if t_s < cmod["t2_s"] else cmod["amplitude"]
        
        # Aileron deflection due to pilot input and SAS
        return cmod["Kp"] * p_b_rps + cmod["Kyar"] * r_b_rps + dela_stick_deg

    # Yaw control via rudder
    def yaw_control(self, t_s, r_b_rps, cmod):
        
        delr_pedal_deg = 0.0
        if cmod["rudder"] and (cmod["t1_s"] <= t_s <= cmod["t3_s"]):
            delr_pedal_deg = -cmod["amplitude"] if t_s < cmod["t2_s"] else cmod["amplitude"]
            
        return cmod["Kr"] * r_b_rps + delr_pedal_deg

    def get_sas_commands(self, t, x, cmod):
        """
        Routes the Stability Augmentation System.
        """
        p_b_rps, q_b_rps, r_b_rps = x[3], x[4], x[5]
        
        dela_cmd_deg = self.roll_control(t, p_b_rps, r_b_rps, cmod)
        dele_cmd_deg = self.pitch_control(t, q_b_rps, cmod)
        delr_cmd_deg = self.yaw_control(t, r_b_rps, cmod)
        
        return dela_cmd_deg, dele_cmd_deg, delr_cmd_deg