import math
import numpy as np
import pandas as pd
from models.vehicle_base import Vehicle

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
    def __init__(self, data_path='models/X15/aerodynamic_model/X15_aerodynamic_database.npz', time_history_path=None):
        # Load database into memory once
        self.db = np.load(data_path)
        
        # Geometry
        self.b_m = 22.36 * FT2M
        self.c_m = 10.27 * FT2M
        self.A_ref_m2 = 18.6
        
        # Mass Bounds
        self.m_dry_kg = 14700 * LB2KG
        self.m_wet_kg = 33000 * LB2KG
        
        # Actuation Time Constants (First-order lag)
        self.tau_a_s = 0.1
        self.tau_e_s = 0.1
        self.tau_r_s = 0.1
        
        # Actuation Position Limits [min_deg, max_deg]
        self.lim_a_pos_deg = [-15.0, 15.0]  # Differential tail roll limit
        self.lim_e_pos_deg = [-35.0, 15.0]  # Pitch limit (usually more trailing-edge up authority)
        self.lim_r_pos_deg = [-7.5, 7.5]  # Rudder limit
        
        # Actuation Rate Limits [deg/s]
        self.lim_a_rate_dps = 50.0
        self.lim_e_rate_dps = 50.0
        self.lim_r_rate_dps = 50.0
        
        # --- Time History Data ---
        self.time_history_s = None
        self.aileron_time_history_deg = None
        self.elevator_time_history_deg = None
        self.rudder_time_history_deg = None
        
        if time_history_path is not None:
            self._load_time_history(time_history_path)
        
        # Initialize Aerodynamic Derivatives and other tables
        self._init_tables()
    
    def _load_time_history(self, path):
        """
        Extracts arrays from a dictionary-structured .npy file.
        Expects keys: 'time', 'elevator', 'aileron', 'rudder'.
        """
        time_hist_data            = pd.read_csv(path, header=0)
        self.time_history_s            = time_hist_data.get('x').values if time_hist_data.get('x') is not None else None
        self.elevator_time_history_deg = time_hist_data.get('dele_deg_vs_time_s').values if time_hist_data.get('dele_deg_vs_time_s') is not None else None
        self.aileron_time_history_deg  = time_hist_data.get('dela_deg_vs_time_s').values if time_hist_data.get('dela_deg_vs_time_s') is not None else None
        self.rudder_time_history_deg   = time_hist_data.get('delr_deg_vs_time_s').values if time_hist_data.get('delr_deg_vs_time_s') is not None else None

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
        
        # Jxx_b_slugft2 = 3600
        # Jxx_b_kgm2 = Jxx_b_slugft2*1.355
        # Jxz_b_slugft2 = -700
        # Jxz_b_kgm2 = Jxz_b_slugft2*1.355
        # Jyy_b_slugft2 = 86000
        # Jyy_b_kgm2 = Jyy_b_slugft2*1.355
        # Jzz_b_slugft2 = 88500
        # Jzz_b_kgm2 = Jzz_b_slugft2*1.355
        
        return Jxx_b_kgm2, Jyy_b_kgm2, Jzz_b_kgm2, Jxz_b_kgm2

    def get_aero_coeffs(self, alpha_deg, Mach, dele_ach_deg):
        """
        Calculates all non-dimensional aerodynamic coefficients for the current state.
        """
        
        # --- Lift ---
        CLwb = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CL_table_alpha_deg_Mach, alpha_deg, Mach)
        CLdele_pdeg = fastInterp2(self.alpha_p_dele_bps_deg, self.Mach_bps, self.CLdele_table_pdeg_v_Mach_AoA_p_dele, alpha_deg + dele_ach_deg, Mach)
        
        # --- Drag ---
        CDwb = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CD_table_v_alpha_Mach, alpha_deg, Mach)# + 0.06 # Added speed brake increment
        
        # --- Sideforce ---
        CYbeta_prad = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CYbeta_table_prad_v_alpha_deg_Mach, alpha_deg, Mach)
        CYdelr_pdeg = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CYdelr_table_pdeg_v_alpha_deg_Mach, alpha_deg, Mach)
        CYdela_pdeg = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CYdela_table_pdeg_v_alpha_deg_Mach, alpha_deg, Mach)
        CYp_prps = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CYp_table_prps_v_alpha_deg_Mach, alpha_deg, Mach)
        CYr_prps = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.CYr_table_prps_v_alpha_deg_Mach, alpha_deg, Mach)
        
        # --- Roll Moment ---
        Clbeta_prad = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Clbeta_pdeg_table_Mach_alpha_deg, alpha_deg, Mach) * R2D
        Clp_prps = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Clp_prps_table_Mach_alpha_deg, alpha_deg, Mach)# * 10 # Correction to get damping right
        Clr_prps = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Clr_prps_table_Mach_alpha_deg, alpha_deg, Mach)
        Cldela_pdeg = fastInterp2(self.alpha_bps_deg, self.Mach_bps, self.Cldela_table_pdeg_v_alpha_deg_Mach, alpha_deg, Mach)# * 0.5 # Correction from WT data
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
        # Fx_b_kgmps2 = -C_w2b[0,0]*drag_kgmps2 + C_w2b[0,1]*side_kgmps2 - C_w2b[0,2]*lift_kgmps2
        # Fy_b_kgmps2 = -C_w2b[1,0]*drag_kgmps2 + C_w2b[1,1]*side_kgmps2 - C_w2b[1,2]*lift_kgmps2
        # Fz_b_kgmps2 = -C_w2b[2,0]*drag_kgmps2 + C_w2b[2,1]*side_kgmps2 - C_w2b[2,2]*lift_kgmps2

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
        
        if cmod["elevator"]:
            # Determine pilot input type
            input_type = cmod.get("type", "doublet")
            
            if input_type == "doublet":
                # Elevator motion due to pilot stick input
                if cmod["t1_s"] <= t_s <= cmod["t3_s"]:
                    dele_stick_deg = -cmod["amplitude"] if t_s < cmod["t2_s"] else cmod["amplitude"]
            
            elif input_type == "time_history":
                if self.time_history_s is not None and self.elevator_time_history_deg is not None:
                    dele_stick_deg = fastInterp1(self.time_history_s, self.elevator_time_history_deg, t_s)
        
        # SAS feedback applied conditionally
        dele_sas_deg = cmod["Kq"] * q_b_rps if cmod["sas"] else 0.0
        
        # Elevator action is superposition of pilot input and SAS
        return dele_sas_deg + dele_stick_deg

    # Roll control via aileron
    def roll_control(self, t_s, p_b_rps, r_b_rps, cmod):
        dela_stick_deg = 0.0
        
        if cmod["aileron"]:
            # Determine pilot input type
            input_type = cmod.get("type", "doublet")
            
            if input_type == "doublet":
                # Aileron motion due to pilot stick input
                if cmod["t1_s"] <= t_s <= cmod["t3_s"]:
                    dela_stick_deg = -cmod["amplitude"] if t_s < cmod["t2_s"] else cmod["amplitude"]
            
            elif input_type == "time_history":
                if self.time_history_s is not None and self.aileron_time_history_deg is not None:
                    dela_stick_deg = fastInterp1(self.time_history_s, self.aileron_time_history_deg, t_s)
        
        # SAS feedback applied conditionally
        dela_sas_deg = (cmod["Kp"] * p_b_rps + cmod["Kyar"] * r_b_rps) if cmod["sas"] else 0.0
        
        # Aileron deflection due to pilot input and SAS
        return dela_sas_deg + dela_stick_deg

    # Yaw control via rudder
    def yaw_control(self, t_s, r_b_rps, cmod):
        delr_pedal_deg = 0.0
        
        if cmod["rudder"]:
            # Determine pilot input type
            input_type = cmod.get("type", "doublet")
            
            if input_type == "doublet":
                # Rudder motion due to pilot pedal input
                if cmod["t1_s"] <= t_s <= cmod["t3_s"]:
                    delr_pedal_deg = -cmod["amplitude"] if t_s < cmod["t2_s"] else cmod["amplitude"]
            
            elif input_type == "time_history":
                if self.time_history_s is not None and self.rudder_time_history_deg is not None:
                    delr_pedal_deg = fastInterp1(self.time_history_s, self.rudder_time_history_deg, t_s)
        
        # SAS feedback applied conditionally
        delr_sas_deg = cmod["Kr"] * r_b_rps if cmod["sas"] else 0.0
            
        return delr_sas_deg + delr_pedal_deg

    def get_trim_values(self, trim_list):
        if trim_list is None:
            return 0, 0, 0
        return trim_list[:3]

    def get_sas_commands(self, t, x, cmod, u_trim):
        """
        Routes the Stability Augmentation System and superimposes commands over trim baseline.
        u_trim is expected as [dela_trim, dele_trim, delr_trim, throttle_trim]
        """
        p_b_rps, q_b_rps, r_b_rps = x[3], x[4], x[5]
        
        # Extract trim baselines
        dela_trim_deg, dele_trim_deg, delr_trim_deg = 0, 0, 0#self.get_trim_values(u_trim)
        
        # Calculate dynamic commands (Stick + Feedback)
        dela_dynamic_deg = self.roll_control(t, p_b_rps, r_b_rps, cmod)
        dele_dynamic_deg = self.pitch_control(t, q_b_rps, cmod)
        delr_dynamic_deg = self.yaw_control(t, r_b_rps, cmod)
        
        # Superimpose dynamic commands onto trim baseline
        dela_cmd_deg = dela_trim_deg + dela_dynamic_deg
        dele_cmd_deg = dele_trim_deg + dele_dynamic_deg
        delr_cmd_deg = delr_trim_deg + delr_dynamic_deg
        
        return dela_cmd_deg, dele_cmd_deg, delr_cmd_deg
    
    def actuator_kinematics(self, cmd_deg, ach_deg, tau_s, pos_lims, rate_lim_dps):
        """
        Computes actuator state derivative enforcing rate and position saturation.
        """
        # 1. Compute unbounded linear rate
        rate_dps = (cmd_deg - ach_deg) / tau_s
        
        # 2. Enforce Rate Saturation (Hydraulic limit)
        rate_dps = np.clip(rate_dps, -rate_lim_dps, rate_lim_dps)
        
        # 3. Enforce Position Saturation (Mechanical hard stops)
        # If we are at or beyond the max limit and trying to push further, rate is zero
        if ach_deg >= pos_lims[1] and rate_dps > 0.0:
            rate_dps = 0.0
        # If we are at or below the min limit and trying to push further, rate is zero
        elif ach_deg <= pos_lims[0] and rate_dps < 0.0:
            rate_dps = 0.0
            
        return rate_dps
    
    def aileron_kinematics(self, dela_cmd_deg, dela_ach_deg):
        return self.actuator_kinematics(dela_cmd_deg, dela_ach_deg, self.tau_a_s, self.lim_a_pos_deg, self.lim_a_rate_dps)
    
    def elevator_kinematics(self, dele_cmd_deg, dele_ach_deg):
        return self.actuator_kinematics(dele_cmd_deg, dele_ach_deg, self.tau_e_s, self.lim_e_pos_deg, self.lim_e_rate_dps)
    
    def rudder_kinematics(self, delr_cmd_deg, delr_ach_deg):
        return self.actuator_kinematics(delr_cmd_deg, delr_ach_deg, self.tau_r_s, self.lim_r_pos_deg, self.lim_r_rate_dps)