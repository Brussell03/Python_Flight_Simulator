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
        
        # Initialize Aerodynamic Derivatives for Legacy Functions
        # These constants satisfy the AttributeError by providing the 
        # coefficients expected by the legacy module wrappers.
        self._init_aero_derivatives()

    def _init_aero_derivatives(self):
        """Initializes constant aerodynamic derivatives for legacy function calls."""
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

    def get_mass_properties(self, m_total_kg):
        """Interpolates and returns the inertia tensor based on current mass."""
        mass_bps = self.db["mass_bps_kg"]
        
        Jxx_b_kgm2 = fastInterp1(mass_bps, self.db["Jxx_mass_kg"], m_total_kg)
        Jyy_b_kgm2 = fastInterp1(mass_bps, self.db["Jyy_mass_kg"], m_total_kg)
        Jzz_b_kgm2 = fastInterp1(mass_bps, self.db["Jzz_mass_kg"], m_total_kg)
        Jxz_b_kgm2 = fastInterp1(mass_bps, self.db["Jxz_mass_kg"], m_total_kg)
        
        return Jxx_b_kgm2, Jyy_b_kgm2, Jzz_b_kgm2, Jxz_b_kgm2

    def get_aero_coeffs(self, alpha_deg, Mach, dele_ach_deg):
        """
        Calculates all non-dimensional aerodynamic coefficients for the current state.
        This removes the interpolation clutter from the EOM.
        """
        a_bps = self.db["alpha_bps_deg"]
        m_bps = self.db["Mach_bps"]
        a_dele_bps = self.db["alpha_p_dele_bps_deg"]
        
        # --- Lift ---
        CLwb = fastInterp2(a_bps, m_bps, self.db["CL_table_alpha_deg_Mach"], alpha_deg, Mach)
        CLdele = fastInterp2(a_dele_bps, m_bps, self.db["CLdele_table_pdeg_v_Mach_AoA_p_dele"], alpha_deg + dele_ach_deg, Mach)
        
        # --- Drag ---
        CDwb = fastInterp2(a_bps, m_bps, self.db["CD_table_v_alpha_Mach"], alpha_deg, Mach) + 0.06 # Added speed brake increment
        
        # --- Sideforce ---
        CYbeta = fastInterp2(a_bps, m_bps, self.db["CYbeta_table_prad_v_alpha_deg_Mach"], alpha_deg, Mach)
        CYdelr = fastInterp2(a_bps, m_bps, self.db["CYdelr_table_pdeg_v_alpha_deg_Mach"], alpha_deg, Mach)
        CYdela = fastInterp2(a_bps, m_bps, self.db["CYdela_table_pdeg_v_alpha_deg_Mach"], alpha_deg, Mach)
        CYp = fastInterp2(a_bps, m_bps, self.db["CYp_table_prps_v_alpha_deg_Mach"], alpha_deg, Mach)
        CYr = fastInterp2(a_bps, m_bps, self.db["CYr_table_prps_v_alpha_deg_Mach"], alpha_deg, Mach)
        
        # --- Roll Moment ---
        Clbeta = fastInterp2(a_bps, m_bps, self.db["Clbeta_pdeg_table_Mach_alpha_deg"], alpha_deg, Mach) * R2D
        Clp = fastInterp2(a_bps, m_bps, self.db["Clp_prps_table_Mach_alpha_deg"], alpha_deg, Mach) * 10 # Correction to get damping right
        Clr = fastInterp2(a_bps, m_bps, self.db["Clr_prps_table_Mach_alpha_deg"], alpha_deg, Mach)
        Cldela = fastInterp2(a_bps, m_bps, self.db["Cldela_table_pdeg_v_alpha_deg_Mach"], alpha_deg, Mach) * 0.5 # Correction from WT data
        Cldelr = fastInterp2(a_bps, m_bps, self.db["Cldelr_table_pdeg_v_alpha_deg_Mach"], alpha_deg, Mach)
        
        # --- Pitch Moment ---
        Cm = fastInterp2(a_bps, m_bps, self.db["Cm_table_alpha_deg_Mach"], alpha_deg, Mach)
        Cmdele = fastInterp2(a_dele_bps, m_bps, self.db["Cmdele_pdeg_table_AoApdele_deg_Mach"], alpha_deg + dele_ach_deg, Mach)
        Cmq = fastInterp2(a_bps, m_bps, self.db["Cmq_pdps_table_alpha_deg_Mach"], alpha_deg, Mach) * R2D
        
        # --- Yaw Moment ---
        Cnbeta = fastInterp2(a_bps, m_bps, self.db["Cnbeta_table_prad_v_alpha_deg_Mach"], alpha_deg, Mach)
        Cnp = fastInterp2(a_bps, m_bps, self.db["Cnp_prps_table_Mach_alpha_deg"], alpha_deg, Mach)
        Cnr = fastInterp2(a_bps, m_bps, self.db["Cnr_prps_table_Mach_alpha_deg"], alpha_deg, Mach)
        Cndela = fastInterp2(a_bps, m_bps, self.db["Cndela_table_pdeg_v_alpha_deg_Mach"], alpha_deg, Mach)
        Cndelr = fastInterp2(a_bps, m_bps, self.db["Cndelr_table_pdeg_v_alpha_deg_Mach"], alpha_deg, Mach)

        # Pack and return everything the EOM needs to calculate dimensional forces
        return {
            'CLwb': CLwb, 'CLdele': CLdele, 'CDwb': CDwb,
            'Cm': Cm, 'Cmdele': Cmdele, 'Cmq': Cmq,
            'CYbeta': CYbeta, 'CYdelr': CYdelr, 'CYdela': CYdela, 'CYp': CYp, 'CYr': CYr,
            'Clbeta': Clbeta, 'Clp': Clp, 'Clr': Clr, 'Cldela': Cldela, 'Cldelr': Cldelr,
            'Cnbeta': Cnbeta, 'Cnp': Cnp, 'Cnr': Cnr, 'Cndela': Cndela, 'Cndelr': Cndelr
        }
    
    def get_engine_burn_rate(self, throttle_perc):
        """Passes the throttle command to the XLR99 engine model."""
        return calculate_fuel_burn_rate(throttle_perc)

    def get_forces_and_moments(self, state_dict, control_dict, C_w2b):
        """
        Calculates full dimensional aerodynamic forces and moments mapped to the body frame.
        Applies engine jet damping and wind-to-body transformations natively.
        """
        # Unpack State
        alpha_rad = state_dict['alpha_rad']
        beta_rad = state_dict['beta_rad']
        alpha_deg = alpha_rad * R2D
        Mach = state_dict['Mach']
        qbar_kgpms2 = state_dict['qbar_kgpms2']
        true_airspeed_mps = state_dict['true_airspeed_mps']
        p_b_rps = state_dict['p_b_rps']
        q_b_rps = state_dict['q_b_rps']
        r_b_rps = state_dict['r_b_rps']
        
        # Unpack Controls
        dele_ach_deg = control_dict['dele_ach_deg']
        dela_ach_deg = control_dict['dela_ach_deg']
        delr_ach_deg = control_dict['delr_ach_deg']
        delsb_deg = control_dict['delsb_deg']
        throttle_perc = control_dict['throttle_perc']
        
        # Angle of attack / sideslip rates (static constraint for rigid body buildup)
        alphadot_rps = 0.0
        betadot_rps = 0.0

        # Retrieve isolated dimensionless coefficients
        aero = self.get_aero_coeffs(alpha_deg, Mach, dele_ach_deg)

        # Apply XLR99 Jet Damping Correction directly to dimensionless terms 
        aero['Cmq'] += -0.05 * throttle_perc
        aero['Cnr'] += -0.02 * throttle_perc

        # Dimensional Wind-Axis Forces
        drag_kgmps2 = CD_X15(aero['CDwb'], self.CDdele_pdeg, self.CDdelsb_pdeg, dele_ach_deg, delsb_deg) * qbar_kgpms2 * self.A_ref_m2
        lift_kgmps2 = CL_X15(aero['CLwb'], self.CLalpha_pdeg, aero['CLdele'], alpha_deg, dele_ach_deg) * qbar_kgpms2 * self.A_ref_m2
        side_kgmps2 = CY_X15(aero['CYbeta'], aero['CYp'], aero['CYr'], self.CYbetadot_pdps, aero['CYdela'], aero['CYdelr'], 
                             beta_rad, p_b_rps, r_b_rps, betadot_rps, dela_ach_deg, delr_ach_deg, true_airspeed_mps, self.b_m) * qbar_kgpms2 * self.A_ref_m2
        
        # Transform Forces to Body Frame using provided C_w2b DCM
        Fx_b_kgmps2 = -(C_w2b[0,0]*drag_kgmps2 + C_w2b[0,1]*side_kgmps2 + C_w2b[0,2]*lift_kgmps2)
        Fy_b_kgmps2 = -(C_w2b[1,0]*drag_kgmps2 + C_w2b[1,1]*side_kgmps2 + C_w2b[1,2]*lift_kgmps2)
        Fz_b_kgmps2 = -(C_w2b[2,0]*drag_kgmps2 + C_w2b[2,1]*side_kgmps2 + C_w2b[2,2]*lift_kgmps2)

        # Dimensional Body-Axis Moments
        l_b_kgm2ps2 = Cl_X15(aero['Clbeta'], aero['Clp'], aero['Clr'], self.Clbetadot_pdps, aero['Cldela'], aero['Cldelr'], 
                             beta_rad, betadot_rps, p_b_rps, r_b_rps, dela_ach_deg, delr_ach_deg, true_airspeed_mps, self.b_m) * qbar_kgpms2 * self.A_ref_m2 * self.b_m
        
        m_b_kgm2ps2 = Cm_X15(aero['Cm'], self.Cmalpha_pdeg, aero['Cmq'], self.Cmalphadot_prps, aero['Cmdele'], self.Cmdelsb_pdeg, 
                             alpha_deg, alphadot_rps, q_b_rps, dele_ach_deg, delsb_deg, true_airspeed_mps, self.c_m) * qbar_kgpms2 * self.A_ref_m2 * self.c_m
        
        n_b_kgm2ps2 = Cn_X15(aero['Cnbeta'], aero['Cnp'], aero['Cnr'], self.Cnbetadot_pdps, aero['Cndela'], aero['Cndelr'], 
                             beta_rad, betadot_rps, p_b_rps, r_b_rps, dela_ach_deg, delr_ach_deg, true_airspeed_mps, self.b_m) * qbar_kgpms2 * self.A_ref_m2 * self.b_m

        return Fx_b_kgmps2, Fy_b_kgmps2, Fz_b_kgmps2, l_b_kgm2ps2, m_b_kgm2ps2, n_b_kgm2ps2
    
    def pitch_control(self, t_s, q_b_rps, cmod):
        
        # Elevator motion due to pilot stick input
        if cmod["elevator"] == 'on':
            if cmod["type"] == 'doublet':
                if  t_s < cmod["t1_s"]:
                    dele_stick_deg = 0
                elif t_s < cmod["t2_s"]:
                    dele_stick_deg = -cmod["amplitude"]
                elif t_s < cmod["t3_s"]:
                    dele_stick_deg =  cmod["amplitude"]
                else:
                    dele_stick_deg = 0
            else:
                print("Error: type key not presently recognized in cmod.")
        elif cmod["elevator"] == 'off':
            dele_stick_deg = 0
        else:
            print("Error: elevator key in cmod dictionary must have value 'on' or 'off'.")
        
        # Elevator action is superposition of pilot input and SAS
        dele_deg = cmod["Kq"]*q_b_rps + dele_stick_deg
        
        return dele_deg

    # Roll control via aileron
    def roll_control(self, t_s, p_b_rps, r_b_rps, cmod):
        
        # Aileron motion due to pilot stick input
        if cmod["aileron"] == 'on':
            if cmod["type"] == 'doublet':
                if  t_s < cmod["t1_s"]:
                    dela_stick_deg = 0
                elif t_s < cmod["t2_s"]:
                    dela_stick_deg = -cmod["amplitude"]
                elif t_s < cmod["t3_s"]:
                    dela_stick_deg =  cmod["amplitude"]
                else:
                    dela_stick_deg = 0
            else:
                print("Error: type key not presently recognized in cmod.")
        elif cmod["aileron"] == 'off':
            dela_stick_deg = 0
        else:
            print("Error: aileron key in cmod dictionary must have value 'on' or 'off'.")
        
        # Aileron deflection due to pilot input and SAS
        dela_deg = cmod["Kp"]*p_b_rps + cmod["Kyar"]*r_b_rps + dela_stick_deg
        
        return dela_deg

    # Yaw control via rudder
    def yaw_control(self, t_s, r_b_rps, cmod):
        
        if cmod["rudder"] == 'on':
            if cmod["type"] == 'doublet':
                if  t_s < cmod["t1_s"]:
                    delr_pedal_deg = 0
                elif t_s < cmod["t2_s"]:
                    delr_pedal_deg = -cmod["amplitude"]
                elif t_s < cmod["t3_s"]:
                    delr_pedal_deg =  cmod["amplitude"]
                else:
                    delr_pedal_deg = 0
            else:
                print("Error: type key not presently recognized in cmod.")
        elif cmod["rudder"] == 'off':
            delr_pedal_deg = 0
        else:
            print("Error: rudder key in cmod dictionary must have value 'on' or 'off'.")
        
        delr_deg = cmod["Kr"]*r_b_rps + delr_pedal_deg
        
        return delr_deg

    def get_sas_commands(self, t, x, cmod):
        """
        Routes the Stability Augmentation System.
        """
        p_b_rps, q_b_rps, r_b_rps = x[3], x[4], x[5]
        
        dela_cmd = self.roll_control(t, p_b_rps, r_b_rps, cmod)
        dele_cmd = self.pitch_control(t, q_b_rps, cmod)
        delr_cmd = self.yaw_control(t, r_b_rps, cmod)
        
        return dela_cmd, dele_cmd, delr_cmd