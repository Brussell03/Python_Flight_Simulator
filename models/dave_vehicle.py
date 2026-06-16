import math

import numpy as np
import pandas as pd
import pyJanus
from src.engine.state_mapping import StateIdxSlices, TrimStateIdxSlices
from models.vehicle_base import Vehicle
from src.utils.unit_conversion import UnitConverter
from src.utils.interpolators import fastInterp1
from src.utils.constants import D2R, FT2M, R2D

class DAVEVehicle(Vehicle):
    def get_var_def(self, system, *var_names):
        """Attempts to fetch a variabledef from the system.
        Accepts one or more potential names (e.g., system, "name1", "name2").
        Returns the first valid var_def found, or None if all fail.
        """
        for name in var_names:
            try:
                var_def = system.get_variabledef(name)
                if var_def is not None:
                    return var_def
            except:
                continue
        print(f"Variable {var_names} not found in {system}")
        return None

    def get_si_val(self, var_def, log = False):
        if var_def is not None:
            return UnitConverter.to_si(var_def.get_value(), str(var_def.units), log=log)
        return 0

    def set_var_val(self, var_def, value, log = False):
        if var_def is not None:
            var_def.set_value(UnitConverter.from_si(value, str(var_def.units), log=log))
        
    def __init__(self, name, short_name, aero_dml_path="models/cannonball/cannonball_aero.dml", inertia_dml_path="models/cannonball/cannonball_inertia.dml",
                 prop_dml_path=None, control_dml_path=None, time_history_path=None):
        super().__init__()
        self._vehicle_name = name
        self._short_name = short_name
        
        # ==========================================
        # 1. Inertia System Initialization
        # ==========================================
        self.inertia_sys = pyJanus.Janus(inertia_dml_path)
        
        self.mass_def = self.get_var_def(self.inertia_sys, "XMASS")
        
        self.Jxx_def = self.get_var_def(self.inertia_sys, "XIXX")
        self.Jyy_def = self.get_var_def(self.inertia_sys, "XIYY")
        self.Jzz_def = self.get_var_def(self.inertia_sys, "XIZZ")
        self.Jxz_def = self.get_var_def(self.inertia_sys, "XIZX")
        self.Jxy_def = self.get_var_def(self.inertia_sys, "XIXY")
        self.Jyz_def = self.get_var_def(self.inertia_sys, "XIYZ")
        
        self.x_cg_b_def = self.get_var_def(self.inertia_sys, "DXCG")
        self.y_cg_b_def = self.get_var_def(self.inertia_sys, "DYCG")
        self.z_cg_b_def = self.get_var_def(self.inertia_sys, "DZCG")
        
        self.cg_pct_mac_def = self.get_var_def(self.inertia_sys, "CG_PCT_MAC")
        self.c_ref = self.get_var_def(self.inertia_sys, "CBAR")
        
        # ==========================================
        # 2. Aero System Initialization
        # ==========================================
        self.aero_sys = pyJanus.Janus(aero_dml_path)
        
        # Constants
        self.S_ref_def = self.get_var_def(self.aero_sys, "SWING", "sref")
        self.S_ref_m2 = self.get_si_val(self.S_ref_def)
        
        self.b_ref = self.get_var_def(self.aero_sys, "BSPAN", "bspan")
        self.b_m = self.get_si_val(self.b_ref)
        
        if self.c_ref is None: self.c_ref = self.get_var_def(self.aero_sys, "CBAR", "cbar")
        self.c_m = self.get_si_val(self.c_ref)
        
        # Inputs
        self.true_airspeed_ref = self.get_var_def(self.aero_sys, "VRW", "vt")
        self.alpha_def = self.get_var_def(self.aero_sys, "alpha")
        self.beta_def = self.get_var_def(self.aero_sys, "beta")
        self.p_b_ref = self.get_var_def(self.aero_sys, "PB", "p")
        self.q_b_ref = self.get_var_def(self.aero_sys, "QB", "q")
        self.r_b_ref = self.get_var_def(self.aero_sys, "RB", "r")
        self.dele_def = self.get_var_def(self.aero_sys, "el")
        self.dela_def = self.get_var_def(self.aero_sys, "ail")
        self.delr_def = self.get_var_def(self.aero_sys, "rdr")
        
        # Outputs
        self.CL_def = self.get_var_def(self.aero_sys, "CL")
        self.CD_def = self.get_var_def(self.aero_sys, "CD")
        self.CY_def = self.get_var_def(self.aero_sys, "CY", "cy")
        self.CX_def = self.get_var_def(self.aero_sys, "CX", "cx")
        self.CZ_def = self.get_var_def(self.aero_sys, "CZ", "cz")
        self.Cl_def = self.get_var_def(self.aero_sys, "Cl", "cl")
        self.Cm_def = self.get_var_def(self.aero_sys, "Cm", "cm")
        self.Cn_def = self.get_var_def(self.aero_sys, "Cn", "cn")
        
        # ==========================================
        # 3. Prop System Initialization
        # ==========================================
        try:
            self.prop_sys = pyJanus.Janus(prop_dml_path)
        except Exception as e:
            self.prop_sys = None
            if prop_dml_path is not None:
                print(f"Propulsion system not found at: {prop_dml_path}")
                print(e)
        
        if self.prop_sys is not None:
            # Inputs
            self.delt_in_ref = self.get_var_def(self.prop_sys, "PWR") # [0, 100]
            self.alt_ref = self.get_var_def(self.prop_sys, "ALT")
            self.mach_ref = self.get_var_def(self.prop_sys, "RMACH")
            
            # Outputs
            self.F_thrust_x_ref = self.get_var_def(self.prop_sys, "FEX")
            self.F_thrust_y_ref = self.get_var_def(self.prop_sys, "FEY")
            self.F_thrust_z_ref = self.get_var_def(self.prop_sys, "FEZ")
            self.M_thrust_l_ref = self.get_var_def(self.prop_sys, "TEL")
            self.M_thrust_m_ref = self.get_var_def(self.prop_sys, "TEM")
            self.M_thrust_n_ref = self.get_var_def(self.prop_sys, "TEN")
        
        # ==========================================
        # 4. Control System Initialization
        # ==========================================
        try:
            self.control_sys = pyJanus.Janus(control_dml_path)
        except Exception as e:
            self.control_sys = None
            if control_dml_path is not None:
                print(f"Control system not found at: {control_dml_path}")
                print(e)
        
        self.maneuver_started = False
        self.maneuver_start_lat_rad = 0
        self.maneuver_start_long_rad = 0
        self.maneuver_start_heading_rad = 0
        
        if self.control_sys is not None:
            # Inputs
            self.throttle_pilot_ref = self.get_var_def(self.control_sys, "throttle") # [0, 1]
            self.pitch_stick_ref = self.get_var_def(self.control_sys, "longStk") # [-1, 1]
            self.roll_stick_ref = self.get_var_def(self.control_sys, "latStk") # [-1, 1]
            self.yaw_pedal_ref = self.get_var_def(self.control_sys, "pedal") # [-1, 1]
            self.sas_toggle_ref = self.get_var_def(self.control_sys, "sasOn") # 0 or 1
            self.ap_toggle_ref = self.get_var_def(self.control_sys, "apOn") # 0 or 1
            self.circumnavigator_toggle_ref = self.get_var_def(self.control_sys, "circlePoleSW") # 0 or 1
            
            # Navigator inputs
            self.lat_ref = self.get_var_def(self.control_sys, "ownshipN_deg")
            self.long_ref = self.get_var_def(self.control_sys, "ownshipE_deg")
            
            # Autopilot command inputs
            self.equiv_airspeed_cmd_ref = self.get_var_def(self.control_sys, "keasCmd") # Equivalent airspeed command
            self.alt_cmd_ref = self.get_var_def(self.control_sys, "altCmd") # Commanded altitude above sea level
            self.lat_dev_error_cmd_ref = self.get_var_def(self.control_sys, "latOffset") # +Right from course
            self.heading_cmd_ref = self.get_var_def(self.control_sys, "baseChiCmd") # True heading of desired ground track (+clockwise from north)
            
            # Sensor feedbacks for SAS and AP
            self.alt_ref = self.get_var_def(self.control_sys, "altMsl")
            self.equiv_airspeed_ref = self.get_var_def(self.control_sys, "Vequiv")
            self.alpha_gnc_def = self.get_var_def(self.control_sys, "alpha")
            self.beta_gnc_def = self.get_var_def(self.control_sys, "beta")
            self.phi_ref = self.get_var_def(self.control_sys, "phi")
            self.theta_ref = self.get_var_def(self.control_sys, "theta")
            self.psi_ref = self.get_var_def(self.control_sys, "psi")
            self.p_b_gnc_ref = self.get_var_def(self.control_sys, "pb")
            self.q_b_gnc_ref = self.get_var_def(self.control_sys, "qb")
            self.r_b_gnc_ref = self.get_var_def(self.control_sys, "rb")
            
            # Trimmed values of longitudinal controls
            self.delt_trim_ref = self.get_var_def(self.control_sys, "throttleTrim") # [0, 1]
            self.pitch_stick_trim_ref = self.get_var_def(self.control_sys, "longStkTrim") # [0, 1]
            
            # Outputs
            self.dele_ref = self.get_var_def(self.control_sys, "el") # Deflection angle
            self.dela_ref = self.get_var_def(self.control_sys, "ail") # Deflection angle
            self.delr_ref = self.get_var_def(self.control_sys, "rdr") # Deflection angle
            self.delt_out_ref = self.get_var_def(self.control_sys, "PWR") # [0, 100]
        
        # ==========================================
        # 5. Actuators
        # ==========================================
        # Actuation Time Constants (First-order lag)
        self.tau_a_s = 0.02
        self.tau_e_s = 0.02
        self.tau_r_s = 0.02
        self.tau_t_s = 0.1
        
        # Actuation Position Limits [min_rad, max_rad]
        self.lim_a_pos_rad = np.array([-21.5, 21.5]) * D2R
        self.lim_e_pos_rad = np.array([-25.0, 25.0]) * D2R
        self.lim_r_pos_rad = np.array([-30.0, 30.0]) * D2R
        self.lim_t_pos_pct = [0.0, 100.0]
        
        # Actuation Rate Limits [rad/s]
        self.lim_a_rate_rps = 80.0 * D2R
        self.lim_e_rate_rps = 80.0 * D2R
        self.lim_r_rate_rps = 80.0 * D2R
        self.lim_t_rate_pctps = 100.0
        
        # # --- Time History Data ---
        # self.time_history_s = None
        # self.aileron_time_history_deg = None
        # self.elevator_time_history_deg = None
        # self.rudder_time_history_deg = None
        
        # if time_history_path is not None:
        #     self._load_time_history(time_history_path)

    # ==========================================
    # Base Class Properties
    # ==========================================
    @property
    def vehicle_name(self) -> str:
        return self._vehicle_name
    
    @property
    def short_name(self) -> str:
        return self._short_name
    
    @property
    def m_dry_kg(self) -> float:
        return self.get_si_val(self.mass_def)
    
    @property
    def m_wet_kg(self) -> float:
        return self.get_si_val(self.mass_def)
    
    def _load_time_history(self, path):
        """
        Extracts arrays from a dictionary-structured .npy file.
        Expects keys: 'time', 'elevator', 'aileron', 'rudder', 'throttle'.
        """
        time_hist_data                 = pd.read_csv(path, header=0)
        self.time_history_s            = time_hist_data.get('x').values if time_hist_data.get('x') is not None else None
        self.elevator_time_history_deg = time_hist_data.get('dele_deg_vs_time_s').values if time_hist_data.get('dele_deg_vs_time_s') is not None else None
        self.aileron_time_history_deg  = time_hist_data.get('dela_deg_vs_time_s').values if time_hist_data.get('dela_deg_vs_time_s') is not None else None
        self.rudder_time_history_deg   = time_hist_data.get('delr_deg_vs_time_s').values if time_hist_data.get('delr_deg_vs_time_s') is not None else None
        self.throttle_time_history_pct = time_hist_data.get('delt_pct_vs_time_s').values if time_hist_data.get('delt_pct_vs_time_s') is not None else None

    # ==========================================
    # Dynamics and Aero Methods
    # ==========================================
    def get_mass_properties(self, m_total_kg):
        Jxx = self.get_si_val(self.Jxx_def)
        Jyy = self.get_si_val(self.Jyy_def)
        Jzz = self.get_si_val(self.Jzz_def)
        Jxz = self.get_si_val(self.Jxz_def)
        Jxy = self.get_si_val(self.Jxy_def)
        Jyz = self.get_si_val(self.Jyz_def)
        return [Jxx, Jyy, Jzz, Jxy, Jxz, Jyz]

    def get_engine_burn_rate(self, delt_ach_pct):
        return 0.0

    def get_forces_and_moments(self, alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_rad, dela_ach_rad, 
                               delr_ach_rad, delsb_deg, delt_ach_pct, C_w2b, speedbrake, h_m):
        
        # 1. Push all state and control inputs to the DAVE-ML aero system
        # Aero inputs
        self.set_var_val(self.true_airspeed_ref, true_airspeed_mps)
        self.set_var_val(self.alpha_def, alpha_rad)
        self.set_var_val(self.beta_def, beta_rad)
        self.set_var_val(self.p_b_ref, p_b_rps)
        self.set_var_val(self.q_b_ref, q_b_rps)
        self.set_var_val(self.r_b_ref, r_b_rps)
        self.set_var_val(self.dele_def, dele_ach_rad)
        self.set_var_val(self.dela_def, dela_ach_rad)
        self.set_var_val(self.delr_def, delr_ach_rad)
        
        # Prop inputs
        if self.prop_sys is not None:
            self.set_var_val(self.delt_in_ref, delt_ach_pct)
            self.set_var_val(self.alt_ref, h_m)
            self.set_var_val(self.mach_ref, Mach)
        
        # 2. Extract and Dimensionalize Forces based on available definitions
        if self.CX_def is not None and self.CZ_def is not None:
            # Native Body-Axis Coefficients
            CX = self.get_si_val(self.CX_def)
            CY = self.get_si_val(self.CY_def)
            CZ = self.get_si_val(self.CZ_def)
            
            Fx_b = qbar_kgpms2 * self.S_ref_m2 * CX
            Fy_b = qbar_kgpms2 * self.S_ref_m2 * CY
            Fz_b = qbar_kgpms2 * self.S_ref_m2 * CZ
            
        elif self.CL_def is not None and self.CD_def is not None:
            # Native Wind-Axis Coefficients
            CL = self.get_si_val(self.CL_def)
            CD = self.get_si_val(self.CD_def)
            CY = self.get_si_val(self.CY_def)
            
            D_N = qbar_kgpms2 * self.S_ref_m2 * CD
            Y_N = qbar_kgpms2 * self.S_ref_m2 * CY
            L_N = qbar_kgpms2 * self.S_ref_m2 * CL
            
            F_wind_N = np.array([-D_N, Y_N, -L_N])
            F_body_N = C_w2b @ F_wind_N
            Fx_b, Fy_b, Fz_b = F_body_N[0], F_body_N[1], F_body_N[2]
        
        else:
            raise ValueError("Aero model must define either (CX, CZ) or (CL, CD).")
        
        if self.prop_sys is not None:
            Fx_b += self.get_si_val(self.F_thrust_x_ref)
            Fy_b += self.get_si_val(self.F_thrust_y_ref)
            Fz_b += self.get_si_val(self.F_thrust_z_ref)
        
        # 3. Extract and Dimensionalize Moments at the Moment Reference Center (MRC)
        Cl = self.get_si_val(self.Cl_def)
        Cm = self.get_si_val(self.Cm_def)
        Cn = self.get_si_val(self.Cn_def)
        
        # Dimensionalize Body-Frame Moments at the Moment Reference Center (MRC)
        l_mrc_b = qbar_kgpms2 * self.S_ref_m2 * self.b_m * Cl
        m_mrc_b = qbar_kgpms2 * self.S_ref_m2 * self.c_m * Cm
        n_mrc_b = qbar_kgpms2 * self.S_ref_m2 * self.b_m * Cn
        
        if self.prop_sys is not None:
            l_mrc_b += self.get_si_val(self.M_thrust_l_ref)
            m_mrc_b += self.get_si_val(self.M_thrust_m_ref)
            n_mrc_b += self.get_si_val(self.M_thrust_n_ref)
        
        # Retrieve Current Center of Mass Offsets
        dx_cg_m = self.get_si_val(self.x_cg_b_def)
        dy_cg_m = self.get_si_val(self.y_cg_b_def)
        dz_cg_m = self.get_si_val(self.z_cg_b_def)
        
        # Transfer Moments from MRC to Center of Mass (CM)
        l_cg_b = l_mrc_b - dy_cg_m * Fz_b + dz_cg_m * Fy_b
        m_cg_b = m_mrc_b - dz_cg_m * Fx_b + dx_cg_m * Fz_b
        n_cg_b = n_mrc_b - dx_cg_m * Fy_b + dy_cg_m * Fx_b
        
        return Fx_b, Fy_b, Fz_b, l_cg_b, m_cg_b, n_cg_b

    # ==========================================
    # Pass-through Kinematics
    # ==========================================
    def get_control_trim_values(self, x_trim_ref):
        if x_trim_ref is None:
            return 0, 0, 0, 0
        return x_trim_ref[TrimStateIdxSlices.ACT_TRIM_SLICE]
    
    def set_gnc_inputs(self, t_s, cmod, amod, lat_rad, long_rad, h_m, alpha_rad, beta_rad, phi_rad, theta_rad, psi_rad, p_b_rps, q_b_rps, r_b_rps, true_airspeed_mps, rho_kgpm3, x_trim_ref):
        if not (cmod.get("sas", False) or cmod.get("ap", False) or cmod.get("circumnavigator", False)):
            return
        
        rho_0 = 1.225 # Standard sea-level density [kg/m^3]
        
        # Pilot inputs
        self.set_var_val(self.throttle_pilot_ref, 0) # [0, 1]
        self.set_var_val(self.pitch_stick_ref, 0) # [-1, 1]
        self.set_var_val(self.roll_stick_ref, 0) # [-1, 1]
        self.set_var_val(self.yaw_pedal_ref, 0) # [-1, 1]
        
        # Navigator inputs
        self.set_var_val(self.lat_ref, lat_rad)
        self.set_var_val(self.long_ref, long_rad)
        
        if cmod.get("sas", False):
            self.set_var_val(self.sas_toggle_ref, 1) # Engaged
        else:
            self.set_var_val(self.sas_toggle_ref, 0)
        
        # Autopilot command inputs
        ap_cfg = cmod.get('autopilot', {})
        ap_enable_time_s = ap_cfg.get("enable_time_s", 1)
        if ap_cfg.get("enabled", False) and t_s >= ap_enable_time_s:
            heading = ap_cfg.get('psi_deg') * D2R if ap_cfg.get('psi_deg') is not None else psi_rad
            
            if self.maneuver_started is False:
                self.maneuver_start_lat_rad = lat_rad
                self.maneuver_start_long_rad = long_rad
                self.maneuver_start_heading_rad = heading
                self.maneuver_started = True
            
            self.set_var_val(self.ap_toggle_ref, 1) # Engaged
            
            h_cmd_m = ap_cfg.get('h_ft') * FT2M
            self.set_var_val(self.alt_cmd_ref, h_cmd_m)
            
            equiv_airspeed_cmd_mps = ap_cfg.get('V_equiv_mps', None)
            if equiv_airspeed_cmd_mps is None:
                true_airspeed_cmd_mps = ap_cfg.get('V_mps')
                rho_cmd_kgpm3 = fastInterp1(amod["alt_m"], amod["rho_kgpm3"], h_cmd_m)
                equiv_airspeed_cmd_mps = true_airspeed_cmd_mps * math.sqrt(rho_cmd_kgpm3 / rho_0)
                
            self.set_var_val(self.equiv_airspeed_cmd_ref, equiv_airspeed_cmd_mps)
            
            if ap_cfg.get('lateral_move', False):
                
                # --- Lateral Deviation Calculation ---
                # Calculate linear displacements in meters
                dy = (lat_rad - self.maneuver_start_lat_rad) * 6378137.0
                dx = (long_rad - self.maneuver_start_long_rad) * math.cos(self.maneuver_start_lat_rad) * 6378137.0
                
                # Project displacement onto the normal of the starting heading
                # Heading is clockwise from North (0 rad is North, pi/2 is East)
                # Normal vector n = [cos(psi), -sin(psi)]
                lat_dev_m = (dx * math.cos(self.maneuver_start_heading_rad)) - (dy * math.sin(self.maneuver_start_heading_rad))
                
                lat_dev_target_m = ap_cfg.get('lat_dev_ft', 0) * FT2M
                
                # print(lat_dev_m)
                # print(lat_dev_target_m + lat_dev_m)
                self.set_var_val(self.lat_dev_error_cmd_ref, lat_dev_target_m + lat_dev_m)
            else:
                self.set_var_val(self.lat_dev_error_cmd_ref, 0)
            
            self.set_var_val(self.heading_cmd_ref, heading)
        else:
            self.set_var_val(self.ap_toggle_ref, 0)
        
        # Sensor feedbacks for SAS and AP
        if cmod.get("sas", False) or cmod.get("ap", False):
            self.set_var_val(self.alt_ref, h_m)
            
            # Equivalent Airspeed Calculation
            equiv_airspeed_mps = true_airspeed_mps * math.sqrt(rho_kgpm3 / rho_0)
            self.set_var_val(self.equiv_airspeed_ref, equiv_airspeed_mps)
            
            self.set_var_val(self.alpha_gnc_def, alpha_rad)
            self.set_var_val(self.beta_gnc_def, beta_rad)
            self.set_var_val(self.phi_ref, phi_rad)
            self.set_var_val(self.theta_ref, theta_rad)
            self.set_var_val(self.psi_ref, psi_rad)
            self.set_var_val(self.p_b_gnc_ref, p_b_rps)
            self.set_var_val(self.q_b_gnc_ref, q_b_rps)
            self.set_var_val(self.r_b_gnc_ref, r_b_rps)
        
        if cmod.get("circumnavigator", -1) != -1:
            self.set_var_val(self.circumnavigator_toggle_ref, cmod.get("circumnavigator")) # 1 = circle N pole, 0 = circle equator/Int'l date line intersection
        
        # --- Trimmed Values of Longitudinal Controls ---
        if x_trim_ref is not None:
            dela_trim_rad, dele_trim_rad, delr_trim_rad, delt_trim_pct = self.get_control_trim_values(x_trim_ref)
            
            # Scale throttle to [0, 1] from 0 - 100%
            # throttle_trim_norm = delt_trim_pct / 100.0
            throttle_trim_norm = 8 / 100
            self.set_var_val(self.delt_trim_ref, throttle_trim_norm)
            
            # Normalize pitch stick trim to perfectly invert the XML's elevator gearing
            # XML formula: el = -25.0 * totLongStk
            # pitch_stick_trim_norm = max(min(dele_trim_rad*R2D / -25.0, 1.0), -1.0)
            pitch_stick_trim_norm = max(min(-0.74 / -25.0, 1.0), -1.0)
            self.set_var_val(self.pitch_stick_trim_ref, pitch_stick_trim_norm)
        else:
            # Fallback if no trim vector is provided
            self.set_var_val(self.delt_trim_ref, 0.08)
            pitch_stick_trim_norm = max(min(-0.74 / -25.0, 1.0), -1.0)
            self.set_var_val(self.pitch_stick_trim_ref, pitch_stick_trim_norm)
    
    def get_sas_commands(self, t, x, cmod, x_trim_ref):
        """
        Routes the Stability Augmentation System and superimposes commands over trim baseline.
        """
        # Extract trim baselines
        dela_trim_rad, dele_trim_rad, delr_trim_rad, delt_trim_pct = self.get_control_trim_values(x_trim_ref)
        
        if not (cmod.get("sas", False) or cmod.get("ap", False) or cmod.get("circumnavigator", False)):
            return dela_trim_rad, dele_trim_rad, delr_trim_rad, delt_trim_pct
        
        p_b_rps, q_b_rps, r_b_rps = x[StateIdxSlices.ROT_SLICE]
        
        dela_dynamic_rad = self.roll_control(t, p_b_rps, r_b_rps, cmod)
        dele_dynamic_rad = self.pitch_control(t, q_b_rps, cmod)
        delr_dynamic_rad = self.yaw_control(t, r_b_rps, cmod)
        delt_dynamic_pct = self.throttle_control(t, cmod)
        
        # return dela_cmd_rad, dele_cmd_rad, delr_cmd_rad, delt_cmd_pct
        return dela_dynamic_rad, dele_dynamic_rad, delr_dynamic_rad, delt_dynamic_pct
    
    def actuator_kinematics(self, cmd_deg, ach_deg, tau_s, pos_lims, rate_lim_dps, dt=None):
        """
        Computes actuator state derivative enforcing rate and position saturation.
        """
        # 1. Compute unbounded linear rate
        if dt is not None and dt > 0:
            rate_dps = (cmd_deg - ach_deg) / dt
        else:
            # Fallback to standard tau if dt is missing
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
    
    def aileron_kinematics(self, dela_cmd_rad, dela_ach_rad):
        return self.actuator_kinematics(dela_cmd_rad*R2D, dela_ach_rad*R2D, self.tau_a_s, self.lim_a_pos_rad*R2D, self.lim_a_rate_rps*R2D)*D2R
    
    def elevator_kinematics(self, dele_cmd_rad, dele_ach_rad):
        return self.actuator_kinematics(dele_cmd_rad*R2D, dele_ach_rad*R2D, self.tau_e_s, self.lim_e_pos_rad*R2D, self.lim_e_rate_rps*R2D)*D2R
    
    def rudder_kinematics(self, delr_cmd_rad, delr_ach_rad):
        return self.actuator_kinematics(delr_cmd_rad*R2D, delr_ach_rad*R2D, self.tau_r_s, self.lim_r_pos_rad*R2D, self.lim_r_rate_rps*R2D)*D2R
    
    def throttle_kinematics(self, delt_cmd_pct, delt_ach_pct):
        return self.actuator_kinematics(delt_cmd_pct, delt_ach_pct, self.tau_t_s, self.lim_t_pos_pct, self.lim_t_rate_pctps)
    
    def roll_control(self, t_s, p_b_rps, r_b_rps, cmod):
        if not (cmod.get("sas", False) or cmod.get("ap", False) or cmod.get("circumnavigator", False)):
            return 0
        
        dela_cmd_rad = self.get_si_val(self.dela_ref)
        return dela_cmd_rad
    
    def pitch_control(self, t_s, q_b_rps, cmod):
        if not (cmod.get("sas", False) or cmod.get("ap", False) or cmod.get("circumnavigator", False)):
            return 0
        
        dele_cmd_rad = self.get_si_val(self.dele_ref, log=False)
        return dele_cmd_rad

    # Yaw control via rudder
    def yaw_control(self, t_s, r_b_rps, cmod):
        if not (cmod.get("sas", False) or cmod.get("ap", False) or cmod.get("circumnavigator", False)):
            return 0
        
        delr_cmd_rad = self.get_si_val(self.delr_ref)
        return delr_cmd_rad
    
    def throttle_control(self, t_s, cmod):
        if not (cmod.get("sas", False) or cmod.get("ap", False) or cmod.get("circumnavigator", False)):
            return 0
        
        delt_cmd_pct = self.get_si_val(self.delt_out_ref)
        return delt_cmd_pct