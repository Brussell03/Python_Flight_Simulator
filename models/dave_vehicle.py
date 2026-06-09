import numpy as np
import pandas as pd
import pyJanus
from models.vehicle_base import Vehicle
from src.utils.unit_conversion import UnitConverter
from src.utils.interpolators import fastInterp1

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
        return None

    def get_si_val(self, var_def):
        if var_def is not None:
            return UnitConverter.to_si(var_def.get_value(), str(var_def.units))
        return 0

    def set_var_val(self, var_def, value):
        if var_def is not None:
            var_def.set_value(UnitConverter.from_si(value, str(var_def.units)))
        
    def __init__(self, name, short_name, aero_dml_path="models/cannonball/cannonball_aero.dml", inertia_dml_path="models/cannonball/cannonball_inertia.dml",
                 prop_dml_path=None, control_dml_path=None, gnc_dml_path=None, time_history_path=None):
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
            self.delt_ref = self.get_var_def(self.prop_sys, "PWR")
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
        # 4. Actuators
        # ==========================================
        # Actuation Time Constants (First-order lag)
        self.tau_a_s = 0.1
        self.tau_e_s = 0.1
        self.tau_r_s = 0.1
        
        # Actuation Position Limits [min_deg, max_deg]
        self.lim_a_pos_deg = [-15.0, 15.0]  # Differential tail roll limit
        self.lim_e_pos_deg = [-35.0, 15.0]  # Pitch limit (usually more trailing-edge up authority)
        self.lim_r_pos_deg = [-7.5, 7.5]    # Rudder limit
        
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
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                               delr_ach_deg, delsb_deg, delt_ach_pct, C_w2b, speedbrake, h_m):
        
        # 1. Push all state and control inputs to the DAVE-ML aero system
        # Aero inputs
        self.set_var_val(self.true_airspeed_ref, true_airspeed_mps)
        self.set_var_val(self.alpha_def, alpha_rad)
        self.set_var_val(self.beta_def, beta_rad)
        self.set_var_val(self.p_b_ref, p_b_rps)
        self.set_var_val(self.q_b_ref, q_b_rps)
        self.set_var_val(self.r_b_ref, r_b_rps)
        self.set_var_val(self.dele_def, dele_ach_deg)
        self.set_var_val(self.dela_def, dela_ach_deg)
        self.set_var_val(self.delr_def, delr_ach_deg)
        
        # Prop inputs
        if self.prop_sys is not None:
            self.set_var_val(self.delt_ref, delt_ach_pct)
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
    def get_trim_values(self, trim_list):
        if trim_list is None:
            return 0, 0, 0, 0
        return trim_list[:4]
    
    def get_sas_commands(self, t, x, cmod, u_trim):
        """
        Routes the Stability Augmentation System and superimposes commands over trim baseline.
        u_trim is expected as [dela_trim, dele_trim, delr_trim, delt_trim]
        """
        p_b_rps, q_b_rps, r_b_rps = x[3], x[4], x[5]
        
        # Extract trim baselines
        dela_trim_deg, dele_trim_deg, delr_trim_deg, delt_trim_pct = self.get_trim_values(u_trim)
        
        # Calculate dynamic commands (Stick + Feedback)
        dela_dynamic_deg = self.roll_control(t, p_b_rps, r_b_rps, cmod)
        dele_dynamic_deg = self.pitch_control(t, q_b_rps, cmod)
        delr_dynamic_deg = self.yaw_control(t, r_b_rps, cmod)
        
        # Superimpose dynamic commands onto trim baseline
        dela_cmd_deg = dela_trim_deg + dela_dynamic_deg
        dele_cmd_deg = dele_trim_deg + dele_dynamic_deg
        delr_cmd_deg = delr_trim_deg + delr_dynamic_deg
        delt_cmd_pct = delt_trim_pct
        
        return dela_cmd_deg, dele_cmd_deg, delr_cmd_deg, delt_cmd_pct
    
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
    
    def aileron_kinematics(self, dela_cmd_deg, dela_ach_deg):
        return self.actuator_kinematics(dela_cmd_deg, dela_ach_deg, self.tau_a_s, self.lim_a_pos_deg, self.lim_a_rate_dps)
    
    def elevator_kinematics(self, dele_cmd_deg, dele_ach_deg):
        return self.actuator_kinematics(dele_cmd_deg, dele_ach_deg, self.tau_e_s, self.lim_e_pos_deg, self.lim_e_rate_dps)
    
    def rudder_kinematics(self, delr_cmd_deg, delr_ach_deg):
        return self.actuator_kinematics(delr_cmd_deg, delr_ach_deg, self.tau_r_s, self.lim_r_pos_deg, self.lim_r_rate_dps)
    
    def throttle_kinematics(self, delt_cmd_pct, delt_ach_pct):
        return 0.0
    
    def pitch_control(self, t_s, q_b_rps, cmod):
        dele_stick_deg = 0.0
        
        if cmod.get("elevator", False):
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
        dele_sas_deg = cmod.get("Kq", 0) * q_b_rps if cmod.get("sas", False) else 0.0
        
        # Elevator action is superposition of pilot input and SAS
        return dele_sas_deg + dele_stick_deg

    # Roll control via aileron
    def roll_control(self, t_s, p_b_rps, r_b_rps, cmod):
        dela_stick_deg = 0.0
        
        if cmod.get("aileron", False):
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
        dela_sas_deg = (cmod.get("Kp", 0) * p_b_rps + cmod.get("Kyar", 0) * r_b_rps) if cmod.get("sas", False) else 0.0
        
        # Aileron deflection due to pilot input and SAS
        return dela_sas_deg + dela_stick_deg

    # Yaw control via rudder
    def yaw_control(self, t_s, r_b_rps, cmod):
        delr_pedal_deg = 0.0
        
        if cmod.get("rudder", False):
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
        delr_sas_deg = cmod.get("Kr", 0) * r_b_rps if cmod.get("sas", False) else 0.0
        
        return delr_sas_deg + delr_pedal_deg