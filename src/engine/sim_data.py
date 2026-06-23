from dataclasses import dataclass
from typing import Optional
import inspect
import numpy as np
import pandas as pd

from src.environment.wind_model import WindModel
from src.environment.earth_model import EarthModel
from models.vehicle_base import Vehicle
from src.utils.constants import R2D

# Enforce kw_only so defaults can be assigned without breaking inheritance or ordering
@dataclass(kw_only=True)
class SimData:
    job_name: str = "Unknown Job"
    description: str = ""
    integrator: str = ""
    vehicle: Optional[Vehicle] = None
    earth_model: Optional[EarthModel] = None
    wind_model: Optional[WindModel] = None
    
    t_s: np.ndarray                 # simulation time steps in seconds
    u_b_mps: np.ndarray             # axial velocity of CM wrt inertial CS resolved in aircraft body fixed CS [m/s]
    v_b_mps: np.ndarray             # lateral velocity of CM wrt inertial CS resolved in aircraft body fixed CS [m/s]
    w_b_mps: np.ndarray             # normal velocity of CM wrt inertial CS resolved in aircraft body fixed CS [m/s]
    p_b_rps: np.ndarray             # roll angular velocity of body fixed CS with respect to inertial CS [rad/s]
    q_b_rps: np.ndarray             # pitch angular velocity of body fixed CS with respect to inertial CS [rad/s]
    r_b_rps: np.ndarray             # yaw angular velocity of body fixed CS with respect to inertial CS [rad/s]
    q0: np.ndarray                  # quaternion scalar component representation of body wrt local navigation frame
    q1: np.ndarray                  # quaternion vector i-axis component representation of body wrt local navigation frame
    q2: np.ndarray                  # quaternion vector j-axis component representation of body wrt local navigation frame
    q3: np.ndarray                  # quaternion vector k-axis component representation of body wrt local navigation frame
    x_e_m: np.ndarray               # x-position in ECEF frame [m]
    y_e_m: np.ndarray               # y-position in ECEF frame [m]
    z_e_m: np.ndarray               # z-position in ECEF frame [m]
    lat_rad: np.ndarray             # geodetic latitude of aircraft CM resolved in WGS84 ellipsoid framework [rad]
    long_rad: np.ndarray            # longitude of aircraft CM resolved in WGS84 ellipsoid framework [rad]
    h_m: np.ndarray                 # height of aircraft CM above the WGS84 reference ellipsoid [m]
    m_fuel_kg: np.ndarray           # current mass of usable fuel [kg]
    
    phi_rad: np.ndarray             # Euler roll angle mapping local navigation frame to body fixed CS [rad]
    theta_rad: np.ndarray           # Euler pitch angle mapping local navigation frame to body fixed CS [rad]
    psi_rad: np.ndarray             # Euler yaw (heading) angle mapping local navigation frame to body fixed CS [rad]
    phi_dot_rps: np.ndarray         # time derivative of Euler roll angle (kinematic roll rate) [rad/s]
    theta_dot_rps: np.ndarray       # time derivative of Euler pitch angle (kinematic pitch rate) [rad/s]
    psi_dot_rps: np.ndarray         # time derivative of Euler yaw angle (kinematic heading rate) [rad/s]
    p_nb_rps: np.ndarray            # roll angular velocity of body fixed CS relative to local navigation frame, resolved in body CS [rad/s]
    q_nb_rps: np.ndarray            # pitch angular velocity of body fixed CS relative to local navigation frame, resolved in body CS [rad/s]
    r_nb_rps: np.ndarray            # yaw angular velocity of body fixed CS relative to local navigation frame, resolved in body CS [rad/s]
    
    cs_mps: np.ndarray              # local speed of sound in the atmosphere [m/s]
    rho_kgpm3: np.ndarray           # local atmospheric air density [kg/m^3]
    p_kgpms2: np.ndarray            # local atmospheric ambient static pressure [N/m^2 or kg/(m*s^2)]
    T_K: np.ndarray                 # local atmospheric ambient absolute temperature [K]
    
    mach: np.ndarray                # Mach number (ratio of true airspeed to local speed of sound) [dimensionless]
    alpha_rad: np.ndarray           # aerodynamic angle of attack [rad]
    beta_rad: np.ndarray            # aerodynamic angle of sideslip [rad]
    true_airspeed_mps: np.ndarray   # magnitude of vehicle velocity vector relative to the local air mass [m/s]
    qbar_kgpms2: np.ndarray         # dynamic pressure [kg/ms^2]
    g_mag_mps2: np.ndarray          # local gravitational acceleration magnitude derived from position-dependent gravity model [m/s^2]
    
    u_n_mps: np.ndarray             # North component of ground velocity vector resolved in local navigation (NED) frame [m/s]
    v_n_mps: np.ndarray             # East component of ground velocity vector resolved in local navigation (NED) frame [m/s]
    w_n_mps: np.ndarray             # Down component of ground velocity vector resolved in local navigation (NED) frame [m/s]
    
    Fx_b_kgmps2: np.ndarray         # net external non-gravitational force along body fixed x-axis (aerodynamic + propulsive) [N]
    Fy_b_kgmps2: np.ndarray         # net external non-gravitational force along body fixed y-axis (aerodynamic + propulsive) [N]
    Fz_b_kgmps2: np.ndarray         # net external non-gravitational force along body fixed z-axis (aerodynamic + propulsive) [N]
    l_b_kgm2ps2: np.ndarray         # net external aerodynamic and propulsive rolling moment about body fixed x-axis [N*m]
    m_b_kgm2ps2: np.ndarray         # net external aerodynamic and propulsive pitching moment about body fixed y-axis [N*m]
    n_b_kgm2ps2: np.ndarray         # net external aerodynamic and propulsive yawing moment about body fixed z-axis [N*m]
    n_x: np.ndarray                 # load factor on the body x-axis
    n_y: np.ndarray                 # load factor on the body y-axis
    n_z: np.ndarray                 # load factor on the body z-axis
    
    dela_ach_rad: np.ndarray        # achieved aileron surface deflection angle [rad]
    dele_ach_rad: np.ndarray        # achieved elevator surface deflection angle [rad]
    delr_ach_rad: np.ndarray        # achieved rudder surface deflection angle [rad]
    delt_ach_pct: np.ndarray        # achieved engine throttle [0.0 to 100.0%]
    
    dela_cmd_rad: np.ndarray        # commanded aileron surface deflection angle from flight control system [rad]
    dele_cmd_rad: np.ndarray        # commanded elevator surface deflection angle from flight control system [rad]
    delr_cmd_rad: np.ndarray        # commanded rudder surface deflection angle from flight control system [rad]
    delt_cmd_pct: np.ndarray        # engine throttle lever position command [0.0 to 100.0%]
    
    dela_trim_rad: np.ndarray        # trimmed aileron surface deflection angle [rad]
    dele_trim_rad: np.ndarray        # trimmed elevator surface deflection angle [rad]
    delr_trim_rad: np.ndarray        # trimmed rudder surface deflection angle [rad]
    delt_trim_pct: np.ndarray        # trimmed engine throttle [0.0 to 100.0%]
    
    W_N_mps: np.ndarray             # North wind component resolved in local navigation (NED) frame [m/s]
    W_E_mps: np.ndarray             # East wind component resolved in local navigation (NED) frame [m/s]
    W_D_mps: np.ndarray             # Down wind component resolved in local navigation (NED) frame [m/s]
    
    def __post_init__(self):
        """Attempts to calculate missing telemetry data using established kinematic relationships."""
        
        # 1. Euler Angles (Derived from Quaternions)
        has_quat = not (np.isnan(self.q0).all() and np.isnan(self.q1).all() and np.isnan(self.q2).all() and np.isnan(self.q3).all())
        if has_quat:
            # Fallback to identity quaternion for partial data
            q0_s = np.nan_to_num(self.q0, nan=1.0)
            q1_s = np.nan_to_num(self.q1, nan=0.0)
            q2_s = np.nan_to_num(self.q2, nan=0.0)
            q3_s = np.nan_to_num(self.q3, nan=0.0)
            
            phi_calc = np.arctan2(2 * (q0_s * q1_s + q2_s * q3_s), 1 - 2 * (q1_s**2 + q2_s**2))
            theta_calc = np.arcsin(np.clip(2 * (q0_s * q2_s - q3_s * q1_s), -1.0, 1.0))
            psi_calc = np.arctan2(2 * (q0_s * q3_s + q1_s * q2_s), 1 - 2 * (q2_s**2 + q3_s**2))
            
            self.phi_rad = np.where(np.isnan(self.phi_rad), phi_calc, self.phi_rad)
            self.theta_rad = np.where(np.isnan(self.theta_rad), theta_calc, self.theta_rad)
            self.psi_rad = np.where(np.isnan(self.psi_rad), psi_calc, self.psi_rad)

        # 2. Frame Transformations (Unconditional generation of DCM for wind mapping)
        phi_safe = np.nan_to_num(self.phi_rad, nan=0.0)
        theta_safe = np.nan_to_num(self.theta_rad, nan=0.0)
        psi_safe = np.nan_to_num(self.psi_rad, nan=0.0)

        c_phi, s_phi = np.cos(phi_safe), np.sin(phi_safe)
        c_theta, s_theta = np.cos(theta_safe), np.sin(theta_safe)
        c_psi, s_psi = np.cos(psi_safe), np.sin(psi_safe)

        # Direction Cosine Matrix (Body -> Nav) components
        R11 = c_theta * c_psi
        R12 = s_phi * s_theta * c_psi - c_phi * s_psi
        R13 = c_phi * s_theta * c_psi + s_phi * s_psi
        R21 = c_theta * s_psi
        R22 = s_phi * s_theta * s_psi + c_phi * c_psi
        R23 = c_phi * s_theta * s_psi - s_phi * c_psi
        R31 = -s_theta
        R32 = s_phi * c_theta
        R33 = c_phi * c_theta
        
        has_nav_vel = not (np.isnan(self.u_n_mps).all() and np.isnan(self.v_n_mps).all() and np.isnan(self.w_n_mps).all())
        has_body_vel = not (np.isnan(self.u_b_mps).all() and np.isnan(self.v_b_mps).all() and np.isnan(self.w_b_mps).all())
        
        # Calculate Body Velocities (Nav -> Body via DCM Transpose)
        if has_nav_vel:
            un = np.nan_to_num(self.u_n_mps, nan=0.0)
            vn = np.nan_to_num(self.v_n_mps, nan=0.0)
            wn = np.nan_to_num(self.w_n_mps, nan=0.0)
            
            u_b_calc = R11 * un + R21 * vn + R31 * wn
            v_b_calc = R12 * un + R22 * vn + R32 * wn
            w_b_calc = R13 * un + R23 * vn + R33 * wn
            
            self.u_b_mps = np.where(np.isnan(self.u_b_mps), u_b_calc, self.u_b_mps)
            self.v_b_mps = np.where(np.isnan(self.v_b_mps), v_b_calc, self.v_b_mps)
            self.w_b_mps = np.where(np.isnan(self.w_b_mps), w_b_calc, self.w_b_mps)
        
        # Calculate Nav Velocities (Body -> Nav via DCM)
        if has_body_vel:
            ub = np.nan_to_num(self.u_b_mps, nan=0.0)
            vb = np.nan_to_num(self.v_b_mps, nan=0.0)
            wb = np.nan_to_num(self.w_b_mps, nan=0.0)
            
            u_n_calc = R11 * ub + R12 * vb + R13 * wb
            v_n_calc = R21 * ub + R22 * vb + R23 * wb
            w_n_calc = R31 * ub + R32 * vb + R33 * wb
            
            self.u_n_mps = np.where(np.isnan(self.u_n_mps), u_n_calc, self.u_n_mps)
            self.v_n_mps = np.where(np.isnan(self.v_n_mps), v_n_calc, self.v_n_mps)
            self.w_n_mps = np.where(np.isnan(self.w_n_mps), w_n_calc, self.w_n_mps)
        
        # 3. Aerodynamic Data Reconstruction
        ub_safe = np.nan_to_num(self.u_b_mps, nan=0.0)
        vb_safe = np.nan_to_num(self.v_b_mps, nan=0.0)
        wb_safe = np.nan_to_num(self.w_b_mps, nan=0.0)

        # Safeguard wind inputs
        wn_safe = np.nan_to_num(self.W_N_mps, nan=0.0) if self.W_N_mps is not None else np.zeros_like(ub_safe)
        we_safe = np.nan_to_num(self.W_E_mps, nan=0.0) if self.W_E_mps is not None else np.zeros_like(ub_safe)
        wd_safe = np.nan_to_num(self.W_D_mps, nan=0.0) if self.W_D_mps is not None else np.zeros_like(ub_safe)

        # Transform Wind from NED to Body Frame via C_n2b
        W_u = R11 * wn_safe + R21 * we_safe + R31 * wd_safe
        W_v = R12 * wn_safe + R22 * we_safe + R32 * wd_safe
        W_w = R13 * wn_safe + R23 * we_safe + R33 * wd_safe

        # Isolate Air-Relative Velocities
        u_air = ub_safe - W_u
        v_air = vb_safe - W_v
        w_air = wb_safe - W_w
        
        vt_calc = np.sqrt(u_air**2 + v_air**2 + w_air**2)
        
        if has_body_vel or has_nav_vel:
            self.true_airspeed_mps = np.where(np.isnan(self.true_airspeed_mps), vt_calc, self.true_airspeed_mps)
            
            vt_safe = np.where(vt_calc == 0, 1e-9, vt_calc)
            
            alpha_calc = np.arctan2(w_air, u_air)
            self.alpha_rad = np.where(np.isnan(self.alpha_rad), alpha_calc, self.alpha_rad)
            
            beta_calc = np.arcsin(np.clip(v_air / vt_safe, -1.0, 1.0))
            self.beta_rad = np.where(np.isnan(self.beta_rad), beta_calc, self.beta_rad)
            
            if not np.isnan(self.cs_mps).all():
                cs_safe = np.where(np.isnan(self.cs_mps) | (self.cs_mps == 0), 1e-9, self.cs_mps)
                mach_calc = self.true_airspeed_mps / cs_safe
                self.mach = np.where(np.isnan(self.mach), mach_calc, self.mach)

        # Safe evaluation: Returns None if the attribute was completely stripped or missing
        vehicle_obj = getattr(self, 'vehicle', None)
        if vehicle_obj is not None:
            print(vehicle_obj)
            has_load_factors = not (np.isnan(self.n_x).all() and np.isnan(self.n_y).all() and np.isnan(self.n_z).all())
            if not has_load_factors:
                vehicle_weight = (self.vehicle.m_dry_kg + self.m_fuel_kg) * self.g_mag_mps2
                self.n_x = self.Fx_b_kgmps2 / vehicle_weight
                self.n_y = self.Fy_b_kgmps2 / vehicle_weight
                self.n_z = self.Fz_b_kgmps2 / vehicle_weight
    
    @property
    def lat_deg(self) -> np.ndarray:
        return self.lat_rad * R2D
    
    @property
    def long_deg(self) -> np.ndarray:
        return self.long_rad * R2D
    
    def save_npz(self, save_path: str):
        """
        Saves the simulation data to an .npz archive.
        Safely extracts metadata from complex objects to prevent pickling errors.
        Handles cases where complex models are None (e.g., when loaded from a file).
        """
        # Package core metadata safely
        meta = {
            'job_name': self.job_name,
            'description': self.description,
            'integrator': self.integrator,
            'vehicle_name': getattr(self.vehicle, 'vehicle_name', str(self.vehicle)) if self.vehicle else "None",
        }

        # Safe extraction for Earth Model
        if self.earth_model is not None:
            meta.update({
                'earth_a': self.earth_model.a,
                'earth_b': self.earth_model.b,
                'earth_omega_rps': self.earth_model.omega_rps,
                'earth_mu': self.earth_model.mu,
                'earth_j2': self.earth_model.j2,
                'earth_g0': self.earth_model.g0,
                'earth_gravity_type': int(self.earth_model.gravity_type),
                'earth_e_sq': self.earth_model.e_sq,
                'earth_e': self.earth_model.e,
            })
            
        # Safe extraction for Wind Model
        if self.wind_model is not None:
            meta.update({
                'wind_type': int(self.wind_model.wind_type),
                'wind_dir_rad': self.wind_model.dir_rad,
                'wind_offset_m': self.wind_model.offset_m,
                'wind_slope_mps': self.wind_model.slope_mps,
            })

        # Filter the dataclass exclusively for native serializable types and numpy arrays
        save_data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                save_data[key] = value
            elif isinstance(value, (int, float, str, bool)):
                save_data[key] = value

        # Execute save, enforcing the same structure you used previously
        np.savez(save_path, **save_data, meta=meta)
    
    def save_csv(self, save_path: str, dt: float = 0.1):
        """
        Re-interpolates the time-history arrays to a fixed time step and saves to a CSV file.
        """
        # Validate time vector
        if getattr(self, 't_s', None) is None or len(self.t_s) < 2:
            print("Insufficient time data to execute interpolation.")
            return

        # Generate the fixed-step time vector
        new_t = np.arange(self.t_s[0], self.t_s[-1], dt)
        
        # Initialize the output dictionary with the new time column
        csv_data = {'t_s': new_t}
        n_original = len(self.t_s)

        # Iterate through dataclass properties
        for key, value in self.__dict__.items():
            if key == 't_s':
                continue
            
            # Exclude metadata; target only arrays matching the time history dimension
            if isinstance(value, np.ndarray) and len(value) == n_original:
                # Linear interpolation for the new time steps
                csv_data[key] = np.interp(new_t, self.t_s, value)

        # Construct DataFrame and export
        df = pd.DataFrame(csv_data)
        df.to_csv(save_path, index=False)
    
    @classmethod
    def from_npz(cls, load_path: str) -> 'SimData':
        """
        Reconstructs a SimData instance from an .npz archive.
        Complex objects (Vehicle, EarthModel, WindModel) are omitted as they 
        cannot be natively serialized into the npz without pickling.
        """
        data = np.load(load_path, allow_pickle=True)
        kwargs = {key: data[key] for key in data.files if key != 'meta'}
        
        if 'meta' in data.files:
            meta = data['meta'].item()
            kwargs['job_name'] = meta.get('job_name', 'Loaded_Job')
            kwargs['description'] = meta.get('description', '')
            kwargs['integrator'] = meta.get('integrator', '')
            
        # Ensure complex models default to None to satisfy initialization
        kwargs['vehicle'] = None
        kwargs['earth_model'] = None
        kwargs['wind_model'] = None
        
        return cls(**kwargs)

    @classmethod
    def from_csv(cls, load_path: str) -> 'SimData':
        """
        Reconstructs a SimData instance from a .csv file.
        Fills missing np.ndarray fields with NaN arrays to maintain matrix shapes.
        """
        df = pd.read_csv(load_path)
        kwargs = {col: df[col].to_numpy() for col in df.columns}
        
        kwargs['job_name'] = 'CSV_Import'
        kwargs['description'] = f'Imported from {load_path}'
        kwargs['integrator'] = 'Unknown'
        kwargs['vehicle'] = None
        kwargs['earth_model'] = None
        kwargs['wind_model'] = None

        # Introspect the dataclass signature to fill missing required arrays with NaNs
        sig = inspect.signature(cls)
        n_rows = len(df)
        
        for param_name, param in sig.parameters.items():
            if param_name not in kwargs:
                if param.annotation == np.ndarray:
                    kwargs[param_name] = np.full(n_rows, np.nan)
                elif param.default == inspect.Parameter.empty:
                    kwargs[param_name] = None
                    
        return cls(**kwargs)