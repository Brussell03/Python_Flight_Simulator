import math
import os
import numpy as np
import yaml
import ussa1976

from src.engine.eom_solver import eom_solver
from src.environment.earth_model import EarthModel
from models.bricks.bricks import NASAAtmos02Brick, NASAAtmos03Brick
from models.spheres.spheres import Blueberry, BowlingBall, Carronade12lb, Musketball50cal, NASAAtmos01Sphere, TsarCannonball
from models.X15.X15 import X15
from models.spheres.DMLCannonball import DMLCannonball
from src.utils.interpolators import fastInterp1
from src.utils.constants import D2R, FT2M
from src.utils.kinematics import dcm_to_quat, ecef_to_ned_dcm, quat_body_to_nav

def resolve_path(base_dir, path):
    # Join only if path is relative
    if path and not os.path.isabs(path):
        return os.path.join(base_dir, path)
    return path

def get_unified_value(config_dict, base_name, to_si_factor):
    """
    Retrieves a configuration value and converts it to SI units.
    Raises ValueError if both _deg/_rad or _fps/_mps variants are provided.
    """
    deg_variant = config_dict.get(f"{base_name}_deg")
    rad_variant = config_dict.get(f"{base_name}_rad")
    
    # Handle velocity variants if used for V instead of angles
    fps_variant = config_dict.get(f"{base_name}_fps")
    mps_variant = config_dict.get(f"{base_name}_mps")

    # Check for angle conflicts
    if deg_variant is not None and rad_variant is not None:
        raise ValueError(f"Ambiguous configuration: Cannot specify both '{base_name}_deg' and '{base_name}_rad'.")
    # Check for velocity conflicts
    if fps_variant is not None and mps_variant is not None:
        raise ValueError(f"Ambiguous configuration: Cannot specify both '{base_name}_fps' and '{base_name}_mps'.")

    # Return the valid parsed value converted to SI
    if deg_variant is not None: return deg_variant * to_si_factor
    if rad_variant is not None: return rad_variant
    if fps_variant is not None: return fps_variant * to_si_factor
    if mps_variant is not None: return mps_variant
    
    return None

def flight_path_angle_check(phi0_rad, theta0_rad, psi0_rad, u0_bf_mps, v0_bf_mps, w0_bf_mps, V_n_mps, gamma_rad, beta_rad):
    # Construct Euler-based DCM from Body to Nav (C_b2n) to transform velocities
    # Using your standard 3-2-1 rotation sequence matching your quaternion logic
    sph, cph = math.sin(phi0_rad), math.cos(phi0_rad)
    sth, cth = math.sin(theta0_rad), math.cos(theta0_rad)
    sps, cps = math.sin(psi0_rad), math.cos(psi0_rad)

    C_b2n = np.array([
        [cps*cth, cps*sth*sph - sps*cph, cps*sth*cph + sps*sph],
        [sps*cth, sps*sth*sph + cps*cph, sps*sth*cph - cps*sph],
        [-sth,    cth*sph,               cth*cph]
    ])
    
    # Transform body velocities into NED frame
    vel_bf = np.array([u0_bf_mps, v0_bf_mps, w0_bf_mps])
    vel_ned = C_b2n @ vel_bf
    v_down = vel_ned[2]

    # Calculate actual kinematic gamma from resolved state
    # Flight path angle mathematically is -asin(V_down / V_total)
    gamma_kinematic = -math.asin(v_down / V_n_mps)

    # Verify that the calculated kinematic path matches our derived gamma_rad
    if not math.isclose(gamma_rad, gamma_kinematic, abs_tol=1e-4):
        raise ValueError(
            f"Kinematic mismatch: The combination of roll ({math.degrees(phi0_rad):.1f}°), "
            f"sideslip ({math.degrees(beta_rad):.1f}°), and pitch breaking out of the 2D plane "
            f"yields a true kinematic flight path angle of {math.degrees(gamma_kinematic):.2f}°, "
            f"which contradicts the configured/derived target gamma of {math.degrees(gamma_rad):.2f}°."
        )

def load_simulation_config(yaml_path):
    """
    Parses the YAML config and returns the required simulation objects.
    """
    # Establish base directory from the YAML file location
    base_dir = os.path.dirname(os.path.abspath(yaml_path))
    
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    meta_cfg = config.get('meta', {})
    instruction_cfg = config.get('instructions', {})
    output_cfg = config.get('output', {})
    compare_cfg = config.get('compare', {})
    init_cond_cfg = config.get('initial_conditions', {})
    trim_cfg = config.get('trim', {})
    control_cfg = config.get('control', {})
    
    # Resolve Path for Control Input
    th_path = None
    if control_cfg.get('type') == 'time_history':
        raw_path = control_cfg.get('input_file')
        th_path = resolve_path(base_dir, raw_path)

    # Instantiate Vehicle Model Factory
    if config['vehicle']['model'] == 'X15':
        vehicle = X15(time_history_path=th_path)
    elif config['vehicle']['model'] == 'Musketball50cal':
        vehicle = Musketball50cal()
    elif config['vehicle']['model'] == 'Carronade12lb':
        vehicle = Carronade12lb()
    elif config['vehicle']['model'] == 'Blueberry':
        vehicle = Blueberry()
    elif config['vehicle']['model'] == 'BowlingBall':
        vehicle = BowlingBall()
    elif config['vehicle']['model'] == 'TsarCannonball':
        vehicle = TsarCannonball()
    elif config['vehicle']['model'] == 'NASAAtmos01Sphere':
        vehicle = NASAAtmos01Sphere()
    elif config['vehicle']['model'] == 'NASAAtmos02Brick':
        vehicle = NASAAtmos02Brick()
    elif config['vehicle']['model'] == 'NASAAtmos03Brick':
        vehicle = NASAAtmos03Brick()
    elif config['vehicle']['model'] == 'X15':
        vehicle = X15(time_history_path=th_path)
    elif config['vehicle']['model'] == 'DMLCannonball':
        # Ensure the DML files exist relative to your execution directory
        aero_path = resolve_path(base_dir, "models/spheres/cannonball_aero.dml")
        inertia_path = resolve_path(base_dir, "models/spheres/cannonball_inertia.dml")
        vehicle = DMLCannonball(aero_dml_path=aero_path, inertia_dml_path=inertia_path)
    else:
        raise ValueError(f"Unknown vehicle model: {config['vehicle']['model']}")
    
    if init_cond_cfg.get('h_m') is not None and init_cond_cfg.get('h_ft') is not None:
        raise ValueError("Ambiguous configuration: Cannot specify both 'h_m' and 'h_ft'.")
    h0_m  = init_cond_cfg['h_m'] if init_cond_cfg.get('h_m') is not None else init_cond_cfg['h_ft'] * FT2M

    # Build Atmosphere Model (amod)
    atmosphere = ussa1976.compute()
    alt_m = atmosphere["z"].values
    rho_kgpm3 = atmosphere["rho"].values
    c_mps = atmosphere["cs"].values
    p_Npm2 = atmosphere["p"].values
    T_K = atmosphere["t"].values
    c0_mps = fastInterp1(alt_m, c_mps, h0_m)
    
    # --- Parse Attitude First (Required for NED transformations) ---
    phi0_rad = get_unified_value(init_cond_cfg, 'phi', D2R) or 0.0
    theta0_rad = get_unified_value(init_cond_cfg, 'theta', D2R) or 0.0
    psi0_rad = get_unified_value(init_cond_cfg, 'psi', D2R) or 0.0
    
    # --- Resolve Kinematic Pitch Plane Angles (alpha, gamma) ---
    # We still need alpha/gamma for the consistency checks
    alpha_cfg = get_unified_value(init_cond_cfg, 'alpha', D2R)
    gamma_cfg = get_unified_value(init_cond_cfg, 'gamma', D2R)
    beta_rad  = get_unified_value(init_cond_cfg, 'beta', D2R) or 0.0
    
    # --- Velocity Mode Detection & Resolution ---
    # Define potential keys for each mode
    air_keys = {'V', 'Mach'}
    body_keys = {'u_b', 'v_b', 'w_b'}
    ned_keys = {'u_n', 'v_n', 'w_n'}
    
    # Helper to check if a specific mode is provided
    def has_mode(keys):
        return any(f"{k}_mps" in init_cond_cfg or f"{k}_fps" in init_cond_cfg or k in init_cond_cfg for k in keys)

    # Validate exclusivity
    active_modes = [m for m, keys in [('air', air_keys), ('body', body_keys), ('ned', ned_keys)] if has_mode(keys)]
    if len(active_modes) > 1:
        raise ValueError(f"Ambiguous velocity configuration: Multiple modes provided ({', '.join(active_modes)}).")
    if not active_modes:
        raise ValueError("Velocity configuration missing: Must provide Airspeed, Body, or NED velocities.")

    mode = active_modes[0]
    
    # Resolve to body frame (u, v, w) in m/s
    if mode == 'air':
        # Existing Airspeed logic
        V_n_mps = get_unified_value(init_cond_cfg, 'V', FT2M)
        if V_n_mps is None and 'Mach' in init_cond_cfg:
            V_n_mps = init_cond_cfg['Mach'] * c0_mps
        
        # Calculate derived body velocities
        # Note: requires alpha/beta
        if alpha_cfg is None: raise ValueError("Airspeed mode requires alpha.")
        u0_b_mps = V_n_mps * math.cos(alpha_cfg) * math.cos(beta_rad)
        v0_b_mps = V_n_mps * math.sin(beta_rad)
        w0_b_mps = V_n_mps * math.sin(alpha_cfg) * math.cos(beta_rad)

    elif mode == 'body':
        u0_b_mps = get_unified_value(init_cond_cfg, 'u_b', FT2M) or 0.0
        v0_b_mps = get_unified_value(init_cond_cfg, 'v_b', FT2M) or 0.0
        w0_b_mps = get_unified_value(init_cond_cfg, 'w_b', FT2M) or 0.0
        V_n_mps = math.sqrt(u0_b_mps**2 + v0_b_mps**2 + w0_b_mps**2)

    elif mode == 'ned':
        # Requires Euler angles (parsed above) to build DCM
        u_n = get_unified_value(init_cond_cfg, 'u_n', FT2M) or 0.0
        v_n = get_unified_value(init_cond_cfg, 'v_n', FT2M) or 0.0
        w_n = get_unified_value(init_cond_cfg, 'w_n', FT2M) or 0.0
        
        # Transform NED to Body
        # C_b2n inverse is transpose
        sph, cph = math.sin(phi0_rad), math.cos(phi0_rad)
        sth, cth = math.sin(theta0_rad), math.cos(theta0_rad)
        sps, cps = math.sin(psi0_rad), math.cos(psi0_rad)
        C_b2n = np.array([
            [cps*cth, cps*sth*sph - sps*cph, cps*sth*cph + sps*sph],
            [sps*cth, sps*sth*sph + cps*cph, sps*sth*cph - cps*sph],
            [-sth,    cth*sph,               cth*cph]
        ])
        C_n2b = C_b2n.T
        vel_ned = np.array([u_n, v_n, w_n])
        vel_bf = C_n2b @ vel_ned
        u0_b_mps, v0_b_mps, w0_b_mps = vel_bf
        V_n_mps = math.sqrt(u0_b_mps**2 + v0_b_mps**2 + w0_b_mps**2)

    # Perform consistency check if alpha/gamma provided
    if gamma_cfg is not None and alpha_cfg is not None:
        flight_path_angle_check(phi0_rad, theta0_rad, psi0_rad, u0_b_mps, v0_b_mps, w0_b_mps, V_n_mps, gamma_cfg, beta_rad)
    
    # Angular rates: priority to dps, then rps, then default to 0.0
    p0_b_rps  = init_cond_cfg.get('p_dps') * D2R if init_cond_cfg.get('p_dps') is not None else init_cond_cfg.get('p_rps', 0.0)
    q0_b_rps  = init_cond_cfg.get('q_dps') * D2R if init_cond_cfg.get('q_dps') is not None else init_cond_cfg.get('q_rps', 0.0)
    r0_b_rps  = init_cond_cfg.get('r_dps') * D2R if init_cond_cfg.get('r_dps') is not None else init_cond_cfg.get('r_rps', 0.0)
    
    q0_0       =   math.cos(psi0_rad/2)*math.cos(theta0_rad/2)*math.cos(phi0_rad/2) + math.sin(psi0_rad/2)*math.sin(theta0_rad/2)*math.sin(phi0_rad/2)
    q1_0       =   math.cos(psi0_rad/2)*math.cos(theta0_rad/2)*math.sin(phi0_rad/2) - math.sin(psi0_rad/2)*math.sin(theta0_rad/2)*math.cos(phi0_rad/2)
    q2_0       =   math.cos(psi0_rad/2)*math.sin(theta0_rad/2)*math.cos(phi0_rad/2) + math.sin(psi0_rad/2)*math.cos(theta0_rad/2)*math.sin(phi0_rad/2)
    q3_0       =   math.sin(psi0_rad/2)*math.cos(theta0_rad/2)*math.cos(phi0_rad/2) - math.cos(psi0_rad/2)*math.sin(theta0_rad/2)*math.sin(phi0_rad/2)
    
    lat0_rad   =   init_cond_cfg.get('lat_deg') * D2R if init_cond_cfg.get('lat_deg') is not None else init_cond_cfg.get('lat_rad', 0.0)
    long0_rad  =   init_cond_cfg.get('long_deg') * D2R if init_cond_cfg.get('long_deg') is not None else init_cond_cfg.get('long_rad', 0.0)
    
    dela_ach_deg = init_cond_cfg.get('dela_ach_deg', 0)
    dele_ach_deg = init_cond_cfg.get('dele_ach_deg', 0)
    delr_ach_deg = init_cond_cfg.get('delr_ach_deg', 0)
    
    m_fuel_kg = init_cond_cfg['m_fuel_kg']
    
    amod = {
        "alt_m": alt_m,
        "rho_kgpm3": rho_kgpm3,
        "c_mps": c_mps,
        "p_Npm2" : p_Npm2,
        "T_K": T_K
    }
    
    earth_type = instruction_cfg.get('earth_model', 'WGS84') # 'WGS84', 'Spherical_Rotating', 'Spherical_NonRotating'
    gravity_mapping = {'constant': 0, 'inverse_square': 1, 'J2': 2}
    gravity_type = gravity_mapping.get(instruction_cfg.get('gravity_type', 'J2'), 2)
    
    if earth_type == 'WGS84':
        earth = EarthModel(gravity_type=gravity_type) # Defaults to WGS84 constants
    elif earth_type == 'Spherical_Rotating':
        # Zero out J2, force polar radius = equatorial radius
        earth = EarthModel(b_m=6378137.0, j2=0.0, gravity_type=gravity_type)
    elif earth_type == 'Spherical_NonRotating':
        earth = EarthModel(b_m=6378137.0, omega_rps=0.0, j2=0.0, gravity_type=gravity_type)
    else:
        raise ValueError("Invalid earth model type")
    
    # Instantiate EOM
    eom = eom_solver(earth_model=earth)
    
    # Convert Geodetic (Lat, Lon, Alt) to ECEF (X, Y, Z)
    sin_lat = math.sin(lat0_rad)
    cos_lat = math.cos(lat0_rad)
    sin_lon = math.sin(long0_rad)
    cos_lon = math.cos(long0_rad)
    
    N = earth.a / math.sqrt(1.0 - earth.e**2 * sin_lat**2)
    
    x0_e = (N + h0_m) * cos_lat * cos_lon
    y0_e = (N + h0_m) * cos_lat * sin_lon
    z0_e = (N * (1.0 - earth.e**2) + h0_m) * sin_lat
    
    # Convert Initial Quaternions from Nav-to-Body to ECEF-to-Body
    # quat_body_to_nav returns C_b2n. The transpose is C_n2b.
    C_n2b = quat_body_to_nav(q0_0, q1_0, q2_0, q3_0).T
    C_e2n = ecef_to_ned_dcm(lat0_rad, long0_rad)
    
    C_e2b = C_n2b @ C_e2n
    q0_e, q1_e, q2_e, q3_e = dcm_to_quat(C_e2b)
    
    # State Vector [u, v, w, p, q, r, q0e, q1e, q2e, q3e, X_e, Y_e, Z_e, dela, dele, delr, m_fuel]
    x0 = [
        u0_b_mps, v0_b_mps, w0_b_mps,
        p0_b_rps, q0_b_rps, r0_b_rps,
        q0_e, q1_e, q2_e, q3_e,
        x0_e, y0_e, z0_e,
        dela_ach_deg, dele_ach_deg, delr_ach_deg, m_fuel_kg
    ]

    return eom, vehicle, amod, meta_cfg, instruction_cfg, output_cfg, compare_cfg, trim_cfg, control_cfg, x0, base_dir