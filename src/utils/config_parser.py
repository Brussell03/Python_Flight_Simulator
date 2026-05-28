import math

import yaml
import ussa1976
from models.X15.X15 import X15
from src.utils.interpolators import fastInterp1
from src.utils.constants import D2R, FT2M

def load_simulation_config(yaml_path):
    """
    Parses the YAML config and returns the required simulation objects.
    """
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Instantiate Vehicle Model Factory
    if config['vehicle']['model'] == 'X15':
        vehicle = X15()
    else:
        raise ValueError(f"Unknown vehicle model: {config['vehicle']['model']}")
    
    meta_cfg = config.get('meta', {})
    instruction_cfg = config.get('instructions', {})
    output_cfg = config.get('output', {})
    init_cond_cfg = config.get('initial_conditions', {})
    trim_cfg = config.get('trim', {})
    control_cfg = config.get('control', {})
    
    h0_m  = init_cond_cfg['h_m'] if init_cond_cfg.get('h_m') is not None else init_cond_cfg['h_ft'] * FT2M

    # Build Atmosphere Model (amod)
    atmosphere = ussa1976.compute()
    alt_m = atmosphere["z"].values
    rho_kgpm3 = atmosphere["rho"].values
    c_mps = atmosphere["cs"].values
    c0_mps = fastInterp1(alt_m, c_mps, h0_m)
    
    alpha_rad  = init_cond_cfg.get('alpha_deg') * D2R if init_cond_cfg.get('alpha_deg') is not None else init_cond_cfg.get('alpha_rad', 0.0)
    beta_rad   = init_cond_cfg.get('beta_deg') * D2R if init_cond_cfg.get('beta_deg') is not None else init_cond_cfg.get('beta_rad', 0.0)
    
    # Trig operations on initial angle of attack and sideslip
    s_alpha    =    math.sin(init_cond_cfg.get('alpha_deg', 0)*D2R)
    c_alpha    =    math.cos(init_cond_cfg.get('alpha_deg', 0)*D2R)
    s_beta     =    math.sin(init_cond_cfg.get('beta_deg', 0)*D2R)
    c_beta     =    math.cos(init_cond_cfg.get('beta_deg', 0)*D2R)
    
    u0_bf_mps  =   c_alpha*c_beta*init_cond_cfg['Mach']*c0_mps
    v0_bf_mps  =   s_beta*init_cond_cfg['Mach']*c0_mps
    w0_bf_mps  =   s_alpha*c_beta*init_cond_cfg['Mach']*c0_mps
    
    # Angular rates: priority to dps, then rps, then default to 0.0
    p0_bf_rps  = init_cond_cfg.get('p_dps') * D2R if init_cond_cfg.get('p_dps') is not None else init_cond_cfg.get('p_rps', 0.0)
    q0_bf_rps  = init_cond_cfg.get('q_dps') * D2R if init_cond_cfg.get('q_dps') is not None else init_cond_cfg.get('q_rps', 0.0)
    r0_bf_rps  = init_cond_cfg.get('r_dps') * D2R if init_cond_cfg.get('r_dps') is not None else init_cond_cfg.get('r_rps', 0.0)
    
    # Attitude angles: priority to deg, then rad, then default to 0.0
    phi0_rad   = init_cond_cfg.get('phi_deg') * D2R if init_cond_cfg.get('phi_deg') is not None else init_cond_cfg.get('phi_rad', 0.0)
    theta0_rad = init_cond_cfg.get('theta_deg') * D2R if init_cond_cfg.get('theta_deg') is not None else init_cond_cfg.get('theta_rad', 0.0)
    psi0_rad   = init_cond_cfg.get('psi_deg') * D2R if init_cond_cfg.get('psi_deg') is not None else init_cond_cfg.get('psi_rad', 0.0)
    
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
        "c_mps": c_mps
    }
    
    # State Vector Trim Guess [u, v, w, p, q, r, q0, q1, q2, q3, lat, long, h, dela, dele, delr, m_fuel]
    x0 = [
        u0_bf_mps, v0_bf_mps, w0_bf_mps,
        p0_bf_rps, q0_bf_rps, r0_bf_rps,
        q0_0, q1_0, q2_0, q3_0,
        lat0_rad, long0_rad, h0_m,
        dela_ach_deg, dele_ach_deg, delr_ach_deg, m_fuel_kg
    ]

    return vehicle, amod, meta_cfg, instruction_cfg, output_cfg, trim_cfg, control_cfg, x0