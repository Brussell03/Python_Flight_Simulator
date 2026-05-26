import math

import yaml
import ussa1976
from models.X15.X15 import X15
from src.utils.interpolators import fastInterp1
from src.utils.constants import D2R

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
    
    ic = config['initial_conditions']
    
    h0_m  = ic['h_m']

    # Build Atmosphere Model (amod)
    atmosphere = ussa1976.compute()
    alt_m = atmosphere["z"].values
    rho_kgpm3 = atmosphere["rho"].values
    c_mps = atmosphere["cs"].values
    c0_mps = fastInterp1(alt_m, c_mps, h0_m)
    
    # Trig operations on initial angle of attack and sideslip
    s_alpha   =    math.sin(ic['alpha_deg']*D2R)
    c_alpha   =    math.cos(ic['alpha_deg']*D2R)
    s_beta    =    math.sin(ic['beta_deg']*D2R)
    c_beta    =    math.cos(ic['beta_deg']*D2R)
    
    u0_bf_mps  =   c_alpha*c_beta*ic['Mach']*c0_mps
    v0_bf_mps  =   s_beta*ic['Mach']*c0_mps
    w0_bf_mps  =   s_alpha*c_beta*ic['Mach']*c0_mps
    
    p0_bf_rps  =   ic['p_rps']
    q0_bf_rps  =   ic['q_rps']
    r0_bf_rps  =   ic['r_rps']
    
    phi0_rad   =   ic['phi_deg'] * D2R
    theta0_rad =   ic['theta_deg'] * D2R
    psi0_rad   =   ic['psi_deg'] * D2R
    
    q0_0       =   math.cos(psi0_rad/2)*math.cos(theta0_rad/2)*math.cos(phi0_rad/2) + math.sin(psi0_rad/2)*math.sin(theta0_rad/2)*math.sin(phi0_rad/2)
    q1_0       =   math.cos(psi0_rad/2)*math.cos(theta0_rad/2)*math.sin(phi0_rad/2) - math.sin(psi0_rad/2)*math.sin(theta0_rad/2)*math.cos(phi0_rad/2)
    q2_0       =   math.cos(psi0_rad/2)*math.sin(theta0_rad/2)*math.cos(phi0_rad/2) + math.sin(psi0_rad/2)*math.cos(theta0_rad/2)*math.sin(phi0_rad/2)
    q3_0       =   math.sin(psi0_rad/2)*math.cos(theta0_rad/2)*math.cos(phi0_rad/2) - math.cos(psi0_rad/2)*math.sin(theta0_rad/2)*math.sin(phi0_rad/2)
    
    lat0_rad   =   ic['lat_deg'] * D2R
    long0_rad  =   ic['long_deg'] * D2R
    
    dela_ach_deg = ic['dela_ach_deg'] * D2R
    dele_ach_deg = ic['dele_ach_deg'] * D2R
    delr_ach_deg = ic['delr_ach_deg'] * D2R
    
    m_fuel_kg = ic['m_fuel_kg']
    
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
    
    # Note: Velocities are typically derived dynamically from Mach/Alpha/Beta in the trim solver, 
    # but we pass the raw ICs to the solver to handle.
    x_guess = [
        ic['Mach'], ic['alpha_deg'] * D2R, ic['beta_deg'] * D2R,
        0.0, 0.0, 0.0, # Rates
        ic['phi_deg'] * D2R, ic['theta_deg'] * D2R, ic['psi_deg'] * D2R,
        ic['lat_deg'] * D2R, ic['long_deg'] * D2R, ic['h_m'],
        ic['dela_ach_deg'] * D2R, ic['dele_ach_deg'] * D2R, ic['delr_ach_deg'] * D2R, 
        ic['m_fuel_kg']
    ]
    
    u_guess = [0.0, 0.0, 0.0, config['control']['throttle_percent']]
    
    analysis = config['analysis']
    analysis['trim_flag'] = analysis.get('trim_flag', 'off') # Defaults to 'off' if missing
    analysis['linearization_flag'] = analysis.get('linearization_flag', 'off')

    return vehicle, amod, config['control'], config['simulation'], ic, x0, x_guess, u_guess, analysis, config