import argparse
import dataclasses
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from src.engine.sim_data import SimData
from src.utils.interpolators import fastInterp1
from src.utils.constants import D2R, FT2M, R2D

# Map column headers directly to SimData attribute names
COLUMN_MAP = {
    'u_mps_vs_time_s': 'u_b_mps', 'v_mps_vs_time_s': 'v_b_mps', 'w_mps_vs_time_s': 'w_b_mps',
    'p_rps_vs_time_s': 'p_b_rps', 'q_rps_vs_time_s': 'q_b_rps', 'r_rps_vs_time_s': 'r_b_rps',
    'q0_vs_time_s': 'q0', 'q1_vs_time_s': 'q1', 'q2_vs_time_s': 'q2', 'q3_vs_time_s': 'q3',
    'lat_rad_vs_time_s': 'lat_rad', 'long_rad_vs_time_s': 'long_rad', 'altitude_ft_vs_time_s': 'h_m',
    'dela_deg_vs_time_s': 'dela_ach_deg', 'dele_deg_vs_time_s': 'dele_ach_deg', 'delr_deg_vs_time_s': 'delr_ach_deg',
    'm_fuel_kg_vs_time_s': 'm_fuel_kg',
    'Mach_vs_time_s': 'mach',
    'alpha_deg_vs_time_s': 'alpha_rad', 'beta_deg_vs_time_s': 'beta_rad',
    'True_Airspeed_mps_vs_time_s': 'true_airspeed_mps',
    'roll_rad_vs_time_s': 'phi_rad', 'phi_rad_vs_time_s': 'phi_rad',
    'pitch_rad_vs_time_s': 'theta_rad', 'theta_rad_vs_time_s': 'theta_rad',
    'yaw_rad_vs_time_s': 'psi_rad', 'psi_rad_vs_time_s': 'psi_rad',
    'u_n_mps_vs_time_s': 'u_n_mps', 'v_n_mps_vs_time_s': 'v_n_mps', 'w_n_mps_vs_time_s': 'w_n_mps',
    'delt_percent_vs_time_s': 'delt_percent',
    
    'feVelocity_ft_s_X': 'u_n_mps',
    'feVelocity_ft_s_Y': 'v_n_mps',
    'feVelocity_ft_s_Z': 'w_n_mps',
    'altitudeMsl_ft': 'h_m',
    'longitude_deg': 'long_rad',
    'latitude_deg': 'lat_rad',
    'localGravity_ft_s2': 'g_mag_mps2',
    'eulerAngle_deg_Yaw': 'psi_rad',
    'eulerAngle_deg_Pitch': 'theta_rad',
    'eulerAngle_deg_Roll': 'phi_rad',
    'bodyAngularRateWrtEi_deg_s_Roll': 'p_b_rps',
    'bodyAngularRateWrtEi_deg_s_Pitch': 'q_b_rps',
    'bodyAngularRateWrtEi_deg_s_Yaw': 'r_b_rps',
    'speedOfSound_ft_s': 'cs_mps',
    'airDensity_slug_ft3': 'rho_kgpm3',
    'ambientPressure_lbf_ft2': 'p_kgpms2',
    'ambientTemperature_dgR': 'T_K',
    'aero_bodyForce_lbf_X': 'Fx_b_kgmps2',
    'aero_bodyForce_lbf_Y': 'Fy_b_kgmps2',
    'aero_bodyForce_lbf_Z': 'Fz_b_kgmps2',
    'aero_bodyMoment_ftlbf_L': 'l_b_kgm2ps2',
    'aero_bodyMoment_ftlbf_M': 'm_b_kgm2ps2',
    'aero_bodyMoment_ftlbf_N': 'n_b_kgm2ps2',
    'mach': 'mach',
    'trueAirspeed_nmi_h': 'true_airspeed_mps'
}

def load_csv(file_path):
    """Loads CSV and ensures it is sorted by the independent variable (time)."""
    df = pd.read_csv(file_path, header=0)
    df = df.sort_values(by=df.columns[0])
    return df

def get_unit_conversion(col_name, attr_name):
    """
    Determines the necessary scaling factor to convert the input data
    to the baseline SI units expected by the sim_data structure.
    """
    col_lower = col_name.lower()
    factor = 1.0
    conversion_msg = ""
    
    # 1. Complex & Imperial conversions
    if 'ft_s2' in col_lower: 
        factor = 0.3048
        conversion_msg = "(ft/s^2 -> m/s^2)"
    elif 'ft_s' in col_lower: 
        factor = 0.3048
        conversion_msg = "(ft/s -> m/s)"
    elif 'ft3' in col_lower and 'slug' in col_lower: 
        factor = 515.378818
        conversion_msg = "(slug/ft^3 -> kg/m^3)"
    elif 'lbf_ft2' in col_lower: 
        factor = 47.880258
        conversion_msg = "(lbf/ft^2 -> N/m^2)"
    elif 'ftlbf' in col_lower: 
        factor = 1.355818
        conversion_msg = "(ft-lbf -> N-m)"
    elif 'lbf' in col_lower: 
        factor = 4.448222
        conversion_msg = "(lbf -> N)"
    elif 'nmi_h' in col_lower: 
        factor = 0.514444
        conversion_msg = "(knots -> m/s)"
    elif 'dgr' in col_lower: 
        factor = 5.0 / 9.0
        conversion_msg = "(deg R -> K)"
    elif '_ft' in col_lower or 'ft_' in col_lower: 
        factor = 0.3048
        conversion_msg = "(ft -> m)"
        
    # 2. Angle conversions
    rad_attrs = {'p_b_rps', 'q_b_rps', 'r_b_rps', 'lat_rad', 'long_rad',
                 'alpha_rad', 'beta_rad', 'phi_rad', 'theta_rad', 'psi_rad',
                 'p_nb_rps', 'q_nb_rps', 'r_nb_rps', 'phi_dot_rps', 'theta_dot_rps', 'psi_dot_rps'}
    
    if attr_name in rad_attrs and ('_deg' in col_lower or 'deg_' in col_lower):
        factor *= D2R
        conversion_msg += " (deg -> rad)"
        
    deg_attrs = {'dela_ach_deg', 'dele_ach_deg', 'delr_ach_deg', 
                 'dela_cmd_deg', 'dele_cmd_deg', 'delr_cmd_deg'}
                 
    if attr_name in deg_attrs and ('_rad' in col_lower or 'rad_' in col_lower):
        factor *= R2D
        conversion_msg += " (rad -> deg)"
        
    return factor, conversion_msg.strip()

def parse_csvs(file_paths, dt=0.01, wind_model=None):
    # Normalize input: if a single Path or string is passed, wrap it in a list
    if isinstance(file_paths, (Path, str)):
        file_paths = [Path(file_paths)]
    
    datasets = []
    global_t_min = float('inf')
    global_t_max = float('-inf')
    
    print("--- Loading and Parsing CSVs ---")
    for file_path in file_paths:
        df = load_csv(file_path)
        
        if len(df.columns) < 2:
            print(f"Warning: '{os.path.basename(file_path)}' lacks sufficient columns. Skipping.")
            continue
            
        time_col = df.columns[0]
        time_sample = df[time_col].values
        
        global_t_min = min(global_t_min, time_sample[0])
        global_t_max = max(global_t_max, time_sample[-1])
        
        # Iterate over all data columns rather than assuming a 2-column layout
        for col_name in df.columns[1:]:
            col_name_clean = col_name.strip()
            attr_name = COLUMN_MAP.get(col_name_clean)
            
            if attr_name is None:
                continue
            
            factor, msg = get_unit_conversion(col_name_clean, attr_name)
            data_sample = df[col_name_clean].values * factor
            
            datasets.append({
                'name': os.path.basename(file_path),
                'col_name': col_name_clean,
                'attr_name': attr_name,
                't': time_sample,
                'data': data_sample,
                'msg': msg
            })

    if not datasets:
        print("No valid mapping datasets found in the provided files. Exiting.")
        sys.exit(1)

    t_start = global_t_min
    t_end = global_t_max
    t_common = np.arange(t_start, t_end + dt, dt)
    n_time_bps = len(t_common)
    
    # Initialize full arrays with NaN to prevent casting bugs
    # Explicitly target only numpy array fields for NaN initialization
    sim_data_kwargs = {}
    for field in dataclasses.fields(SimData):
        if field.type is np.ndarray:
            sim_data_kwargs[field.name] = np.full(n_time_bps, np.nan)
    sim_data_kwargs['t_s'] = t_common
    
    # print(f"\n--- Interpolating Data to Master Time Vector (dt={dt}s) ---")
    
    for ds in datasets:
        # print(f"Mapping: {ds['col_name']} -> {ds['attr_name']} {ds['msg']}")
        
        interpolated_data = np.zeros(n_time_bps)
        time_sample = ds['t']
        data_sample = ds['data']
        
        interpolated_data = np.interp(t_common, time_sample, data_sample)
        
        sim_data_kwargs[ds['attr_name']] = interpolated_data

    if wind_model is not None:
        vec_get_velocity = np.vectorize(wind_model.get_velocity, otypes=[float, float, float])
        
        w_n, w_e, w_d = vec_get_velocity(sim_data_kwargs.get('h_m'))
        
        sim_data_kwargs['W_N_mps'] = w_n
        sim_data_kwargs['W_E_mps'] = w_e
        sim_data_kwargs['W_D_mps'] = w_d

    return SimData(**sim_data_kwargs)

def parse_all_csvs(input_path, dt=0.01, wind_model=None):
    path = Path(input_path)

    # Determine if input is a single file or a directory
    if path.is_file():
        # Validate that the file is actually a .csv
        if path.suffix != '.csv':
            raise ValueError(f"The provided file '{path.name}' is not a .csv file.")
        file_paths = [path]
    elif path.is_dir():
        # Glob only .csv files if it's a directory
        file_paths = list(path.glob('*.csv'))
    else:
        raise FileNotFoundError(f"The path '{input_path}' does not exist.")
    
    # parse all csvs
    sim_datas = []
    file_names = []
    for file_path in file_paths:
        # print(file_path)
        sim_datas.append(parse_csvs(file_path, dt, wind_model))
        file_names.append(file_path.stem)
    
    return sim_datas, file_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse multi-column CSVs into a sim_data .npz file.")
    
    parser.add_argument("files", nargs="+", help="List of .csv files to combine")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step for the interpolated master array (default: 0.01)")
    parser.add_argument("--filename", type=str, default="Combined_Data", help="Name of the output job/file")
    parser.add_argument("--dataname", type=str, default="Flight Test", help="Name of the output data")
    
    args = parser.parse_args()
    sim_data_obj = parse_csvs(args.files, dt=args.dt)

    out_dir = "./combined_data/"
    os.makedirs(out_dir, exist_ok=True)
    
    output_name = args.filename if args.filename != "" else "Combined_Data"
    data_name = args.dataname if args.dataname != "" else "Flight Test"
    
    save_path = os.path.join(out_dir, f"{output_name}.npz")
    meta = {'job_name': output_name, 'data_name': data_name}
    
    np.savez(save_path, **dataclasses.asdict(sim_data_obj), meta=meta)
    
    print(f"\n--- Parsing Complete ---")
    print(f"Data saved to: {save_path}")