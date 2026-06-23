import argparse
from dataclasses import fields
import dataclasses
import os
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
    'lat_rad_vs_time_s': 'lat_rad', 'long_rad_vs_time_s': 'long_rad', 'altitude_ft_vs_time_s': 'h_m', 'h_ft_vs_time_s': 'h_m',
    'dela_deg_vs_time_s': 'dela_ach_rad', 'dele_deg_vs_time_s': 'dele_ach_rad', 'delr_deg_vs_time_s': 'delr_ach_rad',
    'm_fuel_kg_vs_time_s': 'm_fuel_kg',
    'Mach_vs_time_s': 'mach',
    'alpha_deg_vs_time_s': 'alpha_rad', 'beta_deg_vs_time_s': 'beta_rad',
    'True_Airspeed_mps_vs_time_s': 'true_airspeed_mps',
    'roll_rad_vs_time_s': 'phi_rad', 'phi_rad_vs_time_s': 'phi_rad',
    'pitch_rad_vs_time_s': 'theta_rad', 'theta_rad_vs_time_s': 'theta_rad',
    'yaw_rad_vs_time_s': 'psi_rad', 'psi_rad_vs_time_s': 'psi_rad',
    'u_n_mps_vs_time_s': 'u_n_mps', 'v_n_mps_vs_time_s': 'v_n_mps', 'w_n_mps_vs_time_s': 'w_n_mps',
    'delt_percent_vs_time_s': 'delt_percent'
}

def load_csv(file_path):
    """Loads CSV and ensures it is sorted by the independent variable (time)."""
    df = pd.read_csv(file_path, header=0)
    # Assume column 0 is 'x' (time) and column 1 is 'y' (data)
    # Sorting is critical because digitized data points can sometimes loop back slightly
    df = df.sort_values(by=df.columns[0])
    return df

def get_unit_conversion(col_name, attr_name):
    """
    Determines the necessary scaling factor to convert the input data
    to the baseline units expected by the sim_data array.
    """
    col_lower = col_name.lower()
    factor = 1.0
    conversion_msg = ""
    
    # Distance
    if attr_name == 'h_m' and '_ft_' in col_lower:
        factor = FT2M
        conversion_msg = "(Converted ft -> m)"
        
    # Angles
    rad_attrs = {
        'p_b_rps', 'q_b_rps', 'r_b_rps', 'lat_rad', 'long_rad',
        'alpha_rad', 'beta_rad', 'phi_rad', 'theta_rad', 'psi_rad',
        'p_nb_rps', 'q_nb_rps', 'r_nb_rps', 'phi_dot_rps', 'theta_dot_rps', 'psi_dot_rps',
        'dela_ach_rad', 'dele_ach_rad', 'delr_ach_rad'
    }
    
    if attr_name in rad_attrs and '_deg_' in col_lower:
        factor = D2R
        conversion_msg = "(Converted deg -> rad)"
        
    return factor, conversion_msg

def combine_data(file_paths, dt=0.01, output_name="Combined_Data", data_name="Flight Test"):
    
    # 1. Load all valid CSVs and determine global time bounds
    datasets = []
    global_t_min = float('inf')
    global_t_max = float('-inf')
    
    print("--- Loading and Parsing CSVs ---")
    for file_path in file_paths:
        df = load_csv(file_path)
        
        if len(df.columns) < 2:
            print(f"Warning: '{os.path.basename(file_path)}' has fewer than 2 columns. Skipping.")
            continue
            
        col_name = df.columns[1].strip()
        attr_name = COLUMN_MAP.get(col_name)
        
        if attr_name is None:
            print(f"Warning: Column header '{col_name}' in '{os.path.basename(file_path)}' not found in COLUMN_MAP. Skipping.")
            continue
        
        factor, msg = get_unit_conversion(col_name, attr_name)
        
        time_sample = df.iloc[:, 0].values
        data_sample = df.iloc[:, 1].values * factor
        
        global_t_min = min(global_t_min, time_sample[0])
        global_t_max = max(global_t_max, time_sample[-1])
        
        datasets.append({
            'name': os.path.basename(file_path),
            'col_name': col_name,
            'attr_name': attr_name,
            't': time_sample,
            'data': data_sample,
            'msg': msg
        })

    if not datasets:
        print("No valid datasets loaded. Exiting.")
        sys.exit(1)

    # 2. Establish Master Time Vector
    t_start = global_t_min
    t_end = global_t_max
    
    t_common = np.arange(t_start, t_end + dt, dt)
    n_time_bps = len(t_common)
    
    # Explicitly target only numpy array fields for NaN initialization
    sim_data_kwargs = {}
    for field in dataclasses.fields(SimData):
        if field.type is np.ndarray:
            sim_data_kwargs[field.name] = np.full(n_time_bps, np.nan)
    sim_data_kwargs['t_s'] = t_common
    
    print(f"\n--- Interpolating Data to Master Time Vector (dt={dt}s) ---")
    
    # 3. Interpolate each dataset onto the master time vector
    for ds in datasets:
        print(f"Processing: {ds['name']} -> Target: {ds['attr_name']} [{ds['col_name']}] {ds['msg']}")
        
        interpolated_data = np.zeros(n_time_bps)
        time_sample = ds['t']
        data_sample = ds['data']
        
        interpolated_data = np.interp(t_common, time_sample, data_sample)
        
        # Overwrite the NaN array with interpolated data
        sim_data_kwargs[ds['attr_name']] = interpolated_data

    # Instantiate the Dataclass to enforce schema compliance
    sim_data_obj = SimData(**sim_data_kwargs)
    sim_data_obj.job_name = data_name
    sim_data_obj.description = data_name

    # 4. Save to .npz format
    out_dir = "./combined_data/"
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, output_name)
    sim_data_obj.save_npz(save_path + ".npz")
    sim_data_obj.save_csv(save_path + ".csv", dt=dt)
    
    print(f"\n--- Combination Complete ---")
    print(f"Data saved to: {save_path}")
    print(f"Total time steps: {n_time_bps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine and interpolate digitized plot CSVs into a sim_data .npz file.")
    
    parser.add_argument("files", nargs="+", help="List of .csv files to combine")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step for the interpolated master array (default: 0.01)")
    parser.add_argument("--filename", type=str, default="Combined_Data", help="Name of the output job/file")
    parser.add_argument("--dataname", type=str, default="Flight Test", help="Name of the output data")
    
    args = parser.parse_args()
    
    combine_data(args.files, dt=args.dt, output_name=args.filename, data_name=args.dataname)