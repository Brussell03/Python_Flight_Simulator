import argparse
import os
import sys
import numpy as np
import pandas as pd

from src.utils.interpolators import fastInterp1
from src.utils.constants import D2R, FT2M, R2D

# Map column headers directly to sim_data column indices
COLUMN_MAP = {
    'u_mps_vs_time_s': 1, 'v_mps_vs_time_s': 2, 'w_mps_vs_time_s': 3,
    'p_rps_vs_time_s': 4, 'q_rps_vs_time_s': 5, 'r_rps_vs_time_s': 6,
    'q0_vs_time_s': 7, 'q1_vs_time_s': 8, 'q2_vs_time_s': 9, 'q3_vs_time_s': 10,
    'lat_rad_vs_time_s': 11, 'long_rad_vs_time_s': 12, 'altitude_ft_vs_time_s': 13,
    'dela_deg_vs_time_s': 14, 'dele_deg_vs_time_s': 15, 'delr_deg_vs_time_s': 16,
    'm_fuel_kg_vs_time_s': 17,
    'Mach_vs_time_s': 20,
    'alpha_deg_vs_time_s': 21, 'beta_deg_vs_time_s': 22,
    'True_Airspeed_mps_vs_time_s': 23,
    'roll_rad_vs_time_s': 24, 'phi_rad_vs_time_s': 24,
    'pitch_rad_vs_time_s': 25, 'theta_rad_vs_time_s': 25,
    'yaw_rad_vs_time_s': 26, 'psi_rad_vs_time_s': 26,
    'u_n_mps_vs_time_s': 27, 'v_n_mps_vs_time_s': 28, 'w_n_mps_vs_time_s': 29,
    'delt_percent_vs_time_s': 33
}

def load_csv(file_path):
    """Loads CSV and ensures it is sorted by the independent variable (time)."""
    df = pd.read_csv(file_path, header=0)
    # Assume column 0 is 'x' (time) and column 1 is 'y' (data)
    # Sorting is critical because digitized data points can sometimes loop back slightly
    df = df.sort_values(by=df.columns[0])
    return df

def get_unit_conversion(col_name, idx):
    """
    Determines the necessary scaling factor to convert the input data
    to the baseline units expected by the sim_data array.
    """
    col_lower = col_name.lower()
    factor = 1.0
    conversion_msg = ""
    
    # 1. Distance: sim_data expects meters for altitude
    if idx == 13 and '_ft_' in col_lower:
        factor = FT2M
        conversion_msg = "(Converted ft -> m)"
        
    # 2. Angles: sim_data expects radians for rates, lat/lon, aero angles, and euler angles
    rad_indices = {4, 5, 6, 11, 12, 21, 22, 24, 25, 26}
    if idx in rad_indices and '_deg_' in col_lower:
        factor = D2R
        conversion_msg = "(Converted deg -> rad)"
        
    # 3. Control Surfaces: sim_data expects degrees
    deg_indices = {14, 15, 16}
    if idx in deg_indices and '_rad_' in col_lower:
        factor = R2D
        conversion_msg = "(Converted rad -> deg)"
        
    return factor, conversion_msg

def combine_data(file_paths, dt=0.01, output_name="Digitized_Flight_Data"):
    
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
            
        # Extract the exact header of the second column
        col_name = df.columns[1].strip()
        idx = COLUMN_MAP.get(col_name)
        
        if idx is None:
            print(f"Warning: Column header '{col_name}' in '{os.path.basename(file_path)}' not found in COLUMN_MAP. Skipping.")
            continue
        
        # Dynamically scale data based on expected units
        factor, msg = get_unit_conversion(col_name, idx)
        
        time_sample = df.iloc[:, 0].values
        data_sample = df.iloc[:, 1].values * factor
        
        # Track global bounds
        global_t_min = min(global_t_min, time_sample[0])
        global_t_max = max(global_t_max, time_sample[-1])
        
        datasets.append({
            'name': os.path.basename(file_path),
            'col_name': col_name,
            'idx': idx,
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
    
    # Initialize the 34-column sim_data array with np.nan to preserve float operation
    sim_data = np.full((n_time_bps, 34), np.nan)
    sim_data[:, 0] = t_common
    
    print(f"\n--- Interpolating Data to Master Time Vector (dt={dt}s) ---")
    # 3. Interpolate each dataset onto the master time vector
    for ds in datasets:
        print(f"Processing: {ds['name']} -> Index {ds['idx']} [{ds['col_name']}] {ds['msg']}")
        
        interpolated_data = np.zeros(n_time_bps)
        
        time_sample = ds['t']
        data_sample = ds['data']
        
        # Interpolate to the regularly spaced master database
        for ii in range(0, n_time_bps, 1):
            interpolated_data[ii] = fastInterp1(time_sample, data_sample, t_common[ii])
            
        # Insert into the main array
        sim_data[:, ds['idx']] = interpolated_data

    # 4. Save to .npz format
    out_dir = "./output_data/combined_data/"
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, f"{output_name}.npz")
    meta = {'job_name': output_name}
    
    np.savez(save_path, data=sim_data, meta=meta)
    
    print(f"\n--- Combination Complete ---")
    print(f"Data saved to: {save_path}")
    print(f"Total time steps: {n_time_bps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine and interpolate digitized plot CSVs into a sim_data .npz file.")
    
    parser.add_argument("files", nargs="+", help="List of .csv files to combine")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step for the interpolated master array (default: 0.01)")
    parser.add_argument("--name", type=str, default="Digitized_Flight_Data", help="Name of the output job/file")
    
    args = parser.parse_args()
    
    combine_data(args.files, dt=args.dt, output_name=args.name)