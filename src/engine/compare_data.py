import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.utils.plotting import SimulatorPlotter
from src.utils.parse_to_sim_data import parse_all_csvs
from src.utils.config_parser import resolve_path

def load_npz(file_path):
    """Loads npz and extracts individual SimData arrays and meta."""
    loaded = np.load(file_path, allow_pickle=True)
    
    # Cast the NpzFile object to a standard dictionary.
    data_dict = dict(loaded)
    
    return {
        'data': data_dict,
        'name': data_dict['meta'].item()['data_name']
    }

def compare_data(file_paths, save_plots=False):
    """Loads arbitrary list of datasets and plots them."""
    
    datasets = [load_npz(f) for f in file_paths]

    # Setup the output directory
    out_dir = "./comparisons/"
    comparison_name = "_vs_".join([d['name'] for d in datasets])[:50] # Truncate if too long
    plot_dir = os.path.join(out_dir, comparison_name)
    
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = None
    
    plotter = SimulatorPlotter(datasets, plot_dir=plot_dir)

    # Render and display the plots
    plotter.plot_6dof(show=True)
    plotter.plot_attitude(show=True)
    plotter.plot_controls(show=True)
    plotter.plot_aerodynamics(show=True)
    plotter.plot_geodetic(show=True)
    plotter.plot_ned_velocity(show=True)
    
    # Block execution until user closes all plot windows
    plt.show(block=True)

def compare_data_to_files(dataset, compare_cfg, job_dir, title_prefix=""):
    """Loads datasets and plots them with provided dataset."""
    
    compare_path = resolve_path(job_dir, compare_cfg.get('path'))
    plot_cfg = compare_cfg.get('plots', {})
    plot_values = compare_cfg.get('plot_values', True)
    show_values = compare_cfg.get('show_values', True)
    plot_error = compare_cfg.get('plot_error', False)
    show_error = compare_cfg.get('show_error', False)
    save_dir = os.path.join(job_dir, "output/comparisons/") if compare_cfg.get('save_compare', False) else None
    
    datasets = [dataset]
    sim_datas, file_names = parse_all_csvs(compare_path)
    
    for i in range(len(sim_datas)):
        datasets.append({'name': file_names[i], 'data': sim_datas[i]})

    # Setup the output directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    plotter = SimulatorPlotter(datasets, title_prefix=title_prefix, plot_dir=save_dir)

    # Render and display the plots
    if plot_values:
        if plot_cfg.get('6dof', False):         plotter.plot_6dof(show=show_values)
        if plot_cfg.get('attitude', False):     plotter.plot_attitude(show=show_values)
        if plot_cfg.get('controls', False):     plotter.plot_controls(show=show_values)
        if plot_cfg.get('aerodynamics', False): plotter.plot_aerodynamics(show=show_values)
        if plot_cfg.get('geodetic', False):     plotter.plot_geodetic(show=show_values)
        if plot_cfg.get('ned_velocity', False): plotter.plot_ned_velocity(show=show_values)
    
    if show_values and not show_error:
        plt.show(block=True)
    elif not show_values:
        plt.close('all')
    
    if plot_error:
        if plot_cfg.get('6dof', False):         plotter.plot_6dof(show=show_error, error=True)
        if plot_cfg.get('attitude', False):     plotter.plot_attitude(show=show_error, error=True)
        if plot_cfg.get('controls', False):     plotter.plot_controls(show=show_error, error=True)
        if plot_cfg.get('aerodynamics', False): plotter.plot_aerodynamics(show=show_error, error=True)
        if plot_cfg.get('geodetic', False):     plotter.plot_geodetic(show=show_error, error=True)
        if plot_cfg.get('ned_velocity', False): plotter.plot_ned_velocity(show=show_error, error=True)
    
    # Block execution until user closes all plot windows
    if show_error: plt.show(block=True)

if __name__ == "__main__":
    # Command line usage: python compare_data.py path/to/run1.npz path/to/run2.npz ...
    # Command line usage: python -m src.engine.compare_data path/to/run1.npz path/to/run2.npz ...
    parser = argparse.ArgumentParser(description="Compare simulation data files.")
    
    # Positional argument: takes any number of files
    parser.add_argument("files", nargs="+", help="List of .npz files to compare")
    
    # Optional flag
    parser.add_argument("--save", action="store_true", help="Save the plots to disk")
    
    args = parser.parse_args()
    
    compare_data(args.files, save_plots=args.save)