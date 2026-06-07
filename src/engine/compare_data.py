import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.utils.plotting import SimulatorPlotter
from src.utils.parse_to_sim_data import parse_all_csvs

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

def compare_data_to_files(dataset, compare_path, plot_cfg, title_prefix="", save_dir=None, show_plots=False, plot_error=False):
    """Loads datasets and plots them with provided dataset."""
    datasets = [dataset]
    sim_datas, file_names = parse_all_csvs(compare_path)
    
    for i in range(len(sim_datas)):
        datasets.append({'name': file_names[i], 'data': sim_datas[i]})

    # Setup the output directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    plotter = SimulatorPlotter(datasets, title_prefix=title_prefix, plot_dir=save_dir, plot_error=plot_error)

    # Render and display the plots
    if plot_cfg.get('6dof', False):         plotter.plot_6dof(show=show_plots)
    if plot_cfg.get('attitude', False):     plotter.plot_attitude(show=show_plots)
    if plot_cfg.get('controls', False):     plotter.plot_controls(show=show_plots)
    if plot_cfg.get('aerodynamics', False): plotter.plot_aerodynamics(show=show_plots)
    if plot_cfg.get('geodetic', False):     plotter.plot_geodetic(show=show_plots)
    if plot_cfg.get('ned_velocity', False): plotter.plot_ned_velocity(show=show_plots)
    
    # Block execution until user closes all plot windows
    if show_plots: plt.show(block=True)

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