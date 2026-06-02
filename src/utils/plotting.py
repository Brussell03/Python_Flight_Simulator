import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import os
from src.utils.constants import R2D

class SimulatorPlotter:
    """
    Handles all visual output for the flight simulation.
    Maps data arrays to plots using specified indices.
    """
    def __init__(self, dataset_list, plot_dir):
        """
        dataset_list: List of dictionaries [{'name': str, 'data': np.array}]
        """
        self.plot_dir = plot_dir
        
        # If the user passes a single dict, convert it to a list
        if isinstance(dataset_list, dict):
            dataset_list = [dataset_list]
        
        self.datasets = [self._process_dataset(d) for d in dataset_list]
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.datasets)))
        
        self.save = plot_dir is not None
    
    def _process_dataset(self, item):
        """Maps columns to keys and stores the job name."""
        data = item['data']
        return {
            'name': item['name'],
            't': data[:, 0],
            'u': data[:, 1], 'v': data[:, 2], 'w': data[:, 3],
            'p': data[:, 4] * R2D, 'q': data[:, 5] * R2D, 'r': data[:, 6] * R2D,
            'lat': data[:, 11] * R2D, 'lon': data[:, 12] * R2D, 'alt': data[:, 13],
            'dela_ach': data[:, 14], 'dele_ach': data[:, 15], 'delr_ach': data[:, 16],
            'mach': data[:, 20], 'alpha': data[:, 21] * R2D, 'beta': data[:, 22] * R2D, 'tas': data[:, 23],
            'phi': data[:, 24] * R2D, 'theta': data[:, 25] * R2D, 'psi': data[:, 26] * R2D,
            'u_n': data[:, 27], 'v_n': data[:, 28], 'w_n': data[:, 29],
            'dela_cmd': data[:, 30], 'dele_cmd': data[:, 31], 'delr_cmd': data[:, 32],
            'throttle': data[:, 33]
        }

    def _setup_figure(self, title, rows, cols, figsize):
        """Internal helper for centralized figure formatting."""
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.patch.set_facecolor('#121212') 
        fig.suptitle(title, color='#E0E0E0', fontsize=14, fontweight='bold')
        return fig, np.atleast_1d(axes)

    def _format_ax(self, ax, ylabel, xlabel='Time [s]', equal_aspect=False, min_range=1e-2):
        """Internal helper for clean, dark-mode axis aesthetics and scaling limits."""
        ax.set_facecolor('#1E1E1E')
        ax.set_ylabel(ylabel, color='#B0B0B0', fontsize=10)
        ax.set_xlabel(xlabel, color='#B0B0B0', fontsize=10)
        ax.tick_params(colors='#808080', labelsize=9)
        ax.grid(color='#333333', linestyle='--', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color('#404040')
        
        if equal_aspect:
            # Source: Matplotlib documentation for set_aspect
            ax.set_aspect('equal', adjustable='datalim')
        
        # Enforce minimum axis ranges to prevent micro-scaling on noise
        ymin, ymax = ax.get_ylim()
        if abs(ymax - ymin) < min_range:
            mid = (ymax + ymin) / 2.0
            ax.set_ylim(mid - min_range / 2.0, mid + min_range / 2.0)
            
        xmin, xmax = ax.get_xlim()
        if abs(xmax - xmin) < min_range:
            mid = (xmax + xmin) / 2.0
            ax.set_xlim(mid - min_range / 2.0, mid + min_range / 2.0)
    
    def _plot_all_time(self, ax, key):
        """Helper to loop through all loaded datasets, skipping None and NaN arrays."""
        for i, ds in enumerate(self.datasets):
            if key and ds[key] is not None and not np.all(np.isnan(ds[key])):
                ax.plot(ds['t'], ds[key], color=self.colors[i], linewidth=1.2, label=ds['name'])
        
        # Only draw legend if there are items to show
        handles, labels = ax.get_legend_handles_labels()
        if handles and len(self.datasets) > 1:
            ax.legend(handles, labels, loc='best', facecolor='#1E1E1E', edgecolor='#404040', labelcolor='#B0B0B0', fontsize=8)
    
    def _plot_all(self, ax, keyX, keyY):
        """Helper to loop through all loaded datasets, skipping None and NaN arrays."""
        for i, ds in enumerate(self.datasets):
            if keyX and keyY and ds[keyX] is not None and ds[keyY] is not None and not np.all(np.isnan(ds[keyX])) and not np.all(np.isnan(ds[keyY])):
                ax.plot(ds[keyX], ds[keyY], color=self.colors[i], linewidth=1.2, label=ds['name'])
        
        handles, labels = ax.get_legend_handles_labels()
        if handles and len(self.datasets) > 1:
            ax.legend(handles, labels, loc='best', facecolor='#1E1E1E', edgecolor='#404040', labelcolor='#B0B0B0', fontsize=8)
    
    def _plot_gradient_track(self, ax, keyX, keyY):
        """Plots lines with a color gradient mapped to time."""
        all_x = []
        all_y = []

        for ds in self.datasets:
            if ds[keyX] is None or ds[keyY] is None:
                continue
                
            x = ds[keyX]
            y = ds[keyY]
            t = ds['t']
            
            # Filter out NaNs for plotting and bounds calculation
            mask = ~np.isnan(x) & ~np.isnan(y)
            if not np.any(mask):
                continue
                
            x_clean, y_clean = x[mask], y[mask]
            
            # Create segments for LineCollection
            points = np.array([x_clean, y_clean]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create the LineCollection
            norm = plt.Normalize(t.min(), t.max())
            lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=1.5, alpha=0.8)
            lc.set_array(t[mask])
            
            line = ax.add_collection(lc)
            
            # Collect data for manual axis scaling
            all_x.append(x_clean)
            all_y.append(y_clean)

        # Only set limits if we have valid data
        if all_x and all_y:
            x_flat = np.concatenate(all_x)
            y_flat = np.concatenate(all_y)
            
            ax.set_xlim(np.nanmin(x_flat), np.nanmax(x_flat))
            ax.set_ylim(np.nanmin(y_flat), np.nanmax(y_flat))
            
            # Re-add colorbar if data exists
            try:
                cbar = plt.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Time [s]', color='#B0B0B0', fontsize=9)
                cbar.ax.tick_params(colors='#B0B0B0', labelsize=8)
            except UnboundLocalError:
                pass # No data rendered, skip colorbar
    
    def plot_6dof(self, filename="6dof.png", show=False):
        fig, axes = self._setup_figure("6-DOF State Vectors", 2, 3, (12, 8))
        keys = ['u', 'v', 'w', 'p', 'q', 'r']
        labels = ['u [m/s]', 'v [m/s]', 'w [m/s]', 'p [deg/s]', 'q [deg/s]', 'r [deg/s]']
        
        for i, ax in enumerate(axes.flatten()):
            self._plot_all_time(ax, keys[i])
            self._format_ax(ax, labels[i])
            
        plt.tight_layout()
        if self.save: plt.savefig(os.path.join(self.plot_dir, filename), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)

    def plot_attitude(self, filename="attitude.png", show=False):
        fig, axes = self._setup_figure("Euler Angles", 3, 1, (10, 6))
        keys = ['phi', 'theta', 'psi']
        labels = ['Roll Angle [deg]', 'Pitch Angle [deg]', 'Yaw Angle [deg]']
        
        for i, ax in enumerate(axes.flatten()):
            self._plot_all_time(ax, keys[i])
            self._format_ax(ax, labels[i])
            
        plt.tight_layout()
        if self.save: plt.savefig(os.path.join(self.plot_dir, filename), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)

    def plot_controls(self, filename="controls.png", show=False):
        fig, axes = self._setup_figure("Actuation & Controls", 2, 2, (10, 6))
        
        ach_keys = ['dela_ach', 'dele_ach', 'delr_ach', 'throttle']
        cmd_keys = ['dela_cmd', 'dele_cmd', 'delr_cmd', None]
        labels = ['Aileron [deg]', 'Elevator [deg]', 'Rudder [deg]', 'Throttle [%]']
        
        for i, ax in enumerate(axes.flatten()):
            ach_k = ach_keys[i]
            cmd_k = cmd_keys[i]
            
            # Iterate through all loaded datasets
            for ds_idx, ds in enumerate(self.datasets):
                color = self.colors[ds_idx]
                
                # Plot Achieved (Solid, Dataset Color)
                ax.plot(ds['t'], ds[ach_k], color=color, linewidth=1.5, label=f"{ds['name']} (Ach)")
                
                # Plot Command (Dashed, White)
                if cmd_k and ds[cmd_k] is not None:
                    ax.plot(ds['t'], ds[cmd_k], color='white', linewidth=1.2, linestyle='--', alpha=0.7, label=f"{ds['name']} (Cmd)")
            
            self._format_ax(ax, labels[i])
            handles, labels_leg = ax.get_legend_handles_labels()
            if handles: ax.legend(handles, labels_leg, loc='best', facecolor='#1E1E1E', edgecolor='#404040', labelcolor='#B0B0B0', fontsize=8)
            
        plt.tight_layout()
        if self.save: plt.savefig(os.path.join(self.plot_dir, filename), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)
        
    def plot_aerodynamics(self, filename="air_data.png", show=False):
        fig, axes = self._setup_figure("Aerodynamic States", 2, 2, (10, 6))
        keys = ['alpha', 'beta', 'mach', 'tas']
        labels = ['Angle of Attack [deg]', 'Angle of Sideslip [deg]', 'Mach Number', 'True Airspeed [m/s]']
        
        for i, ax in enumerate(axes.flatten()):
            self._plot_all_time(ax, keys[i])
            self._format_ax(ax, labels[i])
            
        plt.tight_layout()
        if self.save: plt.savefig(os.path.join(self.plot_dir, filename), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)

    def plot_geodetic(self, filename="geodetic.png", show=False):
        fig, axes = self._setup_figure("Geodetic Position", 2, 2, (10, 6))
        keys = ['lat', 'lon', 'alt']
        labels = ['Latitude [deg]', 'Longitude [deg]', 'Altitude [m]']
        
        ax_flat = axes.flatten()
        
        # Plot Time-Histories
        for i, sub_ax in enumerate(ax_flat[:-1]):
            self._plot_all_time(sub_ax, keys[i])
            self._format_ax(sub_ax, labels[i])
        
        # Ground Track Plot
        ax_track = ax_flat[3]
        self._plot_gradient_track(ax_track, 'lon', 'lat')
        self._format_ax(ax_track, 'Latitude [deg]', 'Longitude [deg]', equal_aspect=True)
        
        plt.tight_layout()
        if self.save: plt.savefig(os.path.join(self.plot_dir, filename), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)
        
    def plot_ned_velocity(self, filename="ned_velocity.png", show=False):
        fig, axes = self._setup_figure("Inertial Velocity (NED)", 3, 1, (10, 6))
        keys = ['u_n', 'v_n', 'w_n']
        labels = ['North Vel [m/s]', 'East Vel [m/s]', 'Down Vel [m/s]']
        
        for i, ax in enumerate(axes.flatten()):
            self._plot_all_time(ax, keys[i])
            self._format_ax(ax, labels[i])
            
        plt.tight_layout()
        if self.save: plt.savefig(os.path.join(self.plot_dir, filename), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)