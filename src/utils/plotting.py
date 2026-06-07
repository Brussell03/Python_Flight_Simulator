import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import os
from src.utils.constants import R2D

class SimulatorPlotter:
    """
    Handles all visual output for the flight simulation.
    Maps data structures to plots using dataclass attributes or dict keys.
    """
    def __init__(self, dataset_list, title_prefix="", plot_dir=None):
        """
        dataset_list: List of dictionaries [{'name': str, 'data': SimData object or npz dict}]
        """
        self.filetype = '.png'
        
        self.plot_dir = plot_dir
        self.title_prefix = title_prefix + "\n" if title_prefix != "" else ""
        
        # If the user passes a single dict, convert it to a list
        if isinstance(dataset_list, dict):
            dataset_list = [dataset_list]
        
        self.datasets = [self._process_dataset(d) for d in dataset_list]
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.datasets)))
        self.save = plot_dir is not None
    
    def _process_dataset(self, item):
        """Maps SimData attributes to standard internal plotting keys."""
        data = item['data']
        
        def _get_val(obj, key):
            """Safely extract data whether obj is a live Dataclass or a loaded npz dict."""
            if hasattr(obj, key):
                return getattr(obj, key)
            elif isinstance(obj, dict) or hasattr(obj, '__getitem__'):
                return obj[key]
            raise KeyError(f"Missing expected data field: {key}")

        return {
            'name': item['name'],
            't': _get_val(data, 't_s'),
            'u': _get_val(data, 'u_b_mps'),
            'v': _get_val(data, 'v_b_mps'),
            'w': _get_val(data, 'w_b_mps'),
            'p': _get_val(data, 'p_b_rps') * R2D,
            'q': _get_val(data, 'q_b_rps') * R2D,
            'r': _get_val(data, 'r_b_rps') * R2D,
            'lat': _get_val(data, 'lat_rad') * R2D,
            'lon': _get_val(data, 'long_rad') * R2D,
            'alt': _get_val(data, 'h_m'),
            'dela_ach': _get_val(data, 'dela_ach_deg'),
            'dele_ach': _get_val(data, 'dele_ach_deg'),
            'delr_ach': _get_val(data, 'delr_ach_deg'),
            'mach': _get_val(data, 'mach'),
            'alpha': _get_val(data, 'alpha_rad') * R2D,
            'beta': _get_val(data, 'beta_rad') * R2D,
            'tas': _get_val(data, 'true_airspeed_mps'),
            'phi': _get_val(data, 'phi_rad') * R2D,
            'theta': _get_val(data, 'theta_rad') * R2D,
            'psi': _get_val(data, 'psi_rad') * R2D,
            'u_n': _get_val(data, 'u_n_mps'),
            'v_n': _get_val(data, 'v_n_mps'),
            'w_n': _get_val(data, 'w_n_mps'),
            'dela_cmd': _get_val(data, 'dela_cmd_deg'),
            'dele_cmd': _get_val(data, 'dele_cmd_deg'),
            'delr_cmd': _get_val(data, 'delr_cmd_deg'),
            'throttle': _get_val(data, 'delt_percent')
        }

    def _get_comparison_data(self, idx, key, error):
        """
        Retrieves data for plotting. If error is True, returns (ds[i] - base).
        Interpolates the base dataset to match current dataset's time vector.
        """
        ds = self.datasets[idx]
        val = ds[key]
        
        if not error or idx == 0 or val is None:
            return ds['t'], val
        
        # Error calculation mode
        base = self.datasets[0]
        # Interpolate base to current dataset's time
        base_interp = np.interp(ds['t'], base['t'], base[key])
        
        return ds['t'], val - base_interp

    def _setup_figure(self, title, rows, cols, figsize, is_error):
        """Internal helper for centralized figure formatting."""
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.patch.set_facecolor('#121212')
        title_suffix = " Error" if is_error else ""
        fig.suptitle(self.title_prefix + title + title_suffix, color='#E0E0E0', fontsize=14, fontweight='bold')
        return fig, np.atleast_1d(axes)

    def _format_ax(self, ax, ylabel, xlabel='Time [s]', equal_aspect=False, min_range=1e-2, is_error=False):
        """Internal helper for clean, dark-mode axis aesthetics and scaling limits."""
        if is_error:
            # Inserts ' Error' before the unit bracket
            ylabel = ylabel.replace(' [', ' Error [')
        
        ax.set_facecolor('#1E1E1E')
        ax.set_ylabel(ylabel, color='#B0B0B0', fontsize=10)
        ax.set_xlabel(xlabel, color='#B0B0B0', fontsize=10)
        ax.tick_params(colors='#808080', labelsize=9)
        ax.grid(color='#333333', linestyle='--', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color('#404040')
        
        if equal_aspect:
            ax.set_aspect('equal', adjustable='datalim')
        
        if not is_error:
            # Enforce minimum axis ranges to prevent micro-scaling on noise
            ymin, ymax = ax.get_ylim()
            if abs(ymax - ymin) < min_range:
                mid = (ymax + ymin) / 2.0
                ax.set_ylim(mid - min_range / 2.0, mid + min_range / 2.0)
                
            xmin, xmax = ax.get_xlim()
            if abs(xmax - xmin) < min_range:
                mid = (xmax + xmin) / 2.0
                ax.set_xlim(mid - min_range / 2.0, mid + min_range / 2.0)
    
    def _plot_all_time(self, ax, key, is_error):
        """Helper to loop through all loaded datasets, skipping None and NaN arrays."""
        for i, ds in enumerate(self.datasets):
            if is_error and i == 0:
                continue
            
            t, val = self._get_comparison_data(i, key, is_error)
            if val is not None and not np.all(np.isnan(val)):
                ax.plot(t, val, color=self.colors[i], linewidth=1.2, label=ds['name'])
        
        # Only draw legend if there are items to show
        handles, labels = ax.get_legend_handles_labels()
        if handles and len(self.datasets) > 1:
            ax.legend(handles, labels, loc='best', facecolor='#1E1E1E', edgecolor='#404040', labelcolor='#B0B0B0', fontsize=8)
    
    def _plot_all(self, ax, keyX, keyY, is_error):
        """Helper to loop through all loaded datasets, skipping None and NaN arrays."""
        for i, ds in enumerate(self.datasets):
            if is_error and i == 0:
                continue
            
            _, x_val = self._get_comparison_data(i, keyX, is_error)
            _, y_val = self._get_comparison_data(i, keyY, is_error)
            
            if x_val is not None and y_val is not None and not np.all(np.isnan(x_val)) and not np.all(np.isnan(y_val)):
                ax.plot(x_val, y_val, color=self.colors[i], linewidth=1.2, label=self.datasets[i]['name'])
        
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
    
    def plot_6dof(self, filename="6dof", show=False, error=False):
        fig, axes = self._setup_figure("6-DOF State Vectors", 2, 3, (12, 8), error)
        keys = ['u', 'v', 'w', 'p', 'q', 'r']
        labels = ['u [m/s]', 'v [m/s]', 'w [m/s]', 'p [deg/s]', 'q [deg/s]', 'r [deg/s]']
        
        for i, ax in enumerate(axes.flatten()):
            self._plot_all_time(ax, keys[i], error)
            self._format_ax(ax, labels[i], is_error=error)
            
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)

    def plot_attitude(self, filename="attitude", show=False, error=False):
        fig, axes = self._setup_figure("Euler Angles", 3, 1, (10, 6), error)
        keys = ['phi', 'theta', 'psi']
        labels = ['Roll Angle [deg]', 'Pitch Angle [deg]', 'Yaw Angle [deg]']
        
        for i, ax in enumerate(axes.flatten()):
            self._plot_all_time(ax, keys[i], error)
            self._format_ax(ax, labels[i], is_error=error)
            
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)

    def plot_controls(self, filename="controls", show=False, error=False):
        fig, axes = self._setup_figure("Actuation & Controls", 2, 2, (10, 6), error)
        
        ach_keys = ['dela_ach', 'dele_ach', 'delr_ach', 'throttle']
        cmd_keys = ['dela_cmd', 'dele_cmd', 'delr_cmd', None]
        labels = ['Aileron [deg]', 'Elevator [deg]', 'Rudder [deg]', 'Throttle [%]']
        
        for i, ax in enumerate(axes.flatten()):
            ach_k = ach_keys[i]
            cmd_k = cmd_keys[i]
            
            # Iterate through all loaded datasets
            for ds_idx, ds in enumerate(self.datasets):
                if error and ds_idx == 0:
                    continue # Skip base
                    
                color = self.colors[ds_idx]
                
                # Plot Achieved (Solid, Dataset Color)
                ax.plot(ds['t'], ds[ach_k], color=color, linewidth=1.5, label=f"{ds['name']} (Ach)")
                
                # Plot Command (Dashed, White)
                if cmd_k and ds[cmd_k] is not None:
                    ax.plot(ds['t'], ds[cmd_k], color='white', linewidth=1.2, linestyle='--', alpha=0.7, label=f"{ds['name']} (Cmd)")
            
            self._format_ax(ax, labels[i], is_error=error)
            handles, labels_leg = ax.get_legend_handles_labels()
            if handles: ax.legend(handles, labels_leg, loc='best', facecolor='#1E1E1E', edgecolor='#404040', labelcolor='#B0B0B0', fontsize=8)
            
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)
        
    def plot_aerodynamics(self, filename="air_data", show=False, error=False):
        fig, axes = self._setup_figure("Aerodynamic States", 2, 2, (10, 6), error)
        keys = ['alpha', 'beta', 'mach', 'tas']
        labels = ['Angle of Attack [deg]', 'Angle of Sideslip [deg]', 'Mach Number', 'True Airspeed [m/s]']
        
        for i, ax in enumerate(axes.flatten()):
            self._plot_all_time(ax, keys[i], error)
            self._format_ax(ax, labels[i], is_error=error)
            
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)

    def plot_geodetic(self, filename="geodetic", show=False, error=False):
        fig, axes = self._setup_figure("Geodetic Position", 2, 2, (10, 6), error)
        keys = ['lat', 'lon', 'alt']
        labels = ['Latitude [deg]', 'Longitude [deg]', 'Altitude [m]']
        
        ax_flat = axes.flatten()
        
        # Plot Time-Histories
        for i, sub_ax in enumerate(ax_flat[:-1]):
            self._plot_all_time(sub_ax, keys[i], error)
            self._format_ax(sub_ax, labels[i], is_error=error)
        
        # Ground Track Plot
        ax_track = ax_flat[3]
        self._plot_gradient_track(ax_track, 'lon', 'lat')
        self._format_ax(ax_track, 'Latitude [deg]', 'Longitude [deg]', equal_aspect=True, min_range=1e-5, is_error=error)
        
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)
        
    def plot_ned_velocity(self, filename="ned_velocity", show=False, error=False):
        fig, axes = self._setup_figure("Inertial Velocity (NED)", 3, 1, (10, 6), error)
        keys = ['u_n', 'v_n', 'w_n']
        labels = ['North Vel [m/s]', 'East Vel [m/s]', 'Down Vel [m/s]']
        
        for i, ax in enumerate(axes.flatten()):
            self._plot_all_time(ax, keys[i], error)
            self._format_ax(ax, labels[i], is_error=error)
            
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)