import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import os
from src.engine.sim_data import SimData
from src.utils.constants import R2D

class SimulatorPlotter:
    """
    Handles all visual output for the flight simulation..
    """
    def __init__(self, dataset_list, title_prefix="", plot_dir=None):
        """
        dataset_list: List of SimData
        """
        self.filetype = '.png'
        
        self.plot_dir = plot_dir
        self.title_prefix = title_prefix + "\n" if title_prefix != "" else ""
        
        # Enforce strict SimData usage, convert to list if single object is passed
        if isinstance(dataset_list, SimData):
            self.datasets = [dataset_list]
        elif isinstance(dataset_list, list) and all(isinstance(d, SimData) for d in dataset_list):
            self.datasets = dataset_list
        else:
            raise TypeError("dataset_list must be a SimData object or a list of SimData objects.")
        
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.datasets)))
        self.save = plot_dir is not None
        
        # Routing Map: Shorthand Key -> (SimData Attribute, Scaling Factor)
        self.attr_map = {
            't': ('t_s', 1.0),
            'u': ('u_b_mps', 1.0),
            'v': ('v_b_mps', 1.0),
            'w': ('w_b_mps', 1.0),
            'p': ('p_b_rps', R2D),
            'q': ('q_b_rps', R2D),
            'r': ('r_b_rps', R2D),
            'x': ('x_e_m', 1.0),
            'y': ('y_e_m', 1.0),
            'z': ('z_e_m', 1.0),
            'lat': ('lat_rad', R2D),
            'lon': ('long_rad', R2D),
            'alt': ('h_m', 1.0),
            'm_fuel': ('m_fuel_kg', 1.0),
            
            'phi': ('phi_rad', R2D),
            'theta': ('theta_rad', R2D),
            'psi': ('psi_rad', R2D),
            'phi_dot': ('phi_dot_rps', R2D),
            'theta_dot': ('theta_dot_rps', R2D),
            'psi_dot': ('psi_dot_rps', R2D),
            
            'cs': ('cs_mps', 1.0),
            'rho': ('rho_kgpm3', 1.0),
            'P': ('p_kgpms2', 0.001),
            'T': ('T_K', 1.0),
            'qbar': ('qbar_kgpms2', 0.001),
            'g': ('g_mag_mps2', 1.0),
            
            'mach': ('mach', 1.0),
            'alpha': ('alpha_rad', R2D),
            'beta': ('beta_rad', R2D),
            'tas': ('true_airspeed_mps', 1.0),
            
            'u_n': ('u_n_mps', 1.0),
            'v_n': ('v_n_mps', 1.0),
            'w_n': ('w_n_mps', 1.0),
            
            'f_x': ('Fx_b_kgmps2', 0.001),
            'f_y': ('Fy_b_kgmps2', 0.001),
            'f_z': ('Fz_b_kgmps2', 0.001),
            'l': ('l_b_kgm2ps2', 1.0),
            'm': ('m_b_kgm2ps2', 1.0),
            'n': ('n_b_kgm2ps2', 1.0),
            'n_x': ('n_x', 1.0),
            'n_y': ('n_y', 1.0),
            'n_z': ('n_z', 1.0),
            
            'dela_ach': ('dela_ach_rad', R2D),
            'dele_ach': ('dele_ach_rad', R2D),
            'delr_ach': ('delr_ach_rad', R2D),
            'delt_ach': ('delt_ach_pct', 1.0),
            
            'dela_cmd': ('dela_cmd_rad', R2D),
            'dele_cmd': ('dele_cmd_rad', R2D),
            'delr_cmd': ('delr_cmd_rad', R2D),
            'delt_cmd': ('delt_cmd_pct', 1.0),
            
            'dela_trim': ('dela_trim_rad', R2D),
            'dele_trim': ('dele_trim_rad', R2D),
            'delr_trim': ('delr_trim_rad', R2D),
            'delt_trim': ('delt_trim_pct', 1.0),
            
            'w_n': ('W_N_mps', 1.0),
            'w_e': ('W_E_mps', 1.0),
            'w_d': ('W_D_mps', 1.0),
        }
    
    def _get_val(self, ds: SimData, key: str):
        """Extracts and scales attributes from SimData based on internal routing map."""
        attr, scale = self.attr_map[key]
        val = getattr(ds, attr)
        return val * scale if val is not None else None

    def _get_comparison_data(self, idx, key, error):
        """Retrieves scaled telemetry. Computes error array if requested."""
        ds = self.datasets[idx]
        val = self._get_val(ds, key)
        
        if not error or idx == 0 or val is None:
            return ds.t_s, val
        
        base = self.datasets[0]
        base_val = self._get_val(base, key)
        
        if base_val is None:
            return ds.t_s, val # Cannot compute error without baseline data
        
        base_interp = np.interp(ds.t_s, base.t_s, base_val)
        return ds.t_s, val - base_interp

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
                ax.plot(t, val, color=self.colors[i], linewidth=1.2, label=ds.job_name)
        
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
                ax.plot(x_val, y_val, color=self.colors[i], linewidth=1.2, label=self.datasets[i].job_name)
        
        handles, labels = ax.get_legend_handles_labels()
        if handles and len(self.datasets) > 1:
            ax.legend(handles, labels, loc='best', facecolor='#1E1E1E', edgecolor='#404040', labelcolor='#B0B0B0', fontsize=8)
    
    def _plot_gradient_track(self, ax, keyX, keyY):
        """Plots lines with a color gradient mapped to time."""
        all_x = []
        all_y = []

        for ds in self.datasets:
            x = self._get_val(ds, keyX)
            y = self._get_val(ds, keyY)
            t = ds.t_s
            
            if x is None or y is None:
                continue
            
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
        labels = ['u Body Velocity [m/s]', 'v Body Velocity [m/s]', 'w Body Velocity [m/s]', 'Roll Rate [deg/s]', 'Pitch Rate [deg/s]', 'Yaw Rate [deg/s]']
        
        for i, ax in enumerate(axes.flatten()):
            self._plot_all_time(ax, keys[i], error)
            self._format_ax(ax, labels[i], is_error=error)
            
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)

    def plot_attitude(self, filename="attitude", show=False, error=False):
        fig, axes = self._setup_figure("Euler Angles", 3, 1, (12, 8), error)
        keys = ['phi', 'theta', 'psi']
        rate_keys = ['phi_dot', 'theta_dot', 'psi_dot']
        labels = ['Roll Angle [deg]', 'Pitch Angle [deg]', 'Yaw Angle [deg]']
        rate_labels = ['Roll Rate [deg/s]', 'Pitch Rate [deg/s]', 'Yaw Rate [deg/s]']
        
        for i, ax in enumerate(axes.flatten()):
            # Primary Axis (Angles)
            self._plot_all_time(ax, keys[i], error)
            self._format_ax(ax, labels[i], is_error=error)
            
            if error is False and len(self.datasets) == 1:
                # Secondary Axis (Rates)
                ax2 = ax.twinx()
                for ds_idx, ds in enumerate(self.datasets):
                    if error and ds_idx == 0:
                        continue
                    
                    t, rate_val = self._get_comparison_data(ds_idx, rate_keys[i], error)
                    if rate_val is not None and not np.all(np.isnan(rate_val)):
                        # Use dash-dot to distinguish rates from angles visually
                        ax2.plot(t, rate_val, color=self.colors[ds_idx], linewidth=1.0, linestyle='-.', alpha=0.6)
                
                # Format secondary axis manually to avoid wiping out primary grid/background
                ylabel2 = rate_labels[i].replace(' [', ' Error [') if error else rate_labels[i]
                ax2.set_ylabel(ylabel2, color='#909090', fontsize=10)
                ax2.tick_params(colors='#707070', labelsize=9)
                for spine in ax2.spines.values():
                    spine.set_color('#404040')
                ax2.grid(False) # Turn off grid for secondary axis to prevent crisscross pattern
            
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)

    def plot_controls(self, filename="controls", show=False, error=False):
        has_plotted = False
        fig, axes = self._setup_figure("Actuation & Controls", 2, 2, (12, 8), error)
        
        ach_keys = ['dela_ach', 'dele_ach', 'delr_ach', 'delt_ach']
        cmd_keys = ['dela_cmd', 'dele_cmd', 'delr_cmd', 'delt_cmd']
        trim_keys = ['dela_trim', 'dele_trim', 'delr_trim', 'delt_trim']
        labels = ['Aileron [deg]', 'Elevator [deg]', 'Rudder [deg]', 'Throttle [%]']
        
        for i, ax in enumerate(axes.flatten()):
            ach_k = ach_keys[i]
            cmd_k = cmd_keys[i]
            trim_k = trim_keys[i]
            
            # Iterate through all loaded datasets
            for ds_idx, ds in enumerate(self.datasets):
                if error and ds_idx == 0:
                    continue # Skip base
                
                color = self.colors[ds_idx]
                ach_val = self._get_val(ds, ach_k)
                cmd_val = self._get_val(ds, cmd_k)
                trim_val = self._get_val(ds, trim_k)

                # Plot Achieved (Solid, Dataset Color) - Only if data exists and is not all NaN
                if ach_val is not None and not np.all(np.isnan(ach_val)):
                    ax.plot(ds.t_s, ach_val, color=color, linewidth=1.5, label=f"{ds.job_name} (Ach)")
                    has_plotted = True
                
                # Plot Command (Dashed, White) - Only if data exists and is not all NaN
                if cmd_val is not None and not np.all(np.isnan(cmd_val)):
                    ax.plot(ds.t_s, cmd_val, color='white', linewidth=1.2, linestyle='--', alpha=0.7, label=f"{ds.job_name} (Cmd)")
                    has_plotted = True
                
                # Plot Trim (Dotted, Muted Alpha)
                if trim_val is not None and not np.all(np.isnan(trim_val)):
                    ax.plot(ds.t_s, trim_val, color=color, linewidth=1.0, linestyle=':', alpha=0.5, label=f"{ds.job_name} (Trim)")
                    has_plotted = True
            
            self._format_ax(ax, labels[i], is_error=error)
            handles, labels_leg = ax.get_legend_handles_labels()
            if handles: ax.legend(handles, labels_leg, loc='best', facecolor='#1E1E1E', edgecolor='#404040', labelcolor='#B0B0B0', fontsize=8)
        
        # Output suppression logic
        if not has_plotted:
            plt.close(fig)
            return
        
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)
        
    def plot_aerodynamics(self, filename="aerodynamics", show=False, error=False):
        fig, axes = self._setup_figure("Aerodynamic States", 2, 2, (12, 8), error)
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
        
    def plot_air_data(self, filename="air_data", show=False, error=False):
        fig, axes = self._setup_figure("Air Data", 2, 3, (12, 8), error)
        keys = ['cs', 'rho', 'P', 'T', 'qbar', 'g']
        labels = ['Speed of Sound [m/s]', 'Air Density [kg/m^3]', 'Air Pressure [kPa]', 'Air Temperature [K]', 'Dynamic Pressure [kPa]', 'Gravity [m/s^2]']
        
        for i, ax in enumerate(axes.flatten()):
            self._plot_all_time(ax, keys[i], error)
            self._format_ax(ax, labels[i], is_error=error)
            
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)

    def plot_geodetic(self, filename="geodetic", show=False, error=False):
        fig, axes = self._setup_figure("Geodetic Position", 2, 2, (10, 8), error)
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
        self._format_ax(ax_track, 'Latitude [deg]', 'Longitude [deg]', equal_aspect=True, min_range=1e-5)
        
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)
        
    def plot_ned_velocity(self, filename="ned_velocity", show=False, error=False):
        fig, axes = self._setup_figure("Inertial Velocity (NED)", 3, 1, (12, 8), error)
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
    
    def plot_forces(self, filename="forces", show=False, error=False):
        fig, axes = self._setup_figure("Forces and Moments", 2, 3, (12, 8), error)
        keys = ['f_x', 'f_y', 'f_z', 'l', 'm', 'n']
        labels = ['Fx [kN]', 'Fy [kN]', 'Fz [kN]', 'L [Nm]', 'M [Nm]', 'N [Nm]']
        
        for i, ax in enumerate(axes.flatten()):
            self._plot_all_time(ax, keys[i], error)
            self._format_ax(ax, labels[i], is_error=error)
            
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)
    
    def plot_load_factors(self, filename="load_factors", show=False, error=False):
        fig, axes = self._setup_figure("Load Factors", 3, 1, (12, 8), error)
        keys = ['n_x', 'n_y', 'n_z']
        labels = ['nx [g]', 'ny [g]', 'nz [g]']
        
        for i, ax in enumerate(axes.flatten()):
            self._plot_all_time(ax, keys[i], error)
            self._format_ax(ax, labels[i], is_error=error)
            
        plt.tight_layout()
        if self.save:
            name_suffix = "_error" if error else ""
            plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)
    
    def plot_3d_trajectory(self, filename="trajectory_3d", show=False, error=False):
        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor('#121212')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#121212')
        
        ax.xaxis.set_pane_color((0.12, 0.12, 0.12, 1.0))
        ax.yaxis.set_pane_color((0.12, 0.12, 0.12, 1.0))
        ax.zaxis.set_pane_color((0.12, 0.12, 0.12, 1.0))
        
        title_suffix = " Error" if error else ""
        fig.suptitle(self.title_prefix + "3D Spatial Trajectory (Local ENU)" + title_suffix, color='#E0E0E0', fontsize=14, fontweight='bold')
        
        base_ds = self.datasets[0]
        lat0 = base_ds.lat_rad[0] 
        lon0 = base_ds.long_rad[0]
        R_e = 6378137.0 
        
        # Accumulators for bounding box calculation
        all_x, all_y, all_z = [], [], []
        
        for i, ds in enumerate(self.datasets):
            if error and i == 0:
                continue
            
            lat = ds.lat_rad
            lon = ds.long_rad
            alt = ds.h_m
            
            if error:
                base_interp_lat = np.interp(ds.t_s, base_ds.t_s, base_ds.lat_rad)
                base_interp_lon = np.interp(ds.t_s, base_ds.t_s, base_ds.long_rad)
                base_interp_alt = np.interp(ds.t_s, base_ds.t_s, base_ds.h_m)
                
                x = (lon - base_interp_lon) * np.cos(lat0) * R_e
                y = (lat - base_interp_lat) * R_e
                z = alt - base_interp_alt
            else:
                x = (lon - lon0) * np.cos(lat0) * R_e
                y = (lat - lat0) * R_e
                z = alt
                
            ax.plot(x, y, z, color=self.colors[i], linewidth=1.5, label=ds.job_name)
            
            if not error and len(x) > 0:
                ax.scatter(x[0], y[0], z[0], color='green', marker='o', s=30, zorder=5)
                ax.scatter(x[-1], y[-1], z[-1], color='red', marker='x', s=30, zorder=5)
                
            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)

        # --- Enforce 1:1:1 Equal Aspect Ratio ---
        if all_x and all_y and all_z:
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            all_z = np.array(all_z)
            
            # Find the center of the bounding box
            mid_x = (all_x.max() + all_x.min()) * 0.5
            mid_y = (all_y.max() + all_y.min()) * 0.5
            mid_z = (all_z.max() + all_z.min()) * 0.5
            
            # Find the largest dimension to create a cubic bounding volume
            max_range = np.array([all_x.max() - all_x.min(), 
                                  all_y.max() - all_y.min(), 
                                  all_z.max() - all_z.min()]).max() / 2.0
            
            # Apply uniform limits
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Force matplotlib to draw the box as a cube
            ax.set_box_aspect([1, 1, 1])

        ax.set_xlabel('East [m]' if not error else 'East Error [m]', color='#B0B0B0')
        ax.set_ylabel('North [m]' if not error else 'North Error [m]', color='#B0B0B0')
        ax.set_zlabel('Altitude [m]' if not error else 'Altitude Error [m]', color='#B0B0B0')
        ax.tick_params(colors='#808080')
        
        handles, labels = ax.get_legend_handles_labels()
        if handles and len(self.datasets) > 1:
            ax.legend(handles, labels, loc='best', facecolor='#1E1E1E', edgecolor='#404040', labelcolor='#B0B0B0')
            
        plt.tight_layout()
        # if self.save:
        #     name_suffix = "_error" if error else ""
        #     plt.savefig(os.path.join(self.plot_dir, filename + name_suffix + self.filetype), facecolor=fig.get_facecolor(), dpi=150)
        if show: plt.show(block=False)