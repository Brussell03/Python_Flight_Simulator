import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.constants import R2D

class SimulatorPlotter:
    """
    Handles all visual output for the flight simulation.
    Maps data arrays to plots using specified indices.
    """
    def __init__(self, data, plot_dir):
        self.data = data
        self.plot_dir = plot_dir
        
        # Map parameters by columns defined in sim_data
        self.t = self.data[:, 0]
        
        # 6-DOF Body States
        self.u, self.v, self.w = self.data[:, 1], self.data[:, 2], self.data[:, 3]
        self.p, self.q, self.r = self.data[:, 4] * R2D, self.data[:, 5] * R2D, self.data[:, 6] * R2D
        
        # Geodetic Position & Altitude
        self.lat = self.data[:, 11] * R2D
        self.lon = self.data[:, 12] * R2D
        self.alt = self.data[:, 13]
        
        # Controls
        self.dela_ach = self.data[:, 14]
        self.dele_ach = self.data[:, 15]
        self.delr_ach = self.data[:, 16]
        self.throttle = self.data[:, 40]
        
        # Air Data
        self.mach = self.data[:, 20]
        self.alpha = self.data[:, 21] * R2D
        self.beta = self.data[:, 22] * R2D
        self.tas = self.data[:, 23]
        
        # Euler Angles
        self.phi, self.theta, self.psi = self.data[:, 31] * R2D, self.data[:, 32] * R2D, self.data[:, 33] * R2D
        
        # NED Velocities
        self.u_n, self.v_n, self.w_n = self.data[:, 34], self.data[:, 35], self.data[:, 36]

    def _setup_figure(self, title, rows, cols, figsize):
        """Internal helper for centralized figure formatting."""
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.patch.set_facecolor('#121212') 
        fig.suptitle(title, color='#E0E0E0', fontsize=14, fontweight='bold')
        return fig, np.atleast_1d(axes)

    def _format_ax(self, ax, ylabel, xlabel='Time [s]'):
        """Internal helper for clean, dark-mode axis aesthetics."""
        ax.set_facecolor('#1E1E1E')
        ax.set_ylabel(ylabel, color='#B0B0B0', fontsize=10)
        ax.set_xlabel(xlabel, color='#B0B0B0', fontsize=10)
        ax.tick_params(colors='#808080', labelsize=9)
        ax.grid(color='#333333', linestyle='--', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color('#404040')

    def plot_6dof(self, filename="6dof.png"):
        fig, axes = self._setup_figure("6-DOF State Vectors", 2, 3, (12, 8))
        plot_data = [self.u, self.v, self.w, self.p, self.q, self.r]
        labels = ['u [m/s]', 'v [m/s]', 'w [m/s]', 'p [deg/s]', 'q [deg/s]', 'r [deg/s]']
        
        for i, ax in enumerate(axes.flatten()):
            ax.plot(self.t, plot_data[i], color='#00E5FF', linewidth=1.2) 
            self._format_ax(ax, labels[i])
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename), facecolor=fig.get_facecolor(), dpi=150)
        plt.close()

    def plot_attitude(self, filename="attitude.png"):
        fig, axes = self._setup_figure("Euler Angles", 3, 1, (10, 8))
        plot_data = [self.phi, self.theta, self.psi]
        labels = ['Roll Angle [deg]', 'Pitch Angle [deg]', 'Yaw Angle [deg]']
        
        for i, ax in enumerate(axes.flatten()):
            ax.plot(self.t, plot_data[i], color='#FF007F', linewidth=1.2) 
            self._format_ax(ax, labels[i])
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename), facecolor=fig.get_facecolor(), dpi=150)
        plt.close()

    def plot_controls(self, filename="controls.png"):
        fig, axes = self._setup_figure("Actuation & Controls", 2, 2, (10, 6))
        plot_data = [self.dela_ach, self.dele_ach, self.delr_ach, self.throttle]
        labels = ['Aileron [deg]', 'Elevator [deg]', 'Rudder [deg]', 'Throttle [%]']
        
        for i, ax in enumerate(axes.flatten()):
            ax.plot(self.t, plot_data[i], color='#39FF14', linewidth=1.2) 
            self._format_ax(ax, labels[i])
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename), facecolor=fig.get_facecolor(), dpi=150)
        plt.close()
        
    def plot_aerodynamics(self, filename="air_data.png"):
        fig, axes = self._setup_figure("Aerodynamic States", 2, 2, (10, 6))
        
        ax = axes.flatten()
        ax[0].plot(self.t, self.alpha, color='#FFEA00', linewidth=1.2)
        self._format_ax(ax[0], 'Angle of Attack [deg]')
        
        ax[1].plot(self.t, self.beta, color='#FFEA00', linewidth=1.2)
        self._format_ax(ax[1], 'Angle of Sideslip [deg]')
        
        ax[2].plot(self.t, self.mach, color='#FFEA00', linewidth=1.2)
        self._format_ax(ax[2], 'Mach Number')
        
        ax[3].plot(self.t, self.tas, color='#FFEA00', linewidth=1.2)
        self._format_ax(ax[3], 'True Airspeed [m/s]')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename), facecolor=fig.get_facecolor(), dpi=150)
        plt.close()

    def plot_geodetic(self, filename="geodetic.png"):
        fig, axes = self._setup_figure("Geodetic Position", 3, 1, (10, 8))
        plot_data = [self.lat, self.lon, self.alt]
        labels = ['Latitude [deg]', 'Longitude [deg]', 'Altitude [m]']
        
        for i, ax in enumerate(axes.flatten()):
            ax.plot(self.t, plot_data[i], color='#AA00FF', linewidth=1.2)
            self._format_ax(ax, labels[i])
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename), facecolor=fig.get_facecolor(), dpi=150)
        plt.close()
        
    def plot_ned_velocity(self, filename="ned_velocity.png"):
        fig, axes = self._setup_figure("Inertial Velocity (NED)", 3, 1, (10, 8))
        plot_data = [self.u_n, self.v_n, self.w_n]
        labels = ['North Vel [m/s]', 'East Vel [m/s]', 'Down Vel [m/s]']
        
        for i, ax in enumerate(axes.flatten()):
            ax.plot(self.t, plot_data[i], color='#00BFFF', linewidth=1.2)
            self._format_ax(ax, labels[i])
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename), facecolor=fig.get_facecolor(), dpi=150)
        plt.close()