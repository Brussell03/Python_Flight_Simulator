import numpy as np
from scipy import linalg
import control as ctrl
import math
import matplotlib.pyplot as plt

from src.dynamics.eom_wgs84 import eom_wgs84

def compute_state_space(x_trim, u_trim, vehicle, amod, cmod, state_names):
    """
    Computes the full 17x17 A and 17x4 B matrices, then rigorously truncates 
    them to the 10x10 core bare-airframe 6-DOF dynamics.
    
    Args:
        x_trim: Trimmed state vector
        u_trim: Trimmed control vector
        vehicle: Instantiated Vehicle object (replaces vmod)
        amod: Atmosphere model dictionary
        cmod: Control configuration dictionary
        state_names: List of state variable names
    """
    nx = len(x_trim)
    nu = len(u_trim)
    
    A_full = np.zeros((nx, nx))
    B_full = np.zeros((nx, nu))
    
    cmod['trim_flag'] = 'off'
    cmod['linearization_flag'] = 'on'
    
    eps_x = np.array([
        0.1, 0.1, 0.1,                  # u, v, w [m/s]
        1e-4, 1e-4, 1e-4,               # p, q, r [rad/s]
        1e-5, 1e-5, 1e-5, 1e-5,         # q0, q1, q2, q3 []
        1e-6, 1e-6,                     # lat, long [rad]
        1.0,                            # h [m]
        0.1, 0.1, 0.1,                  # dela, dele, delr [deg]
        1.0                             # m_fuel [kg]
    ])
    eps_u = np.array([0.1, 0.1, 0.1, 0.01])
    
    # Preallocate
    dx = np.empty((17,), dtype=float) 
    auxillary_data = np.empty((16,), dtype=float)
    
    print("Perturbing full system...")
    for j in range(nx):
        x_plus = np.copy(x_trim)
        x_plus[j] += eps_x[j]
        cmod['dela_cmd_deg'], cmod['dele_cmd_deg'], cmod['delr_cmd_deg'], cmod['throttle_percent'] = u_trim
        dx_plus, _ = eom_wgs84(0, x_plus, dx.copy(), auxillary_data, vehicle, amod, cmod)
        
        x_minus = np.copy(x_trim)
        x_minus[j] -= eps_x[j]
        dx_minus, _ = eom_wgs84(0, x_minus, dx.copy(), auxillary_data, vehicle, amod, cmod)
        
        A_full[:, j] = (dx_plus - dx_minus) / (2 * eps_x[j])
        
    for j in range(nu):
        u_plus = np.copy(u_trim)
        u_plus[j] += eps_u[j]
        cmod['dela_cmd_deg'], cmod['dele_cmd_deg'], cmod['delr_cmd_deg'], cmod['throttle_percent'] = u_plus
        dx_plus, _ = eom_wgs84(0, x_trim, dx, auxillary_data, vehicle, amod, cmod)
        
        u_minus = np.copy(u_trim)
        u_minus[j] -= eps_u[j]
        cmod['dela_cmd_deg'], cmod['dele_cmd_deg'], cmod['delr_cmd_deg'], cmod['throttle_percent'] = u_minus
        dx_minus, _ = eom_wgs84(0, x_trim, dx, auxillary_data, vehicle, amod, cmod)
        
        B_full[:, j] = (dx_plus - dx_minus) / (2 * eps_u[j])

    cmod['linearization_flag'] = 'off'
    
    # --- TRUNCATION TO CORE 6-DOF ---
    # Isolate u, v, w, p, q, r, q0, q1, q2, q3
    core_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
    act_idx  = [13, 14, 15] # Actuator achieved states
    
    A_core = A_full[np.ix_(core_idx, core_idx)]
    
    # Construct B_core using the partial derivatives of the core states w.r.t achieved deflections
    B_core = np.zeros((len(core_idx), 4))
    B_core[:, 0:3] = A_full[np.ix_(core_idx, act_idx)] # Aero surface inputs (from A_full)
    B_core[:, 3]   = B_full[core_idx, 3]               # Throttle input (from B_full)
    
    core_state_names = [state_names[i] for i in core_idx]
    core_control_names = ['dela_ach', 'dele_ach', 'delr_ach', 'throttle_pct']
    
    return A_core, B_core, core_state_names, core_control_names

def analyze_eigenvalues(A):
    """
    Extracts and categorizes eigenvalues from the A matrix.
    """
    eigenvalues, eigenvectors = linalg.eig(A)
    
    print("\n--- System Eigenvalues (Poles) ---")
    print(f"{'Real':>10} {'Imaginary':>15} {'Freq (rad/s)':>15} {'Damping':>12}")
    print("-" * 55)
    
    for eig in eigenvalues:
        sigma = np.real(eig)
        omega_d = np.imag(eig)
        
        # Ignore kinematic integration poles (0.0) from lat/long/quaternions
        if abs(sigma) < 1e-7 and abs(omega_d) < 1e-7:
            continue
            
        omega_n = math.sqrt(sigma**2 + omega_d**2)
        if omega_n > 0:
            damping = -sigma / omega_n
        else:
            damping = 0.0
            
        print(f"{sigma:10.4f} {omega_d:15.4f} {omega_n:15.4f} {damping:12.4f}")

def analyze_mode_shapes(A_core, core_state_names):
    """
    Calculates eigenvectors using state-scaling on the truncated matrix.
    """
    eigenvalues, eigenvectors = linalg.eig(A_core)
    
    # Scales for: u, v, w, p, q, r, q0, q1, q2, q3
    state_scales = np.array([
        700.0, 50.0, 50.0,   # velocities [m/s]
        0.1, 0.1, 0.1,       # rates [rad/s]
        1.0, 1.0, 1.0, 1.0   # quaternions []
    ])
    
    print("\n--- Scaled Eigenvector Participation (Bare Airframe) ---")
    for i, eig in enumerate(eigenvalues):
        sigma = np.real(eig)
        omega_d = np.imag(eig)
        
        # Omit the singular zero-pole inherent to the quaternion constraint (q0^2+q1^2+q2^2+q3^2=1)
        if abs(sigma) < 1e-6 and abs(omega_d) < 1e-6:
            continue
            
        print(f"\nMode Pole: {sigma:.4f} + {omega_d:.4f}j")
        
        scaled_vector = eigenvectors[:, i] / state_scales
        mag = np.abs(scaled_vector)
        
        total_mag = np.sum(mag)
        if total_mag > 0:
            mag_pct = (mag / total_mag) * 100
        else:
            mag_pct = np.zeros_like(mag)
            
        top_indices = np.argsort(mag_pct)[::-1]
        for j in range(4):
            idx = top_indices[j]
            if mag_pct[idx] > 0.5: 
                print(f"  {core_state_names[idx]:>10}: {mag_pct[idx]:5.1f}%")

def plot_linear_response(A, B, t_end=20.0):
    """
    Simulates the linear state-space model's response to an initial condition disturbance.
    Handles the (outputs, samples) output convention of the python-control library.
    """
    nx = A.shape[0]
    nu = B.shape[1]
    
    C = np.eye(nx)
    D = np.zeros((nx, nu))
    
    sys = ctrl.ss(A, B, C, D)
    t = np.linspace(0, t_end, 1000)
    
    # 0.1 rad/s (~5.7 deg/s) pitch rate spike
    x0 = np.zeros(nx)
    x0[4] = 0.1 
    
    # Simulate response using python-control convention
    t_out, y_out = ctrl.initial_response(sys, t, X0=x0)
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.set_facecolor('black')
    fig.suptitle('Linear Initial Condition Response (Unaugmented X-15)\nDisturbance: 0.1 rad/s Pitch Rate', fontsize=14, fontweight='normal', color='yellow')
    
    axes[0, 0].plot(t_out, y_out[3, :], color='cyan', linewidth=1.5) # p is index 3
    axes[0, 0].set_xlabel('Time [s]', color='white')
    axes[0, 0].set_ylabel('Roll Rate Perturbation [rad/s]', color='white')
    axes[0, 0].grid(True)
    axes[0, 0].set_facecolor('black')
    axes[0, 0].tick_params(colors = 'white')
    
    axes[0, 1].plot(t_out, y_out[4, :], color='magenta', linewidth=1.5) # q is index 4
    axes[0, 1].set_xlabel('Time [s]', color='white')
    axes[0, 1].set_ylabel('Pitch Rate Perturbation [rad/s]', color='white')
    axes[0, 1].grid(True)
    axes[0, 1].set_facecolor('black')
    axes[0, 1].tick_params(colors = 'white')
    
    axes[1, 0].plot(t_out, y_out[5, :], color='yellow', linewidth=1.5) # r is index 5
    axes[1, 0].set_xlabel('Time [s]', color='white')
    axes[1, 0].set_ylabel('Yaw Rate Perturbation [rad/s]', color='white')
    axes[1, 0].grid(True)
    axes[1, 0].set_facecolor('black')
    axes[1, 0].tick_params(colors = 'white')
    
    axes[1, 1].plot(t_out, y_out[1, :], color='lime', linewidth=1.5) # v is index 1
    axes[1, 1].set_xlabel('Time [s]', color='white')
    axes[1, 1].set_ylabel('Lateral Velocity Perturb [m/s]', color='white')
    axes[1, 1].grid(True)
    axes[1, 1].set_facecolor('black')
    axes[1, 1].tick_params(colors = 'white')

    plt.tight_layout()
    plt.show()

def plot_clean_bode(sys, title, color='cyan'):
    """
    Plots a clean, readable Bode plot using direct axes control.
    """
    # Calculate margins
    gm, pm, wg, wp = ctrl.margin(sys)
    
    # Convert Gain Margin to dB
    gm_db = 20 * np.log10(gm) if np.isfinite(gm) else np.inf
    
    # Define frequency range and force inclusion of crossover points
    # This ensures plot markers are perfectly accurate
    freq_points = np.logspace(-2, 4, 1000)
    if wg > 0: freq_points = np.append(freq_points, wg)
    if wp > 0: freq_points = np.append(freq_points, wp)
    omega = np.sort(np.unique(freq_points))
    
    # Generate data
    mag_raw, phase_rad, omega = ctrl.frequency_response(sys, omega)
    
    # 2. Perform Manual Conversions
    mag_db = 20 * np.log10(mag_raw)
    phase_deg = np.rad2deg(phase_rad)
    phase_deg = np.mod(phase_deg + 180, 360) - 180

    with plt.style.context('dark_background'):
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Magnitude Plot
        ax_mag.semilogx(omega, mag_db, color=color, linewidth=2.0)
        ax_mag.axhline(0, color='white', linestyle=':', alpha=0.5) 
        
        if wg > 0:
            # Direct indexing is now safe because wg is explicitly in omega
            idx = np.where(omega == wg)[0][0]
            ax_mag.plot(wg, mag_db[idx], 'ro', markersize=8, label=f'Gain Crossover: {wg:.2f} rad/s')
        
        ax_mag.set_title(f"{title}\nGM: {gm_db:.2f} dB, PM: {pm:.2f} deg")
        ax_mag.set_ylabel('Magnitude [dB]')
        ax_mag.grid(True, which='both', linestyle='--', color='gray', alpha=0.3)
        ax_mag.legend()

        # Phase Plot
        ax_phase.semilogx(omega, phase_deg, color='magenta', linewidth=2.0)
        ax_phase.axhline(-180, color='white', linestyle=':', alpha=0.5) 
        
        if wp > 0:
            idx = np.where(omega == wp)[0][0]
            ax_phase.plot(wp, phase_deg[idx], 'ro', markersize=8, label=f'Phase Crossover: {wp:.2f} rad/s')
            
        ax_phase.set_ylabel('Phase [deg]')
        ax_phase.set_xlabel('Frequency [rad/s]')
        ax_phase.grid(True, which='both', linestyle='--', color='gray', alpha=0.3)
        ax_phase.legend()
        
        plt.tight_layout()
        plt.show()

    print(f"\n--- Analysis for {title} ---")
    print(f"Gain Margin: {gm_db:.2f} dB (at {wg:.2f} rad/s)")
    print(f"Phase Margin: {pm:.2f} deg (at {wp:.2f} rad/s)")

def advanced_stability_analysis(A_core, B_core, core_state_names, core_control_names):
    """
    Performs frequency-domain analysis using direct eigenvalue extraction 
    to avoid MIMO squareness constraints.
    """
    nx = A_core.shape[0]
    nu = B_core.shape[1]
    
    # Observe angular rates: p, q, r
    C = np.zeros((3, nx))
    C[0, 3] = 1.0  
    C[1, 4] = 1.0  
    C[2, 5] = 1.0  
    D = np.zeros((3, nu))
    
    sys = ctrl.ss(A_core, B_core, C, D)
    
    # 1. Manual Pole-Zero Map (Plotting Poles Only)
    # We ignore zeros here to avoid the MIMO squareness error
    poles = ctrl.poles(sys)
    
    plt.figure(figsize=(8, 8))
    plt.style.use('dark_background')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Stability Limit')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='cyan', s=100, label='System Poles')
    
    plt.title('Bare Airframe Pole Map (Eigenvalues)')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(color='gray', linestyle=':', alpha=0.5)
    plt.legend()
    plt.show()

    # 2. Bode Plots (Using SISO slices)
    # Pitch axis: Elevator -> q
    sys_dele_to_q = sys[1, 1] 
    
    # Yaw axis: Rudder -> r
    sys_delr_to_r = sys[2, 2]

    plot_clean_bode(sys_dele_to_q, 'Bode: Elevator to Pitch Rate', color='cyan')
    plot_clean_bode(sys_delr_to_r, 'Bode: Rudder to Yaw Rate', color='yellow')