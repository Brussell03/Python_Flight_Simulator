import numpy as np
import matplotlib.pyplot as plt
import control as ct
import math

from src.engine.state_mapping import StateIdx, AuxIdx, StateIdxSlices, TrimStateIdx, TrimStateIdxSlices

def compute_numerical_jacobians(eom, t, x, x_trim_ref, log_details=False):
    """
    Computes A and B matrices via central difference perturbation.
    """
    n_states = StateIdx.NUM_STATES
    n_controls = len(x[StateIdxSlices.CONTROL_SLICE])
    
    cmod = eom.control_model
    
    A = np.zeros((n_states, n_states))
    B = np.zeros((n_states, n_controls))
    
    # Pre-allocate arrays for the solver
    dx_eval = np.zeros(n_states)
    aux_data = np.zeros(len(AuxIdx))
    
    dx_pert = np.array([
        0.1, 0.1, 0.1,                  # u, v, w [m/s]
        1e-4, 1e-4, 1e-4,               # p, q, r [rad/s]
        1e-5, 1e-5, 1e-5, 1e-5,         # q0, q1, q2, q3 [nd]
        0.1, 0.1, 0.1,                  # x, y, z [m]
        1.0,                            # m_fuel [kg]
        1e-4, 1e-4, 1e-4,               # dela, dele, delr [rad]
        0.01,                           # delt [pct]
    ])
    du_pert = np.array([1e-4, 1e-4, 1e-4, 0.01])

    if log_details: print("\n[Linearization] Computing System Jacobians...")
    
    cmod['linearization_flag'] = True
    u_trim = x_trim_ref[TrimStateIdxSlices.ACT_TRIM_SLICE] if x_trim_ref is not None else x[StateIdxSlices.ACT_SLICE]
    cmod['dela_cmd_rad'], cmod['dele_cmd_rad'], cmod['delr_cmd_rad'], cmod['delt_cmd_pct'] = u_trim

    # Compute A Matrix (State Perturbations)
    for j in range(n_states):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        
        x_plus[j] += dx_pert[j]
        x_minus[j] -= dx_pert[j]
        
        # Note: EOM must support overriding control inputs via aux_data or direct injection
        f_plus, _ = eom.solve_eom(t, x_plus, np.copy(dx_eval), np.copy(aux_data), x_trim_ref)
        f_minus, _ = eom.solve_eom(t, x_minus, np.copy(dx_eval), np.copy(aux_data), x_trim_ref)
        
        A[:, j] = (f_plus - f_minus) / (2 * dx_pert[j])

    # Compute B Matrix (Control Command Perturbations)
    for j in range(n_controls):
        
        # Plus perturbation
        u_plus = np.copy(u_trim)
        u_plus[j] += du_pert[j]
        cmod['dela_cmd_rad'], cmod['dele_cmd_rad'], cmod['delr_cmd_rad'], cmod['delt_cmd_pct'] = u_plus
        f_plus, _ = eom.solve_eom(t, np.copy(x), np.copy(dx_eval), np.copy(aux_data), x_trim_ref)
        
        # Minus perturbation
        u_minus = np.copy(u_trim)
        u_minus[j] -= du_pert[j]
        cmod['dela_cmd_rad'], cmod['dele_cmd_rad'], cmod['delr_cmd_rad'], cmod['delt_cmd_pct'] = u_minus
        f_minus, _ = eom.solve_eom(t, np.copy(x), np.copy(dx_eval), np.copy(aux_data), x_trim_ref)
        
        B[:, j] = (f_plus - f_minus) / (2 * du_pert[j])

    cmod['linearization_flag'] = False
    
    return A, B

def build_state_space_model(A, B, state_names, control_names):
    """
    Constructs the control.StateSpace LTI object.
    Defaults to full observability (C = Identity, D = Zero).
    """
    n_states = A.shape[0]
    n_controls = B.shape[1]
    
    C = np.eye(n_states)
    D = np.zeros((n_states, n_controls))
    
    sys = ct.StateSpace(A, B, C, D)
    sys.state_labels = state_names
    sys.input_labels = control_names
    sys.output_labels = state_names
    return sys

def analyze_eigenvalues(sys, log_details=False):
    """
    Extracts and prints eigenvalues, damping ratios, and natural frequencies.
    """
    if log_details:
        print(f"\n{'='*60}")
        print(f"{'SYSTEM EIGENVALUES & MODES':^60}")
        print(f"{'='*60}")
        print(f"{'Mode/State':<15} | {'Eigenvalue':<20} | {'Damping (ζ)':<10} | {'Freq (ωn) [rad/s]':<15}")
        print("-" * 60)
    
    poles = sys.poles()
    
    for i, p in enumerate(poles):
        wn = np.abs(p)
        if wn > 1e-6:
            zeta = -np.real(p) / wn
        else:
            zeta = 0.0
            
        label = sys.state_labels[i] if i < len(sys.state_labels) else f"Mode {i}"
        
        complex_str = f"{np.real(p):.4f} {'+' if np.imag(p)>=0 else '-'} {np.abs(np.imag(p)):.4f}j"
        if log_details: print(f"{label:<15} | {complex_str:<20} | {zeta:<10.4f} | {wn:<15.4f}")

def _plot_poles_only(sys):
    """Helper method to plot only eigenvalues for non-square state-space matrices."""
    poles = sys.poles()
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='blue', s=70, label='System Poles')
    plt.axvline(0, color='black', linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    plt.title("System Poles")
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")
    plt.grid(True)
    plt.legend()

def plot_pole_zero(sys, inputs=None, outputs=None, show=True):
    """
    Generates a complex plane pole-zero map.
    Handles non-square systems by isolating square subsystems or plotting only poles.
    """
    plt.figure(figsize=(8, 8))
    
    # If specific channels are requested, compute P/Z for those subsystems
    if inputs is not None and outputs is not None:
        idx_in = [sys.input_labels.index(i) for i in inputs if i in sys.input_labels]
        idx_out = [sys.output_labels.index(o) for o in outputs if o in sys.output_labels]
        
        sys_sub = sys[idx_out, idx_in]
        
        try:
            pz_data = ct.pole_zero_map(sys_sub)
            pz_data.plot(grid=True, title="Pole-Zero Map (Subsystem)")
        except NotImplementedError:
            print("\n[Linearization] Subsystem is not square. Plotting poles only.")
            _plot_poles_only(sys_sub)
    else:
        # For the full non-square plant, transmission zeros are unsupported.
        print("\n[Linearization] Full plant is non-square. Plotting system poles only.")
        _plot_poles_only(sys)
        
    if show:
        plt.show()

def plot_bode(sys, inputs=None, outputs=None, show=True):
    """
    Generates Bode plots. 
    MIMO systems will generate dense subplots unless specific input/output pairs are requested.
    """
    plt.figure(figsize=(10, 8))
    
    # Filter MIMO down to requested pairs if provided
    if inputs is not None and outputs is not None:
        idx_in = [sys.input_labels.index(i) for i in inputs if i in sys.input_labels]
        idx_out = [sys.output_labels.index(o) for o in outputs if o in sys.output_labels]
        sys_sub = sys[idx_out, idx_in]
        ct.bode_plot(sys_sub, dB=True, Hz=False, grid=True)
    else:
        # Warning: A full 18x4 state system will create a 72-subplot figure. 
        # You should restrict this in the config.
        ct.bode_plot(sys, dB=True, Hz=False, grid=True)
        
    if show:
        plt.show()

def plot_linear_response(sys, t_end=30.0, input_idx=0, show=True):
    """
    Simulates the linear step response for a specific control channel.
    """
    t = np.linspace(0, t_end, 500)
    
    # Apply a unit step to the specified input channel
    t, y = ct.step_response(sys, T=t, input=input_idx)
    
    plt.figure(figsize=(12, 8))
    for i in range(y.shape[0]):
        # The control library returns (n_states, 1, n_time) for MIMO isolated inputs.
        # Squeeze flattens (1, 500) into the (500,) shape required by matplotlib.
        y_i = np.squeeze(y[i])
        
        # Filter out states with negligible response to keep the plot readable
        if np.max(np.abs(y_i)) > 1e-4:
            plt.plot(t, y_i, label=sys.output_labels[i])
            
    plt.title(f"Linear Step Response (Input: {sys.input_labels[input_idx]})")
    plt.xlabel("Time [s]")
    plt.ylabel("State Magnitude")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if show:
        plt.show()

def execute_linearization_commands(linearization_cfg, eom, t, x, x_trim_ref, state_names, control_names, log_details=False):
    """
    Master dispatcher. Routes the YAML configuration to the appropriate analysis methods.
    """
    if not linearization_cfg.get('enabled', False):
        return

    # 1. Compute State Space
    A, B = compute_numerical_jacobians(eom, t, x, x_trim_ref)
    sys = build_state_space_model(A, B, state_names, control_names)
    
    # 2. Config-Driven Execution
    show_plots = linearization_cfg.get('show_plots', True)
    
    # Extract IO pairs if defined (used for both Bode and P/Z maps)
    bode_cfg = linearization_cfg.get('bode', {})
    io_inputs = bode_cfg.get('inputs', None)
    io_outputs = bode_cfg.get('outputs', None)

    if linearization_cfg.get('eigenvalues', False):
        analyze_eigenvalues(sys)
        
    if linearization_cfg.get('pole_zero_map', False):
        plot_pole_zero(sys, inputs=io_inputs, outputs=io_outputs, show=show_plots)
        
    if bode_cfg.get('enabled', False):
        plot_bode(sys, inputs=io_inputs, outputs=io_outputs, show=show_plots)
            
    if 'step_response' in linearization_cfg:
        step_cfg = linearization_cfg['step_response']
        if step_cfg.get('enabled', False):
            # Target the specific control index (e.g., 'dele' -> 1)
            target_input = step_cfg.get('target_input', control_names[0])
            idx = control_names.index(target_input) if target_input in control_names else 0
            plot_linear_response(sys, t_end=step_cfg.get('t_end', 30.0), input_idx=idx, show=show_plots)

    if show_plots:
        plt.show(block=True)