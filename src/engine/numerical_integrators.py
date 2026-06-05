import numpy as np
from scipy.integrate import solve_ivp

from src.utils.constants import NUM_AUX

def forward_euler(f, t, x, h_s, vmod, amod, cmod, u_trim, dx, auxillary_data):
    """
    Performs forward Euler integration to approximate the solution of a differential equation.
    """

    dx, auxillary_data = f(t, x, dx, auxillary_data, u_trim, vmod, amod, cmod)
    x_new = x + h_s * dx
    
    return x_new, auxillary_data

def AB2(f, t, x, t_prev, x_prev, h_s, vmod, amod, cmod, u_trim, dx, auxillary_data, i):
    """
    Performs the 2nd order Adams-Bashforth method to approximate the solution of a differential equation.
    """

    fim1, auxillary_data = f(t, x, dx, auxillary_data, u_trim, vmod, amod, cmod)
    if i == 0:
        x_new = x + h_s * fim1
    else:
        fim2, _ = f(t_prev, x_prev, dx, auxillary_data, u_trim, vmod, amod, cmod)
        x_new = x + 1.5*h_s*fim1 - 0.5*h_s*fim2
    
    return x_new, auxillary_data


def RK4(f, t, x, h_s, vmod, amod, cmod, u_trim, dx, auxillary_data):
    """
    Performs the 4th order Runge-Kutta method to approximate the solution of a differential equation.
    """
    fim1_k1, auxillary_data = f(t, x, dx, auxillary_data, u_trim, vmod, amod, cmod)
    k1 = h_s*fim1_k1
    fim1_k2, _ = f(t + 0.5*h_s, x + 0.5*k1, dx, auxillary_data, u_trim, vmod, amod, cmod)
    k2 = h_s*fim1_k2
    fim1_k3, _ = f(t + 0.5*h_s, x + 0.5*k2, dx, auxillary_data, u_trim, vmod, amod, cmod)
    k3 = h_s*fim1_k3
    fim1_k4, _ = f(t + h_s, x + k3, dx, auxillary_data, u_trim, vmod, amod, cmod)
    k4 = h_s*fim1_k4 
    x_new = x + 1/6*(k1 + 2.0*k2 + 2.0*k3 + k4)

    return x_new, auxillary_data

def adaptive_integration(eom_func, t_span, t_eval, x0, vehicle, amod, cmod, u_trim, method='RK45', rtol=1e-6, atol=1e-6):
    """
    Adaptive integrator wrapper using scipy's solve_ivp.
    Supports 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', and 'LSODA'.
    """
    print(f"[{method} Integration Engine Active]")
    
    # Preallocate arrays to prevent memory overhead inside the ODE evaluator
    dx_tmp = np.empty(len(x0), dtype=float)
    aux_tmp = np.empty(NUM_AUX, dtype=float)

    def eom_wrapper(t, x):
        # SciPy provides 't' as a float and 'x' as a 1D array
        eom_func(t, x, dx_tmp, aux_tmp, u_trim, vehicle, amod, cmod)
        return dx_tmp

    # Phase 1: State Integration
    res = solve_ivp(
        fun=eom_wrapper,
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol
    )

    if not res.success:
        print(f"\n!!! Integration Warning !!!\nSolver terminated early: {res.message}")

    x_out = res.y
    t_out = res.t

    # --- Commanded Inputs Reconstruction ---
    nt = len(t_out)
    aux_data_accum = np.zeros((NUM_AUX, nt))

    # Run the exact EOM logic on the finalized, accepted state array to harvest aux data.
    # This completely eliminates logic duplication (like rewriting SAS/trim routing).
    for i in range(nt):
        eom_func(t_out[i], x_out[:, i], dx_tmp, aux_tmp, u_trim, vehicle, amod, cmod)
        aux_data_accum[:, i] = aux_tmp

    return t_out, x_out, aux_data_accum