import time
import numpy as np
import sys
import os

from flightgear_python.fg_if import FDMConnection

from src.utils.constants import M2FT, R2D

def fdm_callback(fdm_data, event_pipe):
    # https://flightgear-python.readthedocs.io/en/1.3.0/_autosummary/flightgear_python.fdm_v25.fdm_struct.html
    
    if event_pipe.child_poll():
        
        # Unpack tuple
        alt_m_child, phi_rad_child, theta_rad_child, psi_rad_child, \
        u_b_ftps_child, v_b_ftps_child, w_b_ftps_child, alpha_rad_child, \
        beta_rad_child, lat_rad_child, long_rad_child, dela_rad_child, \
        dele_rad_child, delr_rad_child, v_north_ft_per_s_parent, \
        v_east_ft_per_s_parent, v_down_ft_per_s_parent, \
        = event_pipe.child_recv()  
        
        # Set only the data that we need to (we can force our own values)
        fdm_data['lat_rad']      = lat_rad_child
        fdm_data['lon_rad']      = long_rad_child
        fdm_data['alt_m']        = alt_m_child
        
        fdm_data['phi_rad']      = phi_rad_child
        fdm_data['theta_rad']    = theta_rad_child
        fdm_data['psi_rad']      = psi_rad_child
        
        fdm_data['alpha_rad']    = alpha_rad_child
        fdm_data['beta_rad']     = beta_rad_child
        
        fdm_data['v_body_u']     = u_b_ftps_child
        fdm_data['v_body_v']     = v_b_ftps_child
        fdm_data['v_body_w']     = w_b_ftps_child
        
        fdm_data['v_north_ft_per_s']  = v_north_ft_per_s_parent
        fdm_data['v_east_ft_per_s']   = v_east_ft_per_s_parent
        fdm_data['v_down_ft_per_s']   = v_down_ft_per_s_parent
        
        fdm_data['eng_state']      = [2, 0, 0, 0]
        
        fdm_data['elevator']       = -dele_rad_child
        fdm_data['left_aileron']   = -dela_rad_child
        fdm_data['right_aileron']  = dela_rad_child
        fdm_data['rudder']         = delr_rad_child
        
        fdm_data['speedbrake']     = 0
        
    # Return the whole structure
    return fdm_data

"""
Start FlightGear with: 
`./fgfs.exe --aircraft=X15-new --fdm=null --max-fps=30 --native-fdm=socket,out,30,localhost,5501,udp --native-fdm=socket,in,30,localhost,5502,udp --generic=socket,in,30,localhost,5502,udp,f16_custom_elevator`
"""
if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python visualization.py path/to/simulation_data.npy")
        sys.exit(1)
        
    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        sys.exit(1)
    
    # Get Python 6-DOF simulation data
    sim_data  = np.load(data_path)
    
    fdm_conn = FDMConnection()
    fdm_event_pipe = fdm_conn.connect_rx('localhost', 5501, fdm_callback)
    fdm_conn.connect_tx('localhost', 5502)
    
    # Start the FDM RX/TX loop and Ctrls RX/TX loop
    fdm_conn.start()

    nt_s = sim_data['t_s'].size
    i = 0
    while i < nt_s:
        
        # Increment time step counter
        # i += 1
        
        # Get present altitude
        v_body_u_parent         = sim_data['u_b_mps'][i]*M2FT # (converts m/s to ft/s)
        v_body_v_parent         = sim_data['v_b_mps'][i]*M2FT
        v_body_w_parent         = sim_data['w_b_mps'][i]*M2FT
        
        # Geodetic coordinates from states (Indices 11, 12, 13 mapped to x[10], x[11], x[12])
        lat_rad_parent          = sim_data['lat_rad'][i]
        long_rad_parent         = sim_data['long_rad'][i]
        alt_m_parent            = sim_data['h_m'][i]
        
        # Read reconstructed Euler Angles appended later in the sim_data array
        phi_rad_parent          = sim_data['phi_rad'][i]
        theta_rad_parent        = sim_data['theta_rad'][i]
        psi_rad_parent          = sim_data['psi_rad'][i]
        
        dela_rad_parent         = sim_data['dela_ach_rad'][i]
        dele_rad_parent         = sim_data['dele_ach_rad'][i]
        delr_rad_parent         = sim_data['delr_ach_rad'][i]
        alpha_rad_parent        = sim_data['alpha_rad'][i]
        beta_rad_parent         = sim_data['beta_rad'][i]
        v_north_ft_per_s_parent = sim_data['u_n_mps'][i]*M2FT # (converts m/s to ft/s)
        v_east_ft_per_s_parent  = sim_data['v_n_mps'][i]*M2FT
        v_down_ft_per_s_parent  = sim_data['w_n_mps'][i]*M2FT
        
        # Send tuple (could also do `fdm_conn.event_pipe.parent_send` so you just need to pass around `fdm_conn`)
        fdm_event_pipe.parent_send((alt_m_parent, phi_rad_parent, theta_rad_parent, psi_rad_parent, \
            v_body_u_parent, v_body_v_parent, v_body_w_parent, alpha_rad_parent, beta_rad_parent, \
            lat_rad_parent, long_rad_parent, dela_rad_parent, dele_rad_parent, delr_rad_parent, \
            v_north_ft_per_s_parent, v_east_ft_per_s_parent, v_down_ft_per_s_parent))
        
        i += 1
        time.sleep(0.007) # Target roughly 140hz replay sync
    
    print('\nVisualization completed.')