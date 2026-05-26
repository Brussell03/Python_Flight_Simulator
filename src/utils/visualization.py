"""
Simple Flight Dynamics Model (FDM) example of the X-15
"""
import time
import numpy as np
import sys
import os

from flightgear_python.fg_if import FDMConnection, CtrlsConnection

def ctrls_callback(ctrls_data, event_pipe):
    
    if event_pipe.child_poll():
        
        # Unpack tuple
        dela_deg_child, dele_deg_child, delr_deg_child = event_pipe.child_recv()  
        
        ctrls_data['aileron']  = dela_deg_child
        ctrls_data['elevator'] = dele_deg_child
        ctrls_data['rudder']   = delr_deg_child
        
        return ctrls_data

def fdm_callback(fdm_data, event_pipe):
    
    if event_pipe.child_poll():
        
        # Unpack tuple
        alt_m_child, phi_rad_child, theta_rad_child, psi_rad_child, \
        u_b_ftps_child, v_b_ftps_child, w_b_ftps_child, alpha_rad_child, \
        beta_rad_child, lat_rad_child, long_rad_child, dela_deg_child, \
        dele_deg_child, delr_deg_child, v_north_ft_per_s_parent, \
        v_east_ft_per_s_parent, v_down_ft_per_s_parent, \
        = event_pipe.child_recv()  
        
        # Set only the data that we need to (we can force our own values)
        fdm_data['alt_m']        = alt_m_child  
        fdm_data['phi_rad']      = phi_rad_child
        fdm_data['theta_rad']    = theta_rad_child
        fdm_data['psi_rad']      = psi_rad_child
        fdm_data['v_body_u']     = u_b_ftps_child
        fdm_data['v_body_v']     = v_b_ftps_child
        fdm_data['v_body_w']     = w_b_ftps_child
        fdm_data['alpha_rad']    = alpha_rad_child
        fdm_data['beta_rad']     = beta_rad_child
        fdm_data['lat_rad']      = lat_rad_child
        fdm_data['lon_rad']      = long_rad_child
        fdm_data['elevator']     = dele_deg_child
        fdm_data['v_north_ft_per_s_parent']  = v_north_ft_per_s_parent
        fdm_data['v_east_ft_per_s_parent']   = v_east_ft_per_s_parent
        fdm_data['v_down_ft_per_s_parent']   = v_down_ft_per_s_parent
        
    # Return the whole structure
    return fdm_data  

"""
Start FlightGear with: 
`./fgfs.exe --aircraft=X15 --fdm=null --max-fps=30 --native-fdm=socket,out,30,localhost,5501,udp --native-fdm=socket,in,30,localhost,5502,udp`
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
    data_pysim  = np.load(data_path)

    # Get time and other variables
    t_s   = data_pysim[:,0]; nt_s = t_s.size
    
    ctrls_conn = CtrlsConnection()
    ctrls_event_pipe = ctrls_conn.connect_rx('localhost', 5503, ctrls_callback)
    ctrls_conn.connect_tx('localhost', 5504)
    
    fdm_conn = FDMConnection()
    fdm_event_pipe = fdm_conn.connect_rx('localhost', 5501, fdm_callback)
    fdm_conn.connect_tx('localhost', 5502)
    
    # Start the FDM RX/TX loop and Ctrls RX/TX loop
    fdm_conn.start()
    ctrls_conn.start()

    i = 0
    while i < nt_s:
        
        # Increment time step counter
        i += 1
        
        # Get present altitude
        v_body_u_parent         = data_pysim[i,1]*3.28 # (converts m/s to ft/s)
        v_body_v_parent         = data_pysim[i,2]*3.28
        v_body_w_parent         = data_pysim[i,3]*3.28
        
        # Geodetic coordinates from states (Indices 11, 12, 13 mapped to x[10], x[11], x[12])
        lat_rad_parent          = data_pysim[i,11]
        long_rad_parent         = data_pysim[i,12]
        alt_m_parent            = data_pysim[i,13]
        
        # Read reconstructed Euler Angles appended later in the sim_data array
        phi_rad_parent          = data_pysim[i,31]
        theta_rad_parent        = data_pysim[i,32]
        psi_rad_parent          = data_pysim[i,33]
        
        dela_deg_parent         = data_pysim[i,14]
        dele_deg_parent         = data_pysim[i,15]
        delr_deg_parent         = data_pysim[i,16]
        alpha_rad_parent        = data_pysim[i,21]
        beta_rad_parent         = data_pysim[i,22]
        v_north_ft_per_s_parent = data_pysim[i,34]*3.28 # (converts m/s to ft/s)
        v_east_ft_per_s_parent  = data_pysim[i,35]*3.28
        v_down_ft_per_s_parent  = data_pysim[i,36]*3.28
        
        # Send tuple (could also do `fdm_conn.event_pipe.parent_send` so you just need to pass around `fdm_conn`)
        fdm_event_pipe.parent_send((alt_m_parent, phi_rad_parent, theta_rad_parent, psi_rad_parent, \
            v_body_u_parent, v_body_v_parent, v_body_w_parent, alpha_rad_parent, beta_rad_parent, \
            lat_rad_parent, long_rad_parent, dela_deg_parent, dele_deg_parent, delr_deg_parent, \
            v_north_ft_per_s_parent, v_east_ft_per_s_parent, v_down_ft_per_s_parent))  
        
        ctrls_event_pipe.parent_send((dela_deg_parent, dele_deg_parent, delr_deg_parent,))
        
        i += 1
        time.sleep(0.007) # Target roughly 140hz replay sync
    
    print('\nVisualization completed.')