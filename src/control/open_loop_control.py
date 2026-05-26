def open_loop_aileron(t, oldata):  
    
    if oldata["aileron"] == 'on':
        if oldata["type"] == 'doublet':
            if t < oldata["t1_s"]:
                dela_deg = 0
            elif t < oldata["t2_s"]:
                dela_deg = -oldata["amplitude"]
            elif t < oldata["t3_s"]:
                dela_deg = oldata["amplitude"]
            else:
                dela_deg = 0
        else:
            print("Error: type key not presently recognized in oldata.")
    elif oldata["aileron"] == 'off':
        dela_deg = 0
    else:
        print("Error: aileron key in oldata dictionary must have value 'on' or 'off'.")
        
    return dela_deg

def open_loop_elevator(t, oldata):  
    
    if oldata["elevator"] == 'on':
        if oldata["type"] == 'doublet':
            if t < oldata["t1_s"]:
                dele_deg = 0
            elif t < oldata["t2_s"]:
                dele_deg = -oldata["amplitude"]
            elif t < oldata["t3_s"]:
                dele_deg = oldata["amplitude"]
            else:
                dele_deg = 0
        else:
            print("Error: type key not presently recognized in oldata.")
    elif oldata["elevator"] == 'off':
        dele_deg = 0
    else:
        print("Error: elevator key in oldata dictionary must have value 'on' or 'off'.")
        
    return dele_deg
    
def open_loop_rudder(t, oldata):  
    
    if oldata["rudder"] == 'on':
        if oldata["type"] == 'doublet':
            if t < oldata["t1_s"]:
                delr_deg = 0
            elif t < oldata["t2_s"]:
                delr_deg = -oldata["amplitude"]
            elif t < oldata["t3_s"]:
                delr_deg = oldata["amplitude"]
            else:
                delr_deg = 0
        else:
            print("Error: type key not presently recognized in oldata.")
    elif oldata["rudder"] == 'off':
        delr_deg = 0
    else:
        print("Error: rudder key in oldata dictionary must have value 'on' or 'off'.")
        
    return delr_deg

def open_loop_speed_brake():
    delsb_deg = 0
    
    return delsb_deg

def open_loop_throttle():
    delt_percent = 0
    
    return delt_percent