from models.dave_vehicle import DAVEVehicle

class F16(DAVEVehicle):
    def __init__(self):
        super().__init__(name="General Dynamics F-16 Subsonic Model", short_name="F16", aero_dml_path="models/F16/F16_S119_source/F16_aero.dml", inertia_dml_path="models/F16/F16_S119_source/F16_inertia.dml",
                         prop_dml_path="models/F16/F16_S119_source/F16_prop.dml", control_dml_path="models/F16/F16_S119_source/F16_control.dml",
                         gnc_dml_path="models/F16/F16_S119_source/F16_gnc.dml")