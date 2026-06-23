from collections import namedtuple
from enum import IntEnum

# ------------------------------------------------------
# State Vector Mapping
# ------------------------------------------------------

# Define the fields for the Numba-compatible structure
_StateIdxStruct = namedtuple('StateIdxStruct', [
    'U_B_MPS', 'V_B_MPS', 'W_B_MPS',
    'P_B_RPS', 'Q_B_RPS', 'R_B_RPS',
    'Q0', 'Q1', 'Q2', 'Q3',
    'X_E_M', 'Y_E_M', 'Z_E_M',
    'M_FUEL_KG',
    'DELA_ACH_RAD', 'DELE_ACH_RAD', 'DELR_ACH_RAD', 'DELT_ACH_PCT',
    'NUM_STATES'
])

StateIdx = _StateIdxStruct(
    # Velocities
    U_B_MPS = 0,
    V_B_MPS = 1,
    W_B_MPS = 2,
    
    # Body Rates
    P_B_RPS = 3,
    Q_B_RPS = 4,
    R_B_RPS = 5,
    
    # Quaternions
    Q0 = 6,
    Q1 = 7,
    Q2 = 8,
    Q3 = 9,
    
    # Position
    X_E_M = 10,
    Y_E_M = 11,
    Z_E_M = 12,
    
    # Mass
    M_FUEL_KG = 13,
    
    # Actuators
    DELA_ACH_RAD = 14,
    DELE_ACH_RAD = 15,
    DELR_ACH_RAD = 16,
    DELT_ACH_PCT = 17,
    
    # Total number of states
    NUM_STATES = 18,
)

class StateIdxSlices():
    # Define slice objects for vector extraction
    VEL_SLICE = slice(StateIdx.U_B_MPS, StateIdx.W_B_MPS + 1)
    ROT_SLICE = slice(StateIdx.P_B_RPS, StateIdx.R_B_RPS + 1)
    QUAT_SLICE = slice(StateIdx.Q0, StateIdx.Q3 + 1)
    POS_SLICE = slice(StateIdx.X_E_M, StateIdx.Z_E_M + 1)
    ACT_SLICE = slice(StateIdx.DELA_ACH_RAD, StateIdx.DELT_ACH_PCT + 1)
    STATE_SLICE = slice(StateIdx.U_B_MPS, StateIdx.M_FUEL_KG + 1)
    CONTROL_SLICE = slice(StateIdx.DELA_ACH_RAD, StateIdx.DELT_ACH_PCT + 1)

# ------------------------------------------------------
# Auxilliary Data Vector Mapping
# ------------------------------------------------------

class AuxIdx(IntEnum):
    # Actuators
    DELA_CMD_RAD = 0
    DELE_CMD_RAD = 1
    DELR_CMD_RAD = 2
    DELT_CMD_PCT = 3
    
    # Nav-Relative Rates
    P_NB_RPS = 4
    Q_NB_RPS = 5
    R_NB_RPS = 6
    
    # Forces
    FX_B_KGMPS2 = 7
    FY_B_KGMPS2 = 8
    FZ_B_KGMPS2 = 9
    
    # Moments
    L_B_KGM2PS2 = 10
    M_B_KGM2PS2 = 11
    N_B_KGM2PS2 = 12
    
    # Wind Velocities
    W_N_MPS = 13
    W_E_MPS = 14
    W_D_MPS = 15
    
    # Trims
    DELA_TRIM_RAD = 16
    DELE_TRIM_RAD = 17
    DELR_TRIM_RAD = 18
    DELT_TRIM_PCT = 19

class AuxIdxSlices():
    # Define slice objects for vector extraction
    CMD_SLICE = slice(AuxIdx.DELA_CMD_RAD, AuxIdx.DELT_CMD_PCT + 1)
    TRIM_SLICE = slice(AuxIdx.DELA_TRIM_RAD, AuxIdx.DELT_TRIM_PCT + 1)
    NAV_RATE_SLICE = slice(AuxIdx.P_NB_RPS, AuxIdx.R_NB_RPS + 1)
    FORCE_SLICE = slice(AuxIdx.FX_B_KGMPS2, AuxIdx.FZ_B_KGMPS2 + 1)
    MOMENT_SLICE = slice(AuxIdx.L_B_KGM2PS2, AuxIdx.N_B_KGM2PS2 + 1)
    WIND_SLICE = slice(AuxIdx.W_N_MPS, AuxIdx.W_D_MPS + 1)

# ------------------------------------------------------
# Trim State Vector Mapping
# ------------------------------------------------------

class TrimStateIdx(IntEnum):
    # Velocities
    U_B_MPS = 0
    V_B_MPS = 1
    W_B_MPS = 2
    
    # Body Rates
    P_B_RPS = 3
    Q_B_RPS = 4
    R_B_RPS = 5
    
    # Euler Angles
    PHI_RAD = 6
    THETA_RAD = 7
    PSI_RAD = 8
    
    # Position
    LAT_RAD = 9
    LONG_RAD = 10
    H_M = 11
    
    # Mass
    M_FUEL_KG = 12
    
    # Actuators
    DELA_TRIM_RAD = 13
    DELE_TRIM_RAD = 14
    DELR_TRIM_RAD = 15
    DELT_TRIM_PCT = 16

# Define slice objects for vector extraction
class TrimStateIdxSlices():
    VEL_SLICE = slice(TrimStateIdx.U_B_MPS, TrimStateIdx.W_B_MPS + 1)
    ROT_SLICE = slice(TrimStateIdx.P_B_RPS, TrimStateIdx.R_B_RPS + 1)
    ANGLE_SLICE = slice(TrimStateIdx.PHI_RAD, TrimStateIdx.PSI_RAD + 1)
    POS_SLICE = slice(TrimStateIdx.LAT_RAD, TrimStateIdx.H_M + 1)
    ACT_TRIM_SLICE = slice(TrimStateIdx.DELA_TRIM_RAD, TrimStateIdx.DELT_TRIM_PCT + 1)
    STATE_SLICE = slice(StateIdx.U_B_MPS, StateIdx.M_FUEL_KG + 1)