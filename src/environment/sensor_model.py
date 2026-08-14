import numpy as np
from src.engine.state_mapping import StateIdx
from src.utils.constants import D2R
from src.utils.math_utils import quat_to_dcm, quaternion_derivative

class SensorModel:
	"""
	Simulates imperfect hardware by mapping truth states to noisy measurements.
	"""
	def __init__(self, sensor_cfg, seed=None):
		self.enabled = sensor_cfg.get('enabled', False)
		self.cfg = sensor_cfg
		
		# Isolate the random state for the sensors to ensure Monte Carlo repeatability
		self.rng = np.random.default_rng(seed)

	def get_measurements(self, x_truth):
		"""
		Extracts states and applies configured Gaussian noise and biases.
		"""
		if not self.enabled:
			# Bypass mode: Measurements perfectly match truth
			return x_truth.copy()

		x_meas = x_truth.copy()

		# 1. IMU (Rates and Accelerations)
		imu_cfg = self.cfg.get('imu', {})
		bias = imu_cfg.get('gyro_bias_dps', 0.0)
		noise = imu_cfg.get('gyro_noise_std_dps', 0.0)

		x_meas[StateIdx.P_B_RPS] += (self.rng.normal(bias, noise) * D2R)
		x_meas[StateIdx.Q_B_RPS] += (self.rng.normal(bias, noise) * D2R)
		x_meas[StateIdx.R_B_RPS] += (self.rng.normal(bias, noise) * D2R)
		
		# 2. GPS (Position and Velocity)
		gps_cfg = self.cfg.get('gps', {})
		pos_noise = gps_cfg.get('pos_noise_std_m', 0.0)
		vel_noise = gps_cfg.get('vel_noise_std_mps', 0.0)

		x_meas[StateIdx.X_E_M : StateIdx.Z_E_M + 1] += self.rng.normal(0.0, pos_noise, 3)
		x_meas[StateIdx.U_B_MPS : StateIdx.W_B_MPS + 1] += self.rng.normal(0.0, vel_noise, 3)
		
		return x_meas

class NavigationFilter:
	"""
	Discrete Extended Kalman Filter (EKF) for 6DOF State Estimation.
	"""
	def __init__(self):
		self.n = StateIdx.NUM_STATES
		self.x_est = None
		
		# EKF Covariance Matrices
		self.P = np.eye(self.n) * 1.0     # State Covariance
		self.Q = np.eye(self.n) * 0.01    # Process Noise Covariance
		self.R = np.eye(self.n) * 0.1     # Measurement Noise Covariance

	def estimate_state(self, dt, x_meas):
		"""
		Executes the Predict and Update EKF steps.
		If called inside an RK4 sub-step where dt <= 0, it skips the update 
		to prevent covariance matrix corruption.
		"""
		if self.x_est is None:
			self.x_est = x_meas.copy()
			return self.x_est

		if dt <= 0:
			return self.x_est

		# --- 1. PREDICT ---
		# Predict using a local kinematic derivative model
		dx_pred = self._fsw_kinematic_predict(self.x_est)
		x_pred = self.x_est + dx_pred * dt
		
		# Finite difference Jacobian (F)
		F = self._compute_jacobian(self._fsw_kinematic_predict, self.x_est) * dt + np.eye(self.n)
		self.P = F @ self.P @ F.T + self.Q

		# --- 2. UPDATE ---
		H = np.eye(self.n)
		y = x_meas - x_pred
		
		S = H @ self.P @ H.T + self.R
		K = self.P @ H.T @ np.linalg.inv(S)
		
		self.x_est = x_pred + K @ y
		self.P = (np.eye(self.n) - K @ H) @ self.P
		
		# Quaternion Normalization
		q_norm = np.linalg.norm(self.x_est[StateIdx.Q0 : StateIdx.Q3 + 1])
		if q_norm > 0:
			self.x_est[StateIdx.Q0 : StateIdx.Q3 + 1] /= q_norm

		return self.x_est.copy()

	def _fsw_kinematic_predict(self, state):
		"""
		Internal Flight Software kinematic model for state propagation.
		Assumes constant body rates and velocities over the delta-t.
		"""
		dx = np.zeros_like(state)
		
		u, v, w = state[StateIdx.U_B_MPS : StateIdx.W_B_MPS + 1]
		p, q, r = state[StateIdx.P_B_RPS : StateIdx.R_B_RPS + 1]
		q0, q1, q2, q3 = state[StateIdx.Q0 : StateIdx.Q3 + 1]
		
		# Positional Kinematics
		C_b2e = quat_to_dcm(q0, q1, q2, q3)
		v_e = C_b2e @ np.array([u, v, w], dtype=np.float64)
		dx[StateIdx.X_E_M : StateIdx.Z_E_M + 1] = v_e
		
		# Rotational Kinematics
		omega_b = np.array([p, q, r], dtype=np.float64)
		dx[StateIdx.Q0 : StateIdx.Q3 + 1] = quaternion_derivative(state[StateIdx.Q0 : StateIdx.Q3 + 1], omega_b, 1.0)
		
		return dx

	def _compute_jacobian(self, func, x, eps=1e-4):
		n = len(x)
		J = np.zeros((n, n))
		for i in range(n):
			x_plus, x_minus = x.copy(), x.copy()
			x_plus[i] += eps
			x_minus[i] -= eps
			J[:, i] = (func(x_plus) - func(x_minus)) / (2 * eps)
		return J