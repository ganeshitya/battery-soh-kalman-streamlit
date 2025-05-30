import numpy as np

class KalmanFilter:
    def __init__(self, dt, state_dim, obs_dim):
        """
        Initializes the Kalman Filter.

        Args:
            dt (float): Time step (or cycle step) for the prediction.
            state_dim (int): Dimension of the state vector (e.g., 1 for SOH).
            obs_dim (int): Dimension of the measurement vector (e.g., 1 for a noisy SOH proxy).
        """
        self.dt = dt
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # State vector (x_k) - What we are trying to estimate
        # Initialize x_0 with a guess, e.g., [1.0] for SOH
        self.x = np.zeros((state_dim, 1))

        # State transition matrix (A) - How the state evolves over time
        # For a simple SOH model, A might be an identity matrix if SOH
        # is expected to be constant unless acted upon by degradation.
        # Here, we'll incorporate degradation directly into the prediction.
        self.A = np.eye(state_dim) # State remains the same unless processed

        # Control input matrix (B) - Not explicitly used for passive SOH estimation here,
        # but useful if external inputs (like charge current) directly affect SOH.
        self.B = np.zeros((state_dim, 1)) # No explicit control input for simple SOH

        # Measurement matrix (H) - How measurements relate to the state
        # If we directly measure a noisy SOH, H is identity.
        self.H = np.eye(obs_dim, state_dim) # Assuming observation directly relates to state

        # Process noise covariance (Q) - Uncertainty in the state transition model
        # Higher Q means more uncertainty in how SOH degrades, so filter relies more on measurements.
        self.Q = np.eye(state_dim) * 0.0001 # Small initial value

        # Measurement noise covariance (R) - Uncertainty in the measurements
        # Higher R means measurements are very noisy, so filter relies more on its prediction.
        self.R = np.eye(obs_dim) * 0.1 # Small initial value

        # Error covariance matrix (P) - Uncertainty in our state estimate
        # Initialize with a high value, as our initial estimate is uncertain.
        self.P = np.eye(state_dim) * 1.0 # High initial uncertainty

        # Identity matrix (I) - Used in update step
        self.I = np.eye(state_dim)

    def predict(self, u=0):
        """
        Predicts the next state and error covariance.

        Args:
            u (float): Control input (not actively used in this simple SOH model, defaults to 0).
        """
        # Predict the state: x_k = A * x_{k-1} + B * u_k
        # For SOH, we model degradation directly.
        # Here, A is Identity, so we'll adjust x directly in streamlit_app for degradation,
        # or you could define a more complex A matrix if SOH degradation is part of it.
        # For a simple setup, assume 'A' represents no intrinsic change,
        # and 'process_noise' (Q) accounts for the unpredictability of degradation.
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u) # Simplified: A might be identity here if decay is handled separately
        # Predict the error covariance: P_k = A * P_{k-1} * A_T + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        """
        Updates the state estimate based on a new measurement.

        Args:
            z (np.array): The measurement vector (noisy SOH proxy).
        """
        # Kalman Gain: K_k = P_k * H_T * (H_k * P_k * H_T + R_k)^(-1)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update the state estimate: x_k = x_k + K_k * (z_k - H_k * x_k)
        y = z - np.dot(self.H, self.x) # Measurement residual
        self.x = self.x + np.dot(K, y)

        # Update the error covariance: P_k = (I - K_k * H_k) * P_k
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)

        return self.x
