import numpy as np

class KalmanFilter:
    """
    Algorithm from https://www.geeksforgeeks.org/python/kalman-filter-in-python/
    """
    def __init__(self, F, B, H, Q, R, x0, P0):
        """
        F: State transition matrix (system model).
        B: Control matrix (effect of control input).
        H: Observation matrix (how we measure the state).
        Q: Process noise covariance (uncertainty in the process).
        R: Measurement noise covariance (uncertainty in the measurements).
        x0: Initial state estimate.
        P0: Initial error covariance (initial uncertainty of state estimate).
        """
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, u):
        """
        Computes the predicted next state using the current state (self.x), the system model (F) and the control input (B, u).
        Updates the error covariance (self.P) to reflect increasing uncertainty as time passes.
        Returns the predicted state.
        """
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u) 
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        """
        S: Computes the residual covariance (expected combined uncertainty of prediction and measurement).
        K: Calculates the Kalman Gain (balance factor for blending prediction and measurement).
        y: Measurement residual (innovation), the difference between real measurement and prediction.
        self.x: Updates the state estimate with the innovation, weighted by Kalman Gain.
        I: Identity matrix (for maintaining shape during update).
        self.P: Updates the error covariance to show increased certainty after measurement.
        """
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x
    

if __name__ == "__main__":

    # Set up Kalman filter matrices
    F = np.array([[1, 1], [0, 1]])
    B = np.array([[0.5], [1]])
    H = np.array([[1, 0]])
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1]])

    # Initialize the filter state and uncertainty
    x0 = np.array([[0], [1]])
    P0 = np.array([[1, 0], [0, 1]])
    
    # Instance
    kf = KalmanFilter(F, B, H, Q, R, x0, P0)
    """
    u: Control input (acceleration), affects how position and velocity change.
    z: Measurement, the observed position from a sensor.
    kf.predict(u): Advances the state prediction using the model and control input.
    kf.update(z): Integrates the new measurement to refine the state estimate.
    Prints both the predicted and updated state for inspection.
    """
    u = np.array([[1]])
    z = np.array([[1]])

    predicted_state = kf.predict(u)
    print("Predicted state:\n", predicted_state)

    updated_state = kf.update(z)
    print("Updated state:\n", updated_state)