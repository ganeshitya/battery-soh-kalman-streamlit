import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter # Import your KalmanFilter class

st.set_page_config(layout="wide")

st.title("🔋 Lithium-Ion Battery SOH Estimation (Kalman Filter Simulation)")

st.write("""
This application simulates the degradation of a Lithium-Ion battery's State of Health (SOH)
and demonstrates how a Kalman Filter can estimate the true SOH from noisy measurements.

The 'True SOH' represents the actual, underlying battery degradation.
'Noisy Measurement' simulates what a sensor might observe, which is the True SOH plus random noise.
The 'Kalman Estimated SOH' is the filter's best guess of the True SOH,
combining its internal model of degradation with the noisy measurements.
""")

st.sidebar.header("Simulation Parameters")

# User inputs for simulation
initial_soh = st.sidebar.slider("Initial SOH (%)", 90, 100, 100) / 100.0 # Convert to fraction
cycles = st.sidebar.slider("Number of Cycles / Time Steps", 50, 1000, 500)
soh_degradation_per_cycle = st.sidebar.slider("SOH Degradation per Cycle (%)", 0.001, 0.05, 0.005, format="%.3f") / 100.0

st.sidebar.header("Kalman Filter Parameters")

# Kalman Filter parameters (noise levels)
process_noise_std = st.sidebar.slider("Process Noise (Q) Standard Deviation", 0.0001, 0.01, 0.001, format="%.4f")
measurement_noise_std = st.sidebar.slider("Measurement Noise (R) Standard Deviation", 0.001, 0.1, 0.02, format="%.3f")
initial_estimate_uncertainty = st.sidebar.slider("Initial Estimate Uncertainty (P_0)", 0.1, 5.0, 1.0, format="%.1f")

if st.sidebar.button("Run Simulation"):
    st.subheader("Simulation Results")

    # --- 1. Simulate True SOH Degradation ---
    true_soh_history = []
    current_true_soh = initial_soh

    for i in range(cycles):
        true_soh_history.append(current_true_soh)
        # Simulate degradation: SOH decreases slightly each cycle
        current_true_soh = max(0.0, current_true_soh - soh_degradation_per_cycle)

    true_soh_history = np.array(true_soh_history)

    # --- 2. Generate Noisy Measurements ---
    # Measurements are the true SOH plus random Gaussian noise
    measurement_noise = np.random.normal(0, measurement_noise_std, cycles)
    noisy_measurements = true_soh_history + measurement_noise
    # Ensure measurements don't go out of sensible bounds (e.g., >1 or <0)
    noisy_measurements = np.clip(noisy_measurements, 0.0, 1.0)


    # --- 3. Initialize and Run Kalman Filter ---
    # State: SOH (1 dimension)
    # Observation: Noisy SOH (1 dimension)
    kf = KalmanFilter(dt=1, state_dim=1, obs_dim=1)

    # Set initial state and uncertainty
    kf.x = np.array([[initial_soh]]) # Initial SOH estimate
    kf.P = np.array([[initial_estimate_uncertainty]]) # Initial uncertainty in SOH estimate

    # Set noise covariance matrices based on user input
    kf.Q = np.eye(1) * (process_noise_std**2)
    kf.R = np.eye(1) * (measurement_noise_std**2)

    kalman_estimated_soh = []

    for i in range(cycles):
        # Kalman Filter Prediction Step
        # Here, the prediction model for SOH should also reflect the expected degradation.
        # A simple way to integrate degradation into the KF's prediction:
        # Instead of just A*x, subtract the expected degradation.
        # Note: This is a simplification. A more robust model would incorporate degradation into A or B.
        # For this demo, we'll let Q handle the "unpredictable" degradation that the KF has to learn.
        kf.predict() # Predicts the next state based on its internal model

        # Simulate the SOH degradation for the KF's *prediction*
        # The KF's internal model of how SOH should degrade
        kf.x[0, 0] = max(0.0, kf.x[0, 0] - soh_degradation_per_cycle)


        # Kalman Filter Update Step with the noisy measurement
        measurement = np.array([[noisy_measurements[i]]])
        estimated_soh_state = kf.update(measurement)
        kalman_estimated_soh.append(estimated_soh_state[0, 0])

    kalman_estimated_soh = np.array(kalman_estimated_soh)

    # --- 4. Plotting Results ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(true_soh_history * 100, label='True SOH (%)', color='blue', linewidth=2)
    ax.plot(noisy_measurements * 100, label='Noisy Measurement (%)', color='red', linestyle='--', alpha=0.6)
    ax.plot(kalman_estimated_soh * 100, label='Kalman Estimated SOH (%)', color='green', linewidth=2.5)

    ax.set_title("SOH Estimation Over Cycles")
    ax.set_xlabel("Cycles / Time Steps")
    ax.set_ylabel("State of Health (%)")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(0, 105) # Ensure Y-axis goes from 0 to 100+ for SOH

    st.pyplot(fig)

    st.subheader("How the Kalman Filter Works Here:")
    st.markdown("""
    1.  **Simulated True SOH:** We define a ground truth for how the battery's SOH degrades over cycles.
    2.  **Simulated Noisy Measurements:** We generate sensor readings by adding random noise to the True SOH. These are the "observations" the Kalman Filter receives.
    3.  **Kalman Filter Prediction:** The filter predicts the next SOH based on its last estimate and an internal model of how SOH generally degrades (in this simple case, by subtracting `soh_degradation_per_cycle` from its current estimate). It also predicts its uncertainty.
    4.  **Kalman Filter Update:** When a new noisy measurement comes in, the filter compares it to its prediction. It then adjusts its SOH estimate, giving more weight to the prediction if the measurement is very noisy (high R) or more weight to the measurement if its prediction is very uncertain (high P or Q).
    5.  **Result:** The Kalman Estimated SOH is typically much smoother and closer to the True SOH than the raw, noisy measurements, demonstrating its ability to filter noise and provide an optimal estimate.
    """)

    st.subheader("Experiment with Parameters:")
    st.markdown("""
    * **High Measurement Noise (R):** The Kalman Filter will rely more on its prediction model, leading to a smoother but potentially slower-to-react estimate.
    * **High Process Noise (Q):** The filter will assume more uncertainty in its internal degradation model, making it more responsive to measurements but also more susceptible to measurement noise.
    * **Initial Estimate Uncertainty (P_0):** A high initial uncertainty tells the filter to trust the first few measurements more heavily. As it processes more data, this uncertainty reduces.
    """)
