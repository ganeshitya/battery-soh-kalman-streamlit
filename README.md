# Lithium-Ion Battery SOH Estimation (Kalman Filter Simulation)

## Overview

This project provides a simulation-based demonstration of how a Kalman Filter can be used to estimate the State of Health (SOH) of a Lithium-Ion battery. Unlike traditional methods that rely solely on direct measurements (which are often noisy and indirect for SOH), the Kalman Filter combines a predictive model of battery degradation with noisy sensor readings to provide a more accurate and robust SOH estimate.

The application is built using Streamlit, making it interactive and easy to experiment with different simulation and filter parameters.

## Why Kalman Filter for SOH?

Accurate SOH estimation is critical for:
* **Predicting Remaining Useful Life (RUL):** Crucial for warranties, maintenance, and replacement planning.
* **Optimizing Performance:** Knowing SOH helps in managing charging/discharging cycles effectively.
* **Ensuring Safety:** Severely degraded batteries can pose safety risks.

However, SOH is an internal, unobservable state. Direct measurements like voltage, current, and temperature are noisy and don't directly tell us the SOH. The Kalman Filter excels in such scenarios by:
* **Filtering Noise:** It effectively smooths out noisy sensor data.
* **Estimating Unobservable States:** It can provide an optimal estimate of a system's internal state (like SOH) even when it cannot be measured directly.
* **Combining Information:** It optimally combines a predictive model (how SOH *should* degrade) with actual, noisy observations.

## How the Simulation Works

The Streamlit application simulates the following:

1.  **True SOH Degradation:** A hypothetical, perfectly known SOH curve that degrades linearly over time/cycles. This acts as our "ground truth."
2.  **Noisy Measurements:** We generate simulated sensor readings by adding random Gaussian noise to the "True SOH." These represent the imperfect data a real battery management system (BMS) would receive.
3.  **Kalman Filter Estimation:** A Kalman Filter processes these noisy measurements. It uses:
    * An internal **prediction model** for SOH degradation (similar to the simulated true degradation).
    * The **noisy measurements** to correct its predictions.
    * **Noise covariance matrices (Q and R)** to weigh the trust between its prediction and the incoming measurements.

The result is a comparison plot showing the "True SOH," "Noisy Measurement," and the "Kalman Estimated SOH," demonstrating the filter's ability to track the true state despite measurement uncertainties.

## Features

* **Interactive Parameters:** Adjust initial SOH, degradation rate, number of cycles, and Kalman Filter noise parameters (Process Noise Q, Measurement Noise R, Initial Estimate Uncertainty P_0).
* **Real-time Visualization:** See how the Kalman Filter's estimate tracks the true SOH against noisy measurements.
* **Educational Tool:** Understand the impact of different noise levels on the filter's performance.

## Getting Started

### Prerequisites

* Python 3.7+

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/battery-soh-kalman-streamlit.git](https://github.com/YOUR_USERNAME/battery-soh-kalman-streamlit.git)
    cd battery-soh-kalman-streamlit
    ```
    (Replace `YOUR_USERNAME` with your actual GitHub username)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

After installation, run the Streamlit app from your terminal:

```bash
streamlit run streamlit_app.py
