from scipy.integrate import odeint
from typing import Tuple
from loguru import logger
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


class SimulatedDataGenerator:
    """
    Generates simulated data for anomaly detection in CPS.

    This class creates data that simulate various fault conditions in a CPS.

    Attributes:
        component_b_lag (int): Time lag (in steps) between components A and B.
        tau (float): Time constant of the first-order system.
        taup (float): Time constant of the second-order system.
        du (float): Step height of the input signal.
        zeta (float): Damping factor of the second-order system.
        min_lenght_causal_phase (int): Minimum length of phases of the causal signal.
        max_lenght_causal_phase (int): Maximum length of phases of the causal signal.
        number_phases (int): Number of phases in the generated signals.
        seed (int): Seed value for random generation.
    """

    def __init__(
        self,
        min_lenght_causal_phase: int = 500,
        max_lenght_causal_phase: int = 1000,
        number_phases: int = 500,
        zeta: float = 0.3,
        du: float = 1.0,
        taup: float = 50,
        tau: float = 20.0,
        component_b_lag: int = 200,
        seed: int = 42,
    ) -> None:
        logger.info(f"Random seed in data generator set to {seed}")

        np.random.seed(seed)
        self.component_b_lag = component_b_lag
        self.tau = tau
        self.taup = taup
        self.du = du
        self.zeta = zeta
        self.min_lenght_causal_phase = min_lenght_causal_phase
        self.max_lenght_causal_phase = max_lenght_causal_phase
        self.number_phases = number_phases

    def run(self):
        """
        Executes the generation of simulated data.

        Generates and combines various signals representing the dynamic processes
        in components A and B of a CPS to create a complete dataset for analysis
        purposes.

        Returns:
            DataFrame: A Pandas DataFrame containing the generated time-series data.
        """
        (
            comp_a_signal,
            comp_b_signal,
            comp_a_durations,
            _,
            comp_a_kp_ls,
            _,
        ) = self.generate_causal_factor_signals()

        sig_a1, sig_a2, sig_a3 = self.generate_comp_a_signals(
            comp_a_signal=comp_a_signal,
            comp_a_durations=comp_a_durations,
            comp_a_kp_ls=comp_a_kp_ls,
        )
        sig_b1, sig_b2, sig_b3 = self.generate_comp_b_signals(
            comp_b_signal=comp_b_signal
        )
        min_len = min([len(sig) for sig in [comp_a_signal, comp_b_signal]])

        df = pd.DataFrame(
            dict(
                sig_a1=sig_a1[:min_len],
                sig_a2=sig_a2[:min_len],
                sig_a3=sig_a3[:min_len],
                sig_b1=sig_b1[:min_len],
                sig_b2=sig_b2[:min_len],
                sig_b3=sig_b3[:min_len],
                comp_a_signal=comp_a_signal[:min_len],
                comp_b_signal=comp_b_signal[:min_len],
            )
        )
        return df

    @staticmethod
    def get_comp_signal(comp_duration):
        """helper function: get list of gain values for dynamic system simulation"""
        comp_kp_ls = [-1]
        for i in range(1, len(comp_duration)):
            comp_kp_ls.append(-1 if comp_kp_ls[i - 1] == 1 else 1)
        return np.array(comp_kp_ls)

    @staticmethod
    def lag_signal(signal, lag):
        """
        Create a lagged version of the input signal.

        Parameters:
        - signal (numpy array): Input signal.
        - lag (int): The number of units to delay the signal by.

        Returns:
        - numpy array: Lagged signal.
        """

        # Create an array with the same length as the original signal.
        # Initial values are set to the first value of the original signal.
        lagged_signal = np.full(signal.shape, signal[0])

        # Set the delayed part of the signal
        lagged_signal[lag:] = signal[:-lag]

        return lagged_signal

    def generate_causal_factor_signals(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates causal signals and parameters for components A and B.

        Returns:
            Tuple of np.ndarrays: Contains the generated signals and parameters
            for components A and B. Specifically, this tuple includes the causal
            signals for components A and B, the durations of these signals, and
            key performance indicators for each component.
        """
        logger.info("Generating causal factors for comp a and b")
        comp_a_durations = np.random.randint(
            self.min_lenght_causal_phase,
            self.max_lenght_causal_phase,
            self.number_phases,
        )
        comp_b_durations = np.random.randint(
            self.min_lenght_causal_phase,
            self.max_lenght_causal_phase,
            self.number_phases,
        )

        comp_a_signal = np.concatenate(
            [
                [-1] * i if j % 2 == 0 else [1] * i
                for j, i in enumerate(comp_a_durations)
            ]
        )
        comp_b_signal = self.lag_signal(comp_a_signal, self.component_b_lag)
        comp_a_kp_ls = self.get_comp_signal(comp_a_durations)
        comp_b_kp_ls = self.get_comp_signal(comp_b_durations)
        return (
            comp_a_signal,
            comp_b_signal,
            comp_a_durations,
            comp_b_durations,
            comp_a_kp_ls,
            comp_b_kp_ls,
        )

    def second_order_model(self, x, _, Kp):
        """
        Represents a second-order dynamic system model.

        Args:
            x (list): A list containing the current value and the derivative of the system output.
            _ (ignored): Placeholder for an unused parameter.
            Kp (float): Proportional gain of the system.

        Returns:
            list: Derivatives of the system output.
        """
        y = x[0]
        dydt = x[1]
        dy2dt2 = (-2.0 * self.zeta * self.tau * dydt - y + Kp * self.du) / self.tau**2
        return [dydt, dy2dt2]

    def first_order_model(self, y, _, Kp):
        """
        Represents a first-order dynamic system model.

        Args:
            y (float): The current value of the system output.
            _ (ignored): Placeholder for an unused parameter.
            Kp (float): Proportional gain of the system.

        Returns:
            float: Derivative of the system output.
        """
        return (-y + Kp) / self.taup

    def generate_comp_a_signals(
        self,
        comp_a_signal: np.ndarray,
        comp_a_durations: np.ndarray,
        comp_a_kp_ls: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates signals for component A based on the causal factor A.

        This method integrates the system responses over time using the provided
        signals, durations, and performance indicators for component A.

        Args:
            comp_a_signal (np.ndarray): Array of the causal signal for component A.
            comp_a_durations (np.ndarray): Array of durations for each phase of component A's signal.
            comp_a_kp_ls (np.ndarray): Array of key performance indicators for component A.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Signals representing the response of component A's system.
        """

        logger.info("Computing component a singals")

        # (3) ODE Integrator

        x0_2nd = [-1, 0]
        x0_1st = [-1.1, 0]
        y_2nd_ls = []
        y_1st_ls = []

        for Kp, tmax in zip(comp_a_kp_ls, comp_a_durations):
            t = np.linspace(0, tmax, tmax)
            x_2nd = odeint(self.second_order_model, x0_2nd, t, (Kp,))
            y_2nd = x_2nd[:, 0]
            y_2nd_ls.append(y_2nd)
            x0_2nd = list(x_2nd[-1, :])

            x_1st = odeint(self.first_order_model, x0_1st, t, (Kp - 0.1,))
            y_1st = x_1st[:, 0]
            y_1st_ls.append(y_1st)
            x0_1st = list(x_1st[-1, :])

        sig_a1 = comp_a_signal
        sig_a2 = np.concatenate(y_2nd_ls)
        sig_a3 = np.concatenate(y_1st_ls)

        return sig_a1, sig_a2, sig_a3

    @staticmethod
    def generate_comp_b_signals(comp_b_signal: np.ndarray) -> tuple:
        """
        Generates signals for component B based on a lagged signal from component A.

        Args:
            comp_b_signal (np.ndarray): Array of the lagged signal from component A.

        Returns:
            tuple: A tuple containing three arrays (sig_b1, sig_b2, sig_b3)
            representing the processed signals for component B.
        """

        # (1) Direct lagged signal plus noise
        noise = np.random.normal(0, 0.1, len(comp_b_signal))
        sig_b1 = comp_b_signal + noise

        # (2) Apply low-pass filter
        b, a = butter(4, 0.01, "low")
        sig_b2 = filtfilt(b, a, comp_b_signal)

        # (3) Apply high-pass filter
        b, a = butter(4, 0.02, "high")
        sig_b3 = filtfilt(b, a, comp_b_signal) - 0.5

        return sig_b1, sig_b2, sig_b3
