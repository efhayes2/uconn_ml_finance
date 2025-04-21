import numpy as np
import math
import pandas as pd
from dataclasses import dataclass, field
# Import numba
from numba import jit, float64, int64 # Import specific types for signature if using nopython=True
import time # Import time for timing simulation generation

# --- Dataclass for Simulation Parameters ---
@dataclass
class OUSimParameters:
    """
    Holds all input parameters for the OU simulation.
    Derived parameters like k, dt, num_steps, initial_x
    are calculated from these inputs.
    """
    # Price parameters
    equilibrium_price: float = 50.0
    initial_price: float = 50.0

    # OU process parameters
    half_life_years: float = 5.0 # Used to calculate k
    sigma: float = 0.10          # Volatility

    # Time parameters
    days_per_year: int = 260     # Trading days per year
    total_years: int = 5         # Total simulation duration

    # Simulation parameters
    num_paths: int = 1000        # Number of sample paths to generate
    save_prices_to_csv: bool = False # Option to save prices to CSV

    # Derived parameters (calculated in function/main)
    k: float = field(init=False)
    dt: float = field(init=False)
    num_steps: int = field(init=False)
    initial_x: float = field(init=False)

    def __post_init__(self):
        """Calculate derived parameters after initialization."""
        self.k = math.log(2) / self.half_life_years
        self.dt = 1.0 / self.days_per_year
        self.num_steps = self.total_years * self.days_per_year
        self.initial_x = math.log(self.initial_price / self.equilibrium_price)


# --- OUProcess Class ---
# Represents the parameters of the Ornstein-Uhlenbeck process
class OUProcess:
    """
    Represents an Ornstein-Uhlenbeck process defined by:
    dx = -k(x - mu) dt + sigma dZ
    In this specific case, mu = 0 based on the user's SDE form.
    """
    def __init__(self, k: float, sigma: float, mu: float = 0.0):
        """
        Initializes the OUProcess with parameters.

        Args:
            k (float): Rate of mean reversion (must be > 0).
            sigma (float): Volatility (must be > 0).
            mu (float): Long-term mean of the process (default is 0).
        """
        if k <= 0:
            raise ValueError("Mean reversion rate 'k' must be positive.")
        if sigma <= 0:
            raise ValueError("Volatility 'sigma' must be positive.")

        self.k = k
        self.sigma = sigma
        self.mu = mu # Mean reversion level

    def __str__(self):
        return f"OU Process: dx = -{self.k:.4f}(x - {self.mu:.4f}) dt + {self.sigma:.4f} dZ"

# --- OUPathGenerator Class ---
# Generates sample paths for a given OU process
class OUPathGenerator:
    """
    Generates sample paths for a given Ornstein-Uhlembeck process
    using the Euler-Maruyama method.
    """
    def __init__(self, ou_process: OUProcess, initial_x: float, dt: float, num_steps: int):
        """
        Initializes the path generator.

        Args:
            ou_process (OUProcess): An instance of the OUProcess class.
            initial_x (float): The starting value for the process (x_0).
            dt (float): The time step size.
            num_steps (int): The total number of steps to simulate.
        """
        if not isinstance(ou_process, OUProcess):
            raise TypeError("ou_process must be an instance of OUProcess.")
        if num_steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if dt <= 0:
             raise ValueError("Time step 'dt' must be positive.")

        self.ou_process = ou_process
        self.initial_x = initial_x
        self.dt = dt
        self.num_steps = num_steps
        self.sqrt_dt = math.sqrt(dt) # Pre-calculate sqrt(dt) for efficiency

    def generate_path(self):
        """
        Generates a single sample path (not Numba optimized).
        Included for completeness, but generate_multiple_paths is used for batch sim.
        """
        path = np.zeros(self.num_steps + 1)
        path[0] = self.initial_x # Set the initial value

        # Simulate using Euler-Maruyama: x_{t+dt} = x_t + k(mu - x_t)dt + sigma * sqrt(dt) * epsilon
        # Where epsilon is a standard normal random variable
        for i in range(self.num_steps):
            current_x = path[i]
            drift = self.ou_process.k * (self.ou_process.mu - current_x) * self.dt
            diffusion = self.ou_process.sigma * self.sqrt_dt * np.random.randn() # sigma * sqrt(dt) * N(0,1)
            path[i+1] = current_x + drift + diffusion

        return path

    # --- Numba-optimized core generation function ---
    # Apply numba JIT compilation to this static method for speed
    # Using nopython=True requires all operations inside to be Numba compatible.
    # We pass necessary scalar values and arrays directly.
    @staticmethod
    @jit(nopython=True) # Use nopython=True for best performance
    def _generate_multiple_paths_jit(
        num_paths: int64,
        num_steps: int64,
        initial_x: float64,
        k: float64,
        sigma: float64,
        mu: float64, # Mean reversion level
        dt: float64,
        sqrt_dt: float64,
        random_shocks: np.ndarray # Shape (num_paths, num_steps) - Numba understands NumPy arrays
    ) -> np.ndarray:
        """
        Numba-optimized core function to generate multiple OU paths.
        Simulates paths vectorially over steps using a Python loop compiled by Numba.
        """
        paths = np.zeros((num_paths, num_steps + 1), dtype=np.float64) # Ensure float64 dtype
        paths[:, 0] = initial_x # Set the initial value for all paths

        # Simulate paths vectorially over steps
        # This loop is what Numba will accelerate
        for i in range(num_steps):
            current_x = paths[:, i] # Get the values for all paths at the current step
            drift = k * (mu - current_x) * dt
            diffusion = sigma * sqrt_dt * random_shocks[:, i] # Apply corresponding random shocks
            paths[:, i+1] = current_x + drift + diffusion

        return paths
    # -------------------------------------------------


    def generate_multiple_paths(self, num_paths: int) -> np.ndarray:
        """
        Generates multiple sample paths using the Numba-optimized core function.

        Args:
            num_paths (int): The number of paths to generate.

        Returns:
            numpy.ndarray: A 2D array of shape (num_paths, num_steps + 1)
                           containing all generated paths.
        """
        if num_paths <= 0:
             raise ValueError("Number of paths must be positive.")

        # Generate all random normal variables needed at once
        # This part runs in standard Python/NumPy, but is usually fast.
        # Ensure dtype matches the Numba function's expectation
        random_shocks = np.random.randn(num_paths, self.num_steps).astype(np.float64)

        # Call the numba-optimized core function
        # Pass all necessary parameters explicitly from self and self.ou_process
        paths = self._generate_multiple_paths_jit(
            num_paths,
            self.num_steps,
            self.initial_x,
            self.ou_process.k,
            self.ou_process.sigma,
            self.ou_process.mu,
            self.dt,
            self.sqrt_dt,
            random_shocks
        )

        return paths

# --- Simulation Function ---
def generate_ou_paths_and_prices(params: OUSimParameters) -> pd.DataFrame:
    """
    Generates OU paths (x_t), converts to prices (p_t), and optionally saves prices to CSV.

    Args:
        params (OUSimParameters): Instance of OUSimParameters dataclass.

    Returns:
        pd.DataFrame: DataFrame containing the generated price paths (p_t).
    """
    # 1. Create an instance of the OUProcess
    # Parameters k, sigma, mu are needed for the OUProcess instance
    ou_process_instance = OUProcess(k=params.k, sigma=params.sigma, mu=0.0) # mu=0 based on dx = -kx dt

    # 2. Create an instance of the Path Generator
    # Parameters initial_x, dt, num_steps are needed for the Path Generator instance
    path_generator = OUPathGenerator(
        ou_process=ou_process_instance,
        initial_x=params.initial_x,
        dt=params.dt,
        num_steps=params.num_steps
    )

    # 3. Generate the sample paths for x_t using the (now Numba-optimized) generator
    print(f"Generating {params.num_paths} sample paths for {params.total_years} years ({params.num_steps} daily steps)...")
    # Add timing for simulation generation
    sim_start_time = time.time()
    sample_paths_x = path_generator.generate_multiple_paths(params.num_paths)
    sim_end_time = time.time()
    print(f"Simulation generation complete in {sim_end_time - sim_start_time:.2f} seconds.")

    # The 'sample_paths_x' variable now holds a NumPy array of shape (num_paths, num_steps + 1)
    # Each row is a sample path of x_t over time.
    # The columns represent time steps from t=0 to t=num_steps.

    # --- Convert x_t paths to price p_t paths ---
    # Use the relationship p_t = p_e * exp(x_t)
    # This conversion is also a good candidate for numba if it becomes a bottleneck,
    # but np.exp is usually fast.
    price_paths_p = params.equilibrium_price * np.exp(sample_paths_x)
    print("Converted x_t paths to price p_t paths.")
    # ------------------------------------------

    # --- Optionally save the generated p_t paths to a CSV file using pandas ---
    if params.save_prices_to_csv:
        p_csv_filename = 'ou_prices.csv'
        # Convert NumPy array to pandas DataFrame
        price_df = pd.DataFrame(price_paths_p)
        # Save to CSV, skipping index and header, formatting floats
        price_df.to_csv(p_csv_filename, index=False, header=False, float_format='%.3f')
        print(f"Sample paths (p_t) saved to {p_csv_filename} with 3 decimal places.")
    # -------------------------------------------------------------------------

    # Return the price paths as a pandas DataFrame
    return pd.DataFrame(price_paths_p)


# --- Main Execution ---
# This block runs only when the script is executed directly
if __name__ == "__main__":
    # Define simulation parameters with save_prices_to_csv=True for direct run
    sim_params = OUSimParameters(save_prices_to_csv=True)

    # Generate and save the paths
    generated_price_paths_df = generate_ou_paths_and_prices(sim_params)

    # Example usage after generation (optional)
    # print(f"\nShape of generated price paths DataFrame: {generated_price_paths_df.shape}")
    # print("First 5 rows of the first 5 steps:")
    # print(generated_price_paths_df.iloc[:5, :5])
