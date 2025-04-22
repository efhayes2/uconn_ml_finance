from __future__ import annotations # Allows forward references for type hints
import numpy as np
import math
import os # Import os for checking file existence
# Import the RLParameters dataclass definition is NOT needed here anymore
# because of from __future__ import annotations.
# The definition of RLParameters must be in the file that imports TradingUtils.

# Need to import the actual RLParameters type for type checking tools,
# but using __future__.annotations avoids the runtime circular import issue.
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from ou_rl_trader import RLParameters # Import for type checking only

class TradingUtils:
    """
    A collection of static utility methods for trading calculations and data handling.
    """

    @staticmethod
    # Use 'RLParameters' as a string or rely on __future__.annotations
    def calculate_costs(delta_n: int, rl_params: 'RLParameters') -> tuple[float, float, float]:
        """
        Calculates spread, impact, and total transaction costs for a trade.

        Args:
            delta_n (int): The intended change in holding (trade size).
            rl_params (RLParameters): Instance of RLParameters dataclass
                                       containing cost parameters (trans_cost_in_ticks, lot_size).

        Returns:
            tuple[float, float, float]: (spread_cost, impact_cost, total_cost)
        """
        # Use trans_cost_in_ticks for cost calculation amount
        spread_cost = rl_params.trans_cost_in_ticks * abs(delta_n)
        impact_cost = (delta_n)**2 * rl_params.trans_cost_in_ticks / rl_params.lot_size
        total_cost = spread_cost + impact_cost
        return spread_cost, impact_cost, total_cost

    @staticmethod
    # Use 'RLParameters' as a string or rely on __future__.annotations
    def clip_holding(holding: int, rl_params: 'RLParameters') -> int:
        """
        Clips a holding amount to the nearest allowed holding level defined in RLParameters.

        Args:
            holding (int): The intended holding amount.
            rl_params (RLParameters): Instance of RLParameters dataclass
                                       containing the allowed holdings space.

        Returns:
            int: The clipped holding amount, which is one of the allowed holdings.
        """
        allowed_holdings = np.array(rl_params.holdings_space)
        # Find the index of the nearest allowed holding
        nearest_idx = np.abs(allowed_holdings - holding).argmin()
        return int(allowed_holdings[nearest_idx])

    @staticmethod
    def save_q_table(q_table: np.ndarray, filename: str):
        """
        Saves the Q-table to a CSV file.

        Args:
            q_table (np.ndarray): The Q-table to save.
            filename (str): The name of the file to save to.
        """
        try:
            np.savetxt(filename, q_table, delimiter=',')
            print(f"Q-table saved to {filename}")
        except Exception as e:
            print(f"Error saving Q-table to {filename}: {e}")

    @staticmethod
    def load_q_table(filename: str, expected_shape: tuple[int, int]) -> np.ndarray | None:
        """
        Loads a Q-table from a CSV file.

        Args:
            filename (str): The name of the file to load from.
            expected_shape (tuple[int, int]): The expected shape of the loaded Q-table
                                              (state_space_size, num_actions).

        Returns:
            np.ndarray | None: The loaded Q-table if successful and shape matches,
                               otherwise None.
        """
        if not os.path.exists(filename):
            print(f"Q-table file not found: {filename}")
            return None

        try:
            q_table = np.loadtxt(filename, delimiter=',')
            # Ensure the loaded table is 2D even if it was 1D (e.g., 1 row)
            if q_table.ndim == 1:
                 q_table = q_table.reshape(1, -1)

            if q_table.shape == expected_shape:
                print(f"Q-table loaded successfully from {filename}")
                return q_table
            else:
                print(f"Q-table shape mismatch in {filename}. Expected {expected_shape}, got {q_table.shape}. Starting with new table.")
                return None
        except Exception as e:
            print(f"Error loading Q-table from {filename}: {e}. Starting with new table.")
            return None

