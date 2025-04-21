import numpy as np
import pandas as pd
import math
import random
import time # Import the time module
from dataclasses import dataclass, field
# Note: This version loads from CSV, it does NOT import from ou_simulation_code

# --- RL Parameters Dataclass ---
@dataclass
class RLParameters:
    """
    Holds parameters for the Q-learning algorithm and environment.
    """
    # Trading parameters
    tick_size: float = 0.10
    lot_size: int = 100
    k_action_range: int = 5 # Action space is lot_size * [-k_action_range, ..., k_action_range]
    m_holdings_range: int = 10 # Allowed holdings are lot_size * [-m_holdings_range, ..., m_holdings_range]
    price_levels: int = 1000 # Number of discrete price levels

    # Cost parameters
    spread_cost_per_lot: float = field(init=False) # Calculated from tick_size
    k_star_risk_param: float = 10e-4 # Risk aversion parameter for reward

    # Q-learning parameters
    alpha: float = 0.001 # Learning rate
    epsilon: float = 0.1 # Exploration rate
    gamma: float = 0.95  # Discount factor (default, not specified by user)

    # Training parameters
    print_frequency: int = 100 # How often to print episode progress
    # Note: This version does not use replay buffer or batch size parameters directly


    def __post_init__(self):
        """Calculate derived parameters after initialization."""
        self.spread_cost_per_lot = self.tick_size * self.lot_size # Spread cost is tick_size * abs(delta_n_t) -> spread cost per lot is tick_size * lot_size

    @property
    def action_space(self):
        """Defines the possible trade sizes (delta_n_t)."""
        return [i * self.lot_size for i in range(-self.k_action_range, self.k_action_range + 1)]

    @property
    def holdings_space(self):
        """Defines the allowed holding sizes (n_t)."""
        return [i * self.lot_size for i in range(-self.m_holdings_range, self.m_holdings_range + 1)]

    @property
    def state_space_size(self):
        """Calculates the total number of discrete states."""
        return self.price_levels * len(self.holdings_space)

# --- Trading Environment Class ---
class TradingEnvironment:
    """
    Represents the trading environment based on price paths.
    Handles state transitions, costs, and rewards.
    """
    def __init__(self, price_paths_df: pd.DataFrame, rl_params: RLParameters):
        """
        Initializes the trading environment.

        Args:
            price_paths_df (pd.DataFrame): DataFrame containing price paths.
                                          Each row is a path, columns are time steps.
            rl_params (RLParameters): Instance of RLParameters dataclass.
        """
        self.price_paths = price_paths_df.values # Use numpy array for efficiency
        self.num_paths, self.num_steps = self.price_paths.shape # num_steps is total steps - 1
        self.num_steps -= 1 # Adjust to be number of transitions

        self.rl_params = rl_params

        # Define discrete price levels and holding levels based on params
        # Price levels are tick_size * {1, ..., 1000}, capped/floored outside this range
        self.min_price = self.rl_params.tick_size * 1
        self.max_price = self.rl_params.tick_size * self.rl_params.price_levels # 0.10 * 1000 = 100.00

        # Map holding values to indices
        self._holding_to_index = {h: i for i, h in enumerate(self.rl_params.holdings_space)}
        self._index_to_holding = {i: h for i, h in enumerate(self.rl_params.holdings_space)}

        # Map action values to indices
        self._action_to_index = {a: i for i, a in enumerate(self.rl_params.action_space)}
        self._index_to_action = {i: a for i, a in enumerate(self.rl_params.action_space)}

        # State variables (will be set in reset)
        self.current_path_idx = -1
        self.current_step_idx = -1
        self.current_holding = 0 # Start with 0 shares
        self.cumulative_value = 0.0 # Start with 0 portfolio value
        self.current_state_index = -1 # Discrete state index

    def _discretize_price(self, price: float) -> int:
        """
        Discretizes a continuous price into an integer index [0, price_levels-1].
        Prices below min_price are floored to min_price bin.
        Prices above max_price are capped to max_price bin.
        """
        # Cap/floor price to the defined range
        clipped_price = max(self.min_price, min(self.max_price, price))
        # Calculate index: (clipped_price / tick_size) - 1
        # Round to nearest tick, then convert to index
        price_index = int(round(clipped_price / self.rl_params.tick_size)) - 1
        # Ensure index is within [0, price_levels - 1]
        return max(0, min(self.rl_params.price_levels - 1, price_index))

    def _discretize_holding(self, holding: int) -> int:
        """
        Discretizes a continuous holding into an integer index [0, len(holdings_space)-1].
        Assumes holding is already a multiple of lot_size.
        If not exactly in the space, finds the nearest allowed holding.
        """
        # If the holding is exactly one of the allowed values, return its index directly.
        if holding in self._holding_to_index:
            return self._holding_to_index[holding]

        # If not exactly in the allowed space (should ideally not happen with correct clipping),
        # find the index of the nearest allowed holding and return its index.
        allowed_holdings = np.array(self.rl_params.holdings_space)
        nearest_holding = allowed_holdings[np.abs(allowed_holdings - holding).argmin()]
        return self._holding_to_index[nearest_holding]


    def _get_state_index(self, price_index: int, holding: int) -> int:
        """
        Maps discrete price index and continuous holding to a single state index.
        Holding is discretized internally.
        State = (price_index, holding_index)
        State Index = price_index * size_of_holdings_space + holding_index
        """
        holding_index = self._discretize_holding(holding)
        state_index = price_index * len(self.rl_params.holdings_space) + holding_index
        return state_index

    def _get_state_tuple(self, state_index: int) -> tuple[int, int]:
        """
        Maps a single state index back to (price_index, holding_index).
        Useful for debugging/analysis.
        """
        size_of_holdings_space = len(self.rl_params.holdings_space)
        price_index = state_index // size_of_holdings_space
        holding_index = state_index % size_of_holdings_space
        holding = self._index_to_holding[holding_index] # Get continuous holding value
        return price_index, holding # Return price index and continuous holding value


    def reset(self, path_index: int):
        """
        Resets the environment to the start of a specified price path.

        Args:
            path_index (int): The index of the price path to use for the episode.

        Returns:
            int: The initial state index.
        """
        if not 0 <= path_index < self.num_paths:
            raise ValueError(f"path_index must be between 0 and {self.num_paths - 1}")

        self.current_path_idx = path_index
        self.current_step_idx = 0 # Start at the first price (index 0)
        self.current_holding = 0  # Start with no shares
        self.cumulative_value = 0.0 # Start with zero portfolio value

        # Initial state is (price at step 0, holding at step -1 which is 0)
        initial_price = self.price_paths[self.current_path_idx, self.current_step_idx]
        initial_price_index = self._discretize_price(initial_price)
        # State is (current price, previous holding) -> (p_0, n_{-1})
        self.current_state_index = self._get_state_index(initial_price_index, self.current_holding) # Holding at t-1 is 0

        return self.current_state_index

    def step(self, action_index: int) -> tuple[int, float, bool]:
        """
        Takes an action in the environment.

        Args:
            action_index (int): The index of the action to take.

        Returns:
            tuple: (next_state_index, reward, done)
        """
        if not 0 <= action_index < len(self.rl_params.action_space):
             raise ValueError(f"action_index must be between 0 and {len(self.rl_params.action_space) - 1}")

        # Get action value (delta_n_t)
        delta_n_t = self._index_to_action[action_index]

        # Get current price p_t and next price p_{t+1}
        p_t = self.price_paths[self.current_path_idx, self.current_step_idx]

        # Check if it's the last step
        done = self.current_step_idx >= self.num_steps - 1

        if done:
            # If it's the last step, no price change to calculate reward
            # Agent might want to liquidate position here.
            # For simplicity in this step function, we'll just return 0 reward and done.
            # A more complex environment might handle terminal state rewards/penalties.
            next_state_index = self.current_state_index # Stay in the last state
            reward = 0.0 # No reward from price change at the very last step
            # Optional: Add liquidation penalty/reward here
            # e.g., if self.current_holding != 0: reward -= abs(self.current_holding) * self.rl_params.tick_size
        else:
            p_t_plus_1 = self.price_paths[self.current_path_idx, self.current_step_idx + 1]

            # Calculate the new holding n_t = n_{t-1} + delta_n_t
            # Clip the resulting holding to the allowed range H
            intended_n_t = self.current_holding + delta_n_t
            n_t = self.clip_holding(intended_n_t) # Use the clip_holding method

            # Calculate costs based on the *intended* trade size delta_n_t
            spread_cost_t = self.rl_params.tick_size * abs(delta_n_t)
            impact_cost_t = (delta_n_t)**2 * self.rl_params.tick_size / self.rl_params.lot_size
            total_cost_t = spread_cost_t + impact_cost_t

            # Calculate change in value (delta_v_t_plus_1) based on user's definition structure
            # This represents the gain/loss from holding n_t shares from t to t+1, minus the cost incurred at t
            delta_v_t_plus_1 = n_t * (p_t_plus_1 - p_t) - total_cost_t

            # Calculate the reward r_{t+1} based on user's formula
            reward = delta_v_t_plus_1 - 0.5 * self.rl_params.k_star_risk_param * (delta_v_t_plus_1)**2

            # Update current holding and step index
            self.current_holding = n_t
            self.current_step_idx += 1

            # Determine the next state s_{t+1} = (p_{t+1}, n_t)
            next_price = self.price_paths[self.current_path_idx, self.current_step_idx] # This is p_{t+1}
            next_price_index = self._discretize_price(next_price)
            next_state_index = self._get_state_index(next_price_index, self.current_holding)

            # Update the current state index for the next step
            self.current_state_index = next_state_index


        # Return next state, reward, and done flag
        return next_state_index, reward, done

    def clip_holding(self, holding: int) -> int:
        """Clips a holding amount to the nearest allowed holding level."""
        allowed_holdings = np.array(self.rl_params.holdings_space)
        # Find the index of the nearest allowed holding
        nearest_idx = np.abs(allowed_holdings - holding).argmin()
        return int(allowed_holdings[nearest_idx])


# --- Tabular Q-Learning Agent Class ---
class TabularQLearner:
    """
    Implements a tabular Q-learning agent (sequential updates).
    """
    def __init__(self, state_space_size: int, action_space: list[int], rl_params: RLParameters):
        """
        Initializes the Q-learner.

        Args:
            state_space_size (int): The total number of discrete states.
            action_space (list[int]): A list of possible action values (delta_n_t).
            rl_params (RLParameters): Instance of RLParameters dataclass.
        """
        self.state_space_size = state_space_size
        self.action_space = action_space
        self.num_actions = len(action_space)
        self.rl_params = rl_params

        # Initialize Q-table with zeros
        self.q_table = np.zeros((self.state_space_size, self.num_actions))

        # Map action values to indices (needed for Q-table lookup)
        self._action_to_index = {a: i for i, a in enumerate(self.rl_params.action_space)}
        self._index_to_action = {i: a for i, a in enumerate(self.rl_params.action_space)}


    def choose_action(self, state_index: int) -> int:
        """
        Chooses an action using an epsilon-greedy policy.

        Args:
            state_index (int): The index of the current state.

        Returns:
            int: The index of the chosen action.
        """
        # Explore: Choose a random action with probability epsilon
        if random.uniform(0, 1) < self.rl_params.epsilon:
            action_index = random.randrange(self.num_actions)
        # Exploit: Choose the action with the highest Q-value for the current state
        else:
            # Get Q-values for the current state across all actions
            q_values_current_state = self.q_table[state_index, :]
            # Choose the index of the action with the maximum Q-value
            # Use argmax. If multiple actions have the same max value, argmax returns the first one.
            # Add a small random noise to break ties randomly if needed, though argmax default is often fine.
            action_index = np.argmax(q_values_current_state)

        return action_index

    def learn(self, state_index: int, action_index: int, reward: float, next_state_index: int):
        """
        Updates the Q-table using the Q-learning algorithm (sequential update).

        Q(s,a) <- Q(s,a) + alpha [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Args:
            state_index (int): Index of the current state (s).
            action_index (int): Index of the action taken (a).
            reward (float): The reward received (r).
            next_state_index (int): Index of the next state (s').
        """
        # Get the current Q-value for the state-action pair
        current_q = self.q_table[state_index, action_index]

        # Get the maximum Q-value for the next state across all possible actions (max_a' Q(s',a'))
        max_future_q = np.max(self.q_table[next_state_index, :])

        # Calculate the Q-learning target
        q_target = reward + self.rl_params.gamma * max_future_q

        # Update the Q-value
        self.q_table[state_index, action_index] = current_q + self.rl_params.alpha * (q_target - current_q)

    def train(self, env: TradingEnvironment, num_episodes: int, print_frequency: int = 100, start_time: float = None):
        """
        Runs the training loop for the Q-learning agent (sequential updates).

        Args:
            env (TradingEnvironment): The trading environment instance.
            num_episodes (int): The total number of episodes (price paths) to train on.
            print_frequency (int): How often to print episode progress.
            start_time (float, optional): The time when training started (from time.time()).
                                          Used for printing elapsed time. Defaults to None.
        """
        print(f"Starting training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            # Reset the environment for the start of a new episode (price path)
            current_state_index = env.reset(path_index=episode)
            done = False
            total_reward = 0 # Track reward per episode

            # Loop through steps in the episode (time steps along the path)
            for step in range(env.num_steps):
                # Agent chooses an action based on the current state
                action_index = self.choose_action(current_state_index)

                # Environment takes a step based on the chosen action
                next_state_index, reward, done = env.step(action_index)

                # Agent learns from the transition (sequential update)
                self.learn(current_state_index, action_index, reward, next_state_index)

                # Update the current state
                current_state_index = next_state_index

                total_reward += reward # Accumulate reward for the episode

                if done:
                    break # End of episode/path

            # Print progress
            if (episode + 1) % print_frequency == 0:
                elapsed_time = time.time() - start_time if start_time is not None else 0
                print(f"Episode {episode + 1}/{num_episodes} finished | Total Reward: {total_reward:.2f} | Elapsed Time: {elapsed_time:.2f} seconds")

        print("-" * 30)
        print("Training complete.")
        print("Q-table shape:", self.q_table.shape)


# --- Main Execution ---
# This block runs only when the script is executed directly

if __name__ == "__main__":
    # --- Set the total number of paths for simulation and training ---
    num_training_paths = 1_000
    # -----------------------------------------------------------------

    # 1) Set up all the needed parameters
    # Use defaults for RLParameters, but add print_frequency
    rl_params = RLParameters(
        print_frequency=100 # Print progress every 100 episodes
    )

    print("RL Parameters:")
    print(rl_params)
    print(f"Action Space: {rl_params.action_space}")
    print(f"Holdings Space: {rl_params.holdings_space}")
    print(f"Total State Space Size: {rl_params.state_space_size}")
    print("-" * 30)

    # Load price data from the CSV file
    # Note: This version loads from CSV, assumes ou_prices.csv exists
    price_file = 'ou_prices.csv'
    try:
        # Assuming the CSV has no header and no index column
        # We only load the number of paths specified by num_training_paths
        # Ensure the CSV has AT LEAST num_training_paths rows
        price_paths_df = pd.read_csv(price_file, header=None, index_col=None, nrows=num_training_paths)
        if price_paths_df.shape[0] < num_training_paths:
             print(f"Warning: Only {price_paths_df.shape[0]} paths available in '{price_file}', requested {num_training_paths}.")
             num_training_paths = price_paths_df.shape[0] # Adjust number of episodes to actual loaded paths

        print(f"Successfully loaded {price_paths_df.shape[0]} paths from '{price_file}'")
        print(f"Shape of loaded price data (paths, steps+1): {price_paths_df.shape}")
    except FileNotFoundError:
        print(f"Error: Price file '{price_file}' not found.")
        print("Please run the simulation script (ou_simulation_code) first to generate 'ou_prices.csv'.")
        exit() # Exit if the price file is not found
    except Exception as e:
        print(f"Error loading price data: {e}")
        exit()


    # Instantiate the Environment
    # The environment automatically gets num_paths from the DataFrame shape
    env = TradingEnvironment(price_paths_df=price_paths_df, rl_params=rl_params)
    print("Trading Environment created.")

    # Instantiate the Agent
    agent = TabularQLearner(
        state_space_size=rl_params.state_space_size,
        action_space=rl_params.action_space,
        rl_params=rl_params
    )
    print("Tabular Q-Learning Agent created.")
    print("-" * 30)

    # Start training
    print("Starting training...")
    start_time = time.time() # --- Start timing ---
    # Pass the start_time to the train method
    agent.train(env=env, num_episodes=env.num_paths, print_frequency=rl_params.print_frequency, start_time=start_time)
    end_time = time.time() # --- End timing ---
    print(f"Total training duration: {end_time - start_time:.2f} seconds.")
    print("-" * 30)

    # --- After Training ---
    # The agent.q_table now contains the learned Q-values.
    # You can analyze the Q-table or evaluate the agent's performance
    # using an evaluation loop (setting epsilon=0 for pure exploitation).
    # Example: print the Q-values for the first state across all actions
    # print("\nQ-values for the first state:")
    # print(agent.q_table[0, :])
