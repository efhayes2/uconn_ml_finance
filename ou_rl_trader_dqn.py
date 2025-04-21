import numpy as np
import pandas as pd
import math
import random
import time
from dataclasses import dataclass, field
from collections import deque
# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import the simulation function and parameters dataclass
# This function will now be faster because of numba in ou_simulation_code
from ou_simulation_code import generate_ou_paths_and_prices, OUSimParameters

# --- RL Parameters Dataclass ---
@dataclass
class RLParameters:
    """
    Holds parameters for the DQN algorithm and environment.
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

    # Q-learning parameters (now for DQN)
    alpha: float = 0.001 # Learning rate for optimizer (e.g., Adam lr)
    epsilon: float = 0.1 # Exploration rate
    gamma: float = 0.95  # Discount factor

    # Training parameters
    print_frequency: int = 100 # How often to print episode progress
    replay_buffer_size: int = 100000 # Maximum size of the replay buffer
    batch_size: int = 64         # Number of experiences to sample for each learning update
    learn_every_steps: int = 4   # How often to perform a learning step (e.g., every 4 steps)
    learn_start_size: int = 1000 # Start learning only after buffer has this many experiences
    target_update_frequency: int = 1000 # New: How often to update the target network (in steps)

    # DQN Network parameters
    state_input_size: int = field(init=False) # Calculated based on price_levels + holdings_space
    hidden_layer_size: int = 64 # Size of hidden layers in the Q-network
    num_hidden_layers: int = 2 # Number of hidden layers
    action_output_size: int = field(init=False) # Calculated based on action_space size

    def __post_init__(self):
        """Calculate derived parameters after initialization."""
        self.spread_cost_per_lot = self.tick_size * self.lot_size # Spread cost is tick_size * abs(delta_n_t) -> spread cost per lot is tick_size * lot_size
        self.state_input_size = self.price_levels + len(self.holdings_space) # One-hot encoding size
        self.action_output_size = len(self.action_space)

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
        """Calculates the total number of discrete states (for tabular mapping, not direct DQN input size)."""
        # Note: DQN doesn't use this directly as a table size, but it's the total number of discrete states
        return self.price_levels * len(self.holdings_space)

# --- Replay Buffer Class ---
class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples (state, action, reward, next_state).
    State is stored as the original state index (int).
    """
    def __init__(self, buffer_size: int):
        """
        Intializes a ReplayBuffer object.

        Args:
            buffer_size (int): Maximum size of buffer.
        """
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.experience = tuple[int, int, float, int] # Define the type of experience tuple

    def add(self, state: int, action: int, reward: float, next_state: int):
        """Add a new experience to memory."""
        e = (state, action, reward, next_state)
        self.buffer.append(e)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        if len(self.buffer) < batch_size:
            raise ValueError("Buffer does not contain enough experiences to sample a batch.")

        experiences = random.sample(self.buffer, k=batch_size)

        # Convert batch of experience tuples into separate numpy arrays
        # State and next_state are indices (int)
        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences]) # Action indices
        rewards = np.array([e[2] for e in experiences], dtype=np.float32)
        next_states = np.array([e[3] for e in experiences])

        return (states, actions, rewards, next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)

# --- Q-Network Model ---
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_input_size: int, action_output_size: int, hidden_layer_size: int, num_hidden_layers: int):
        """Initialize parameters and build model.
        Args:
            state_input_size (int): Dimension of each state (after one-hot encoding)
            action_output_size (int): Dimension of action space
            hidden_layer_size (int): Number of nodes in each hidden layer
            num_hidden_layers (int): Number of hidden layers
        """
        super(QNetwork, self).__init__()

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(state_input_size, hidden_layer_size))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))

        # Output layer
        self.layers.append(nn.Linear(hidden_layer_size, action_output_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for layer in self.layers[:-1]: # Apply ReLU to hidden layers
            x = F.relu(layer(x))
        return self.layers[-1](x) # Output layer (no activation)


# --- Trading Environment Class ---
# (This class remains the same as it provides the state, reward, next_state logic)
class TradingEnvironment:
    """
    Represents the trading environment based on price paths.
    Handles state transitions, costs, and rewards.
    """
    def __init__(self, price_paths_df: pd.DataFrame, rl_params: RLParameters):
        """
        Intializes the trading environment.

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


# --- DQN Agent Class ---
class DQNAgent:
    """
    Implements a Deep Q-Learning agent using PyTorch.
    """
    def __init__(self, rl_params: RLParameters):
        """
        Initializes the DQN agent.

        Args:
            rl_params (RLParameters): Instance of RLParameters dataclass.
        """
        self.rl_params = rl_params
        self.action_space = rl_params.action_space
        self.num_actions = len(self.action_space)

        # Determine device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize policy and target networks
        self.policy_net = QNetwork(
            self.rl_params.state_input_size,
            self.rl_params.action_output_size,
            self.rl_params.hidden_layer_size,
            self.rl_params.num_hidden_layers
        ).to(self.device)
        self.target_net = QNetwork(
            self.rl_params.state_input_size,
            self.rl_params.action_output_size,
            self.rl_params.hidden_layer_size,
            self.rl_params.num_hidden_layers
        ).to(self.device)

        # Copy weights from policy_net to target_net initially
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Set target network to evaluation mode

        # Optimizer and Loss Function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.rl_params.alpha)
        self.criterion = nn.MSELoss() # Mean Squared Error Loss

        # Initialize replay buffer
        self.memory = ReplayBuffer(self.rl_params.replay_buffer_size)
        self.t_step = 0 # Counter for steps to control learning frequency and target updates

        # Map action values to indices (needed for Q-table lookup)
        self._action_to_index = {a: i for i, a in enumerate(self.rl_params.action_space)}
        self._index_to_action = {i: a for i, a in enumerate(self.rl_params.action_space)}


    def _state_index_to_one_hot(self, state_indices: np.ndarray) -> torch.Tensor:
        """
        Converts a batch of state indices into a batch of one-hot encoded state tensors.
        The state index is a single integer representing (price_index, holding_index).
        We need to convert this back to (price_index, holding_index) to create the one-hot vector.
        The one-hot vector will have size price_levels + len(holdings_space).
        """
        if not isinstance(state_indices, np.ndarray):
             state_indices = np.array([state_indices]) # Handle single index case

        batch_size = state_indices.shape[0]
        one_hot_states = torch.zeros(batch_size, self.rl_params.state_input_size, device=self.device)

        size_of_holdings_space = len(self.rl_params.holdings_space)

        for i, state_index in enumerate(state_indices):
            # Convert single state index back to (price_index, holding_index)
            price_index = state_index // size_of_holdings_space
            holding_index = state_index % size_of_holdings_space

            # Set the appropriate bits in the one-hot vector
            one_hot_states[i, price_index] = 1.0
            one_hot_states[i, self.rl_params.price_levels + holding_index] = 1.0

        return one_hot_states


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
            # Convert state index to one-hot tensor
            state_tensor = self._state_index_to_one_hot(state_index)

            # Get Q-values from the policy network
            self.policy_net.eval() # Set policy network to evaluation mode
            with torch.no_grad(): # Disable gradient calculation for inference
                q_values = self.policy_net(state_tensor)
            self.policy_net.train() # Set policy network back to training mode

            # Choose the action with the highest Q-value
            action_index = q_values.argmax().item() # .item() gets the scalar value from a tensor

        return action_index

    def step(self, state_index: int, action_index: int, reward: float, next_state_index: int):
        """
        Saves experience in replay buffer and triggers learning if conditions met.

        Args:
            state_index (int): Index of the current state (s).
            action_index (int): Index of the action taken (a).
            reward (float): The reward received (r).
            next_state_index (int): Index of the next state (s').
        """
        # Save experience in replay buffer
        self.memory.add(state_index, action_index, reward, next_state_index)

        # Increment step counter
        self.t_step += 1

        # Perform learning step if conditions met
        # Learn every `learn_every_steps` steps after `learn_start_size` experiences are collected
        if self.t_step % self.rl_params.learn_every_steps == 0 and len(self.memory) > self.rl_params.learn_start_size:
            # Sample a batch and learn
            experiences = self.memory.sample(self.rl_params.batch_size)
            self.learn_batch(experiences)

        # Update target network periodically
        if self.t_step % self.rl_params.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval() # Ensure target network stays in eval mode


    def learn_batch(self, experiences: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """
        Updates the policy network using a batch of experiences.

        Args:
            experiences (tuple): A tuple of numpy arrays (states, actions, rewards, next_states).
        """
        states_indices, actions, rewards, next_states_indices = experiences

        # Convert numpy arrays to PyTorch tensors and move to device
        states = self._state_index_to_one_hot(states_indices)
        actions = torch.from_numpy(actions).long().to(self.device) # Actions are indices (long type)
        rewards = torch.from_numpy(rewards).float().to(self.device) # Rewards are float
        next_states = self._state_index_to_one_hot(next_states_indices)

        # --- Calculate Q-learning targets ---
        # Get Q values for next states from target network: Q_target(s', a')
        # We need the max Q value over all actions for each next state s'
        # target_net(next_states) gives Q values for all actions for the batch of next states
        # .max(1)[0] gets the maximum Q value for each state in the batch
        max_future_q_targets = self.target_net(next_states).max(1)[0].detach() # Detach from graph

        # Calculate the target Q values: r + gamma * max_a' Q_target(s', a')
        q_targets = rewards + (self.rl_params.gamma * max_future_q_targets)
        # ------------------------------------

        # --- Calculate Q-values from policy network ---
        # Get Q values for current states from policy network: Q_policy(s, a)
        # We only need the Q value for the action *actually taken* (a) for each state (s) in the batch
        # policy_net(states) gives Q values for all actions for the batch of states
        # .gather(1, actions.unsqueeze(1)) selects the Q value corresponding to the action taken
        # unsqueeze(1) is needed because gather expects the index tensor to have the same number of dimensions as the source tensor
        q_expected = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1) # Remove extra dimension


        # --- Compute Loss and backpropagate ---
        loss = self.criterion(q_expected, q_targets) # Calculate MSE loss

        self.optimizer.zero_grad() # Zero gradients
        loss.backward()           # Backpropagate loss
        # Optional: Clip gradients to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()     # Update weights


    def train(self, env: TradingEnvironment, num_episodes: int, print_frequency: int = 100, start_time: float = None):
        """
        Runs the training loop for the DQN agent.

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
                # Action is chosen based on policy_net
                action_index = self.choose_action(current_state_index)

                # Environment takes a step based on the chosen action
                # This gives us the experience tuple
                next_state_index, reward, done = env.step(action_index)

                # Agent saves experience and potentially learns from a batch
                # Learning happens inside agent.step based on learn_every_steps
                self.step(current_state_index, action_index, reward, next_state_index)

                # Update the current state
                current_state_index = next_state_index

                total_reward += reward # Accumulate reward for the episode

                if done:
                    break # End of episode/path

            # Print progress
            if (episode + 1) % print_frequency == 0:
                elapsed_time = time.time() - start_time if start_time is not None else 0
                print(f"Episode {episode + 1}/{num_episodes} finished | Total Reward: {total_reward:.2f} | Elapsed Time: {elapsed_time:.2f} seconds | Buffer Size: {len(self.memory)}")

        print("-" * 30)
        print("Training complete.")


# --- Main Execution ---
# This block runs only when the script is executed directly

if __name__ == "__main__":
    # --- Set the total number of paths for simulation and training ---
    num_training_paths = 10000
    # -----------------------------------------------------------------

    # 1) Set up all the needed parameters
    # Use defaults from OUSimParameters, but override num_paths and saving
    sim_params = OUSimParameters(
        num_paths=num_training_paths, # Use the variable here
        save_prices_to_csv=False # Don't save prices to CSV during training run
    )

    # Use defaults for RLParameters, but override batch/buffer/learning params and print_frequency
    # Adjust replay_buffer_size and learn_start_size based on num_training_paths if needed
    rl_params = RLParameters(
        print_frequency=100, # Print progress every 100 episodes
        replay_buffer_size=100000, # Buffer size
        batch_size=64,         # Batch size for learning
        learn_every_steps=4,   # Learn every 4 steps
        learn_start_size=1000, # Start learning after 1000 experiences
        target_update_frequency=1000, # Update target network every 1000 steps
        alpha=0.001, # Optimizer learning rate
        hidden_layer_size=64, # Network size
        num_hidden_layers=2 # Number of hidden layers
    )

    print("Simulation Parameters:")
    print(sim_params)
    print("-" * 30)
    print("RL Parameters:")
    print(rl_params)
    print(f"Action Space: {rl_params.action_space}")
    print(f"Holdings Space: {rl_params.holdings_space}")
    # State space size for tabular mapping (not direct DQN input size)
    print(f"Tabular State Space Size: {rl_params.state_space_size}")
    # DQN input size (one-hot encoding size)
    print(f"DQN State Input Size: {rl_params.state_input_size}")
    print(f"DQN Action Output Size: {rl_params.action_output_size}")
    print("-" * 30)


    # Generate price data directly using the simulation function
    print("Generating price data...")
    price_paths_df = generate_ou_paths_and_prices(sim_params)
    print(f"Price data generated with shape (paths, steps+1): {price_paths_df.shape}")
    print("-" * 30)


    # Instantiate the Environment
    # The environment automatically gets num_paths from the DataFrame shape
    env = TradingEnvironment(price_paths_df=price_paths_df, rl_params=rl_params)
    print("Trading Environment created.")

    # Instantiate the Agent (DQN Agent)
    agent = DQNAgent(rl_params=rl_params)
    print("DQN Agent created.")
    print(f"Agent Policy Network: {agent.policy_net}")
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
    # The agent.policy_net and agent.target_net now contain the learned weights.
    # You can analyze the learned policy by feeding states to agent.policy_net
    # or evaluate the agent's performance using an evaluation loop (setting epsilon=0).
