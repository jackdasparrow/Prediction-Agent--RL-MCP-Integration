"""
Lightweight Reinforcement Learning Agent
Implements contextual bandit (LinUCB/Thompson Sampling) and simple DQN for ranking instruments.
Optimized for edge deployment (RTX 3060 + 16-32GB RAM).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from pathlib import Path
import joblib
import json
import logging
from typing import Dict, List, Tuple, Optional
import random

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class LinUCBAgent:
    """
    Linear Upper Confidence Bound (LinUCB) Contextual Bandit
    Efficient for ranking with uncertainty estimates.
    """
    
    def __init__(
        self,
        n_features: int,
        alpha: float = 1.0,
        model_dir: str = "./models"
    ):
        """
        Args:
            n_features: Number of context features
            alpha: Exploration parameter (higher = more exploration)
            model_dir: Directory to save models
        """
        self.n_features = n_features
        self.alpha = alpha
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize matrices for each arm (symbol)
        self.arms = {}  # Dict of arm_id -> {'A': matrix, 'b': vector}
        self.total_reward = 0
        self.n_rounds = 0
        
    def _init_arm(self, arm_id: str):
        """Initialize matrices for a new arm"""
        self.arms[arm_id] = {
            'A': np.identity(self.n_features),
            'b': np.zeros(self.n_features)
        }
    
    def select_action(self, context: np.ndarray, arm_id: str) -> Tuple[float, float]:
        """
        Select action using UCB
        
        Args:
            context: Feature vector for the symbol
            arm_id: Symbol identifier
            
        Returns:
            (score, confidence) tuple
        """
        if arm_id not in self.arms:
            self._init_arm(arm_id)
        
        A = self.arms[arm_id]['A']
        b = self.arms[arm_id]['b']
        
        # Compute A^-1
        A_inv = np.linalg.inv(A)
        
        # Theta estimate
        theta = A_inv.dot(b)
        
        # UCB score
        confidence = self.alpha * np.sqrt(context.dot(A_inv).dot(context))
        score = theta.dot(context) + confidence
        
        return float(score), float(confidence)
    
    def update(self, arm_id: str, context: np.ndarray, reward: float):
        """
        Update arm parameters after observing reward
        
        Args:
            arm_id: Symbol identifier
            context: Feature vector
            reward: Observed reward
        """
        if arm_id not in self.arms:
            self._init_arm(arm_id)
        
        self.arms[arm_id]['A'] += np.outer(context, context)
        self.arms[arm_id]['b'] += reward * context
        
        self.total_reward += reward
        self.n_rounds += 1
    
    def rank_symbols(
        self,
        contexts: Dict[str, np.ndarray],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, float]]:
        """
        Rank all symbols by their UCB scores
        
        Args:
            contexts: Dict of symbol -> feature vector
            top_k: Return only top k symbols
            
        Returns:
            List of (symbol, score, confidence) tuples, sorted by score
        """
        rankings = []
        
        for symbol, context in contexts.items():
            score, confidence = self.select_action(context, symbol)
            rankings.append((symbol, score, confidence))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            rankings = rankings[:top_k]
        
        return rankings
    
    def save(self, filename: str = "linucb_agent.pkl"):
        """Save agent state"""
        save_path = self.model_dir / filename
        state = {
            'arms': self.arms,
            'n_features': self.n_features,
            'alpha': self.alpha,
            'total_reward': self.total_reward,
            'n_rounds': self.n_rounds
        }
        joblib.dump(state, save_path)
        logger.info(f"LinUCB agent saved to {save_path}")
    
    def load(self, filename: str = "linucb_agent.pkl"):
        """Load agent state"""
        load_path = self.model_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        state = joblib.load(load_path)
        self.arms = state['arms']
        self.n_features = state['n_features']
        self.alpha = state['alpha']
        self.total_reward = state['total_reward']
        self.n_rounds = state['n_rounds']
        logger.info(f"LinUCB agent loaded from {load_path}")


class ThompsonSamplingAgent:
    """
    Thompson Sampling Contextual Bandit
    Bayesian approach with posterior sampling.
    """
    
    def __init__(
        self,
        n_features: int,
        lambda_: float = 1.0,
        v: float = 1.0,
        model_dir: str = "./models"
    ):
        """
        Args:
            n_features: Number of context features
            lambda_: Regularization parameter
            v: Noise variance
            model_dir: Directory to save models
        """
        self.n_features = n_features
        self.lambda_ = lambda_
        self.v = v
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.arms = {}
        self.total_reward = 0
        self.n_rounds = 0
    
    def _init_arm(self, arm_id: str):
        """Initialize parameters for a new arm"""
        self.arms[arm_id] = {
            'B': self.lambda_ * np.identity(self.n_features),
            'mu': np.zeros(self.n_features),
            'f': np.zeros(self.n_features)
        }
    
    def select_action(self, context: np.ndarray, arm_id: str) -> Tuple[float, float]:
        """
        Select action by sampling from posterior
        
        Args:
            context: Feature vector
            arm_id: Symbol identifier
            
        Returns:
            (score, confidence) tuple
        """
        if arm_id not in self.arms:
            self._init_arm(arm_id)
        
        B = self.arms[arm_id]['B']
        f = self.arms[arm_id]['f']
        
        # Compute posterior mean
        B_inv = np.linalg.inv(B)
        mu = B_inv.dot(f)
        
        # Sample from posterior
        theta_sample = np.random.multivariate_normal(mu, self.v * B_inv)
        
        # Compute score
        score = theta_sample.dot(context)
        confidence = np.sqrt(context.dot(B_inv).dot(context))
        
        return float(score), float(confidence)
    
    def update(self, arm_id: str, context: np.ndarray, reward: float):
        """Update arm parameters"""
        if arm_id not in self.arms:
            self._init_arm(arm_id)
        
        self.arms[arm_id]['B'] += np.outer(context, context)
        self.arms[arm_id]['f'] += reward * context
        self.arms[arm_id]['mu'] = np.linalg.inv(self.arms[arm_id]['B']).dot(self.arms[arm_id]['f'])
        
        self.total_reward += reward
        self.n_rounds += 1
    
    def rank_symbols(
        self,
        contexts: Dict[str, np.ndarray],
        top_k: Optional[int] = None,
        n_samples: int = 100
    ) -> List[Tuple[str, float, float]]:
        """
        Rank symbols using Thompson sampling
        
        Args:
            contexts: Dict of symbol -> feature vector
            top_k: Return only top k symbols
            n_samples: Number of posterior samples for stable ranking
            
        Returns:
            List of (symbol, score, confidence) tuples
        """
        rankings = {}
        
        # Sample multiple times for stable rankings
        for _ in range(n_samples):
            for symbol, context in contexts.items():
                score, confidence = self.select_action(context, symbol)
                if symbol not in rankings:
                    rankings[symbol] = {'scores': [], 'confidences': []}
                rankings[symbol]['scores'].append(score)
                rankings[symbol]['confidences'].append(confidence)
        
        # Average scores
        results = []
        for symbol, data in rankings.items():
            avg_score = np.mean(data['scores'])
            avg_confidence = np.mean(data['confidences'])
            results.append((symbol, avg_score, avg_confidence))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def save(self, filename: str = "thompson_agent.pkl"):
        """Save agent state"""
        save_path = self.model_dir / filename
        state = {
            'arms': self.arms,
            'n_features': self.n_features,
            'lambda_': self.lambda_,
            'v': self.v,
            'total_reward': self.total_reward,
            'n_rounds': self.n_rounds
        }
        joblib.dump(state, save_path)
        logger.info(f"Thompson Sampling agent saved to {save_path}")
    
    def load(self, filename: str = "thompson_agent.pkl"):
        """Load agent state"""
        load_path = self.model_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        state = joblib.load(load_path)
        self.arms = state['arms']
        self.n_features = state['n_features']
        self.lambda_ = state['lambda_']
        self.v = state['v']
        self.total_reward = state['total_reward']
        self.n_rounds = state['n_rounds']
        logger.info(f"Thompson Sampling agent loaded from {load_path}")


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Simple DQN network for Q-value estimation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """
    Deep Q-Network Agent for ranking
    Lightweight implementation optimized for CPU/single GPU.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,  # long, hold, short
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        device: str = 'cpu',
        model_dir: str = "./models"
    ):
        """
        Args:
            state_dim: Dimension of state (features)
            action_dim: Number of actions
            hidden_dim: Hidden layer size
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_capacity: Replay buffer size
            batch_size: Training batch size
            device: Device to use ('cpu' or 'cuda')
            model_dir: Directory to save models
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training stats
        self.total_reward = 0
        self.episode_rewards = []
        self.losses = []
        
        logger.info(f"DQN Agent initialized on {self.device}")
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            evaluate: If True, use greedy policy (no exploration)
            
        Returns:
            Action index
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def get_action_scores(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions
        
        Args:
            state: Current state
            
        Returns:
            Array of Q-values
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step
        
        Returns:
            Loss value or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Unpack batch
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def rank_symbols(
        self,
        contexts: Dict[str, np.ndarray],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, int, float]]:
        """
        Rank symbols using DQN Q-values
        
        Args:
            contexts: Dict of symbol -> feature vector
            top_k: Return only top k symbols
            
        Returns:
            List of (symbol, score, action, confidence) tuples
        """
        rankings = []
        
        for symbol, context in contexts.items():
            q_values = self.get_action_scores(context)
            action = q_values.argmax()
            score = q_values[action]
            
            # Confidence based on Q-value spread
            confidence = float(np.exp(score) / np.exp(q_values).sum())
            
            rankings.append((symbol, float(score), int(action), confidence))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            rankings = rankings[:top_k]
        
        return rankings
    
    def save(self, filename: str = "dqn_agent.pt"):
        """Save agent state"""
        save_path = self.model_dir / filename
        
        checkpoint = {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"DQN agent saved to {save_path}")
    
    def load(self, filename: str = "dqn_agent.pt"):
        """Load agent state"""
        load_path = self.model_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.total_reward = checkpoint['total_reward']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
        
        logger.info(f"DQN agent loaded from {load_path}")


class RLTrainer:
    """Training loop for RL agents"""
    
    def __init__(
        self,
        agent,
        feature_store: Dict[str, pd.DataFrame],
        agent_type: str = "linucb"  # or "thompson" or "dqn"
    ):
        """
        Args:
            agent: RL agent instance
            feature_store: Dictionary of symbol -> features DataFrame
            agent_type: Type of agent
        """
        self.agent = agent
        self.feature_store = feature_store
        self.agent_type = agent_type
        self.training_history = []
    
    def _compute_reward(self, symbol: str, action: int, horizon: int = 1) -> float:
        """
        Compute reward based on future returns
        
        Args:
            symbol: Symbol identifier
            action: Action taken (for DQN: 0=short, 1=hold, 2=long)
            horizon: Prediction horizon
            
        Returns:
            Reward value
        """
        df = self.feature_store[symbol]
        
        # Use target_return generated by feature pipeline; fallback to generic returns if present
        ret_col = 'target_return' if 'target_return' in df.columns else ('returns' if 'returns' in df.columns else None)
        if ret_col is None or len(df) < horizon:
            return 0.0
        
        # Get future return
        future_return = df[ret_col].iloc[-horizon:].mean()
        
        if self.agent_type == "dqn":
            # DQN: align action with return direction
            if action == 2:  # long
                reward = future_return
            elif action == 0:  # short
                reward = -future_return
            else:  # hold
                reward = 0.0
        else:
            # Bandit: direct return as reward
            reward = future_return
        
        # Scale reward
        return reward * 100  # Scale to reasonable range
    
    def train_bandit(
        self,
        n_rounds: int = 100,
        top_k: int = 20,
        horizon: int = 1
    ) -> Dict:
        """
        Train contextual bandit agent
        
        Args:
            n_rounds: Number of training rounds
            top_k: Number of symbols to select each round
            horizon: Prediction horizon
            
        Returns:
            Training statistics
        """
        logger.info(f"Training {self.agent_type} agent for {n_rounds} rounds")
        
        cumulative_reward = 0
        rewards_per_round = []
        
        for round_num in range(n_rounds):
            # Prepare contexts for all symbols
            contexts = {}
            for symbol, df in self.feature_store.items():
                if len(df) > 0:
                    # Use last row features
                    feature_cols = [col for col in df.columns if col not in 
                                  ['Open', 'High', 'Low', 'Close', 'Volume', 
                                   'symbol', 'source', 'fetch_timestamp', 'target', 'target_direction']]
                    context = df[feature_cols].iloc[-1].values
                    
                    # Handle NaN
                    context = np.nan_to_num(context, 0)
                    contexts[symbol] = context
            
            # Rank and select top-k
            rankings = self.agent.rank_symbols(contexts, top_k=top_k)
            
            # Update based on observed rewards
            round_reward = 0
            for symbol, score, confidence in rankings:
                reward = self._compute_reward(symbol, action=2, horizon=horizon)  # Assume long action
                self.agent.update(symbol, contexts[symbol], reward)
                round_reward += reward
            
            cumulative_reward += round_reward
            rewards_per_round.append(round_reward)
            
            if (round_num + 1) % 10 == 0:
                avg_reward = np.mean(rewards_per_round[-10:])
                logger.info(f"Round {round_num + 1}/{n_rounds} - Avg Reward: {avg_reward:.4f}")
        
        stats = {
            'cumulative_reward': cumulative_reward,
            'avg_reward': cumulative_reward / n_rounds,
            'rewards_per_round': rewards_per_round,
            'total_rounds': n_rounds
        }
        
        self.training_history.append(stats)
        
        return stats
    
    def train_dqn(
        self,
        n_episodes: int = 100,
        max_steps: int = 50,
        target_update_freq: int = 10,
        horizon: int = 1
    ) -> Dict:
        """
        Train DQN agent
        
        Args:
            n_episodes: Number of training episodes
            max_steps: Max steps per episode
            target_update_freq: Target network update frequency
            horizon: Prediction horizon
            
        Returns:
            Training statistics
        """
        logger.info(f"Training DQN agent for {n_episodes} episodes")
        
        symbols = list(self.feature_store.keys())
        
        for episode in range(n_episodes):
            episode_reward = 0
            episode_losses = []
            
            for step in range(max_steps):
                # Sample random symbol
                symbol = random.choice(symbols)
                df = self.feature_store[symbol]
                
                if len(df) < 2:
                    continue
                
                # Get state
                feature_cols = [col for col in df.columns if col not in 
                              ['Open', 'High', 'Low', 'Close', 'Volume', 
                               'symbol', 'source', 'fetch_timestamp', 'target', 'target_direction']]
                state = df[feature_cols].iloc[-2].values
                next_state = df[feature_cols].iloc[-1].values
                
                state = np.nan_to_num(state, 0)
                next_state = np.nan_to_num(next_state, 0)
                
                # Select action
                action = self.agent.select_action(state)
                
                # Compute reward
                reward = self._compute_reward(symbol, action, horizon)
                
                # Store transition
                done = (step == max_steps - 1)
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Train
                loss = self.agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                
                episode_reward += reward
            
            # Update target network
            if (episode + 1) % target_update_freq == 0:
                self.agent.update_target_network()
            
            self.agent.episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.agent.episode_rewards[-10:])
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                logger.info(f"Episode {episode + 1}/{n_episodes} - Reward: {avg_reward:.2f}, Loss: {avg_loss:.4f}, Epsilon: {self.agent.epsilon:.3f}")
        
        stats = {
            'episode_rewards': self.agent.episode_rewards,
            'avg_reward': np.mean(self.agent.episode_rewards),
            'total_episodes': n_episodes
        }
        
        return stats
    
    def evaluate(self, top_k: int = 20) -> Dict:
        """
        Evaluate agent performance
        
        Args:
            top_k: Number of top symbols to evaluate
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating agent...")
        
        # Prepare contexts
        contexts = {}
        for symbol, df in self.feature_store.items():
            if len(df) > 0:
                feature_cols = [col for col in df.columns if col not in 
                              ['Open', 'High', 'Low', 'Close', 'Volume', 
                               'symbol', 'source', 'fetch_timestamp', 'target', 'target_direction']]
                context = df[feature_cols].iloc[-1].values
                context = np.nan_to_num(context, 0)
                contexts[symbol] = context
        
        # Get rankings
        if self.agent_type == "dqn":
            rankings = self.agent.rank_symbols(contexts, top_k=top_k)
            # rankings: (symbol, score, action, confidence)
            selected_symbols = [(r[0], r[2]) for r in rankings]  # symbol, action
        else:
            rankings = self.agent.rank_symbols(contexts, top_k=top_k)
            # rankings: (symbol, score, confidence)
            selected_symbols = [(r[0], 2) for r in rankings]  # Assume long action
        
        # Compute simulated returns
        returns = []
        for symbol, action in selected_symbols:
            reward = self._compute_reward(symbol, action, horizon=1)
            returns.append(reward)
        
        # Compute metrics
        metrics = {
            'mean_return': np.mean(returns),
            'sharpe_proxy': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            'win_rate': np.mean([r > 0 for r in returns]),
            'top_symbols': [r[0] for r in rankings[:10]]
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics


# Standalone script for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing LinUCB Agent ===")
    n_features = 50
    linucb = LinUCBAgent(n_features=n_features, alpha=1.0)
    
    # Create dummy contexts
    contexts = {
        f'SYMBOL_{i}': np.random.randn(n_features) 
        for i in range(10)
    }
    
    # Rank symbols
    rankings = linucb.rank_symbols(contexts, top_k=5)
    print(f"Top 5 ranked symbols:")
    for symbol, score, conf in rankings:
        print(f"  {symbol}: score={score:.3f}, confidence={conf:.3f}")
    
    # Update with rewards
    for symbol, context in contexts.items():
        reward = np.random.randn()  # Simulated reward
        linucb.update(symbol, context, reward)
    
    print(f"Total reward: {linucb.total_reward:.3f}")
    
    print("\n=== Testing Thompson Sampling Agent ===")
    thompson = ThompsonSamplingAgent(n_features=n_features)
    rankings = thompson.rank_symbols(contexts, top_k=5, n_samples=50)
    print(f"Top 5 ranked symbols:")
    for symbol, score, conf in rankings:
        print(f"  {symbol}: score={score:.3f}, confidence={conf:.3f}")
    
    print("\n=== Testing DQN Agent ===")
    dqn = DQNAgent(state_dim=n_features, action_dim=3, device='cpu')
    
    # Simulate training
    for _ in range(100):
        state = np.random.randn(n_features)
        action = dqn.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(n_features)
        dqn.store_transition(state, action, reward, next_state, False)
        loss = dqn.train_step()
    
    rankings = dqn.rank_symbols(contexts, top_k=5)
    print(f"Top 5 ranked symbols:")
    for symbol, score, action, conf in rankings:
        action_name = ['short', 'hold', 'long'][action]
        print(f"  {symbol}: score={score:.3f}, action={action_name}, confidence={conf:.3f}")
    
    print("\n[OK] RL agent tests complete!")