"""
Lightweight Reinforcement Learning Agent
Implements contextual bandit (LinUCB/Thompson Sampling) and simple DQN for ranking instruments.
Optimized for edge deployment (RTX 3060 + 16-32GB RAM).
Works with features from both Yahoo Finance and Alpha Vantage data.
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
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class FeedbackMemory:
    """
    Manages user feedback for RL agent learning
    Stores feedback in feedback_memory.json and logs adjustments
    """
    
    def __init__(self, memory_file: str = "feedback_memory.json", log_file: str = "logs/feedback_loop.json"):
        self.memory_file = Path(memory_file)
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> Dict:
        """Load existing feedback data"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load feedback memory: {e}")
        return {"feedback_history": [], "reward_adjustments": {}}
    
    def _save_feedback(self):
        """Save feedback data to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback memory: {e}")
    
    def add_feedback(self, symbol: str, predicted_action: str, user_feedback: str, 
                    confidence: float = 0.0, features: Dict = None):
        """
        Add user feedback for a prediction
        
        Args:
            symbol: Stock symbol
            predicted_action: Action predicted by RL agent (long/short/hold)
            user_feedback: User's feedback (correct/incorrect)
            confidence: Confidence score of the prediction
            features: Feature vector used for prediction
        """
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "predicted_action": predicted_action,
            "user_feedback": user_feedback,
            "confidence": confidence,
            "features": features or {}
        }
        
        self.feedback_data["feedback_history"].append(feedback_entry)
        
        # Calculate reward adjustment
        reward_adjustment = self._calculate_reward_adjustment(predicted_action, user_feedback, confidence)
        
        # Update reward adjustments
        if symbol not in self.feedback_data["reward_adjustments"]:
            self.feedback_data["reward_adjustments"][symbol] = []
        
        self.feedback_data["reward_adjustments"][symbol].append({
            "timestamp": feedback_entry["timestamp"],
            "adjustment": reward_adjustment,
            "action": predicted_action,
            "feedback": user_feedback
        })
        
        # Log the feedback
        self._log_feedback(feedback_entry, reward_adjustment)
        
        # Save to file
        self._save_feedback()
        
        logger.info(f"Feedback recorded for {symbol}: {predicted_action} -> {user_feedback} (adjustment: {reward_adjustment})")
        
        return reward_adjustment
    
    def _calculate_reward_adjustment(self, predicted_action: str, user_feedback: str, confidence: float) -> float:
        """Calculate reward adjustment based on feedback"""
        if user_feedback.lower() == "correct":
            # Positive reward for correct predictions, scaled by confidence
            return confidence * 0.1
        elif user_feedback.lower() == "incorrect":
            # Negative reward for incorrect predictions, scaled by confidence
            return -confidence * 0.1
        else:
            return 0.0
    
    def _log_feedback(self, feedback_entry: Dict, reward_adjustment: float):
        """Log feedback to feedback_loop.json"""
        log_entry = {
            "timestamp": feedback_entry["timestamp"],
            "symbol": feedback_entry["symbol"],
            "predicted_action": feedback_entry["predicted_action"],
            "user_feedback": feedback_entry["user_feedback"],
            "confidence": feedback_entry["confidence"],
            "reward_adjustment": reward_adjustment,
            "action": "feedback_recorded"
        }
        
        try:
            # Load existing logs
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = {"feedback_logs": []}
            
            logs["feedback_logs"].append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs["feedback_logs"]) > 1000:
                logs["feedback_logs"] = logs["feedback_logs"][-1000:]
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")
    
    def get_recent_feedback(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Get recent feedback entries"""
        history = self.feedback_data["feedback_history"]
        
        if symbol:
            history = [entry for entry in history if entry["symbol"] == symbol]
        
        return history[-limit:]
    
    def get_reward_adjustments(self, symbol: str) -> List[float]:
        """Get reward adjustments for a symbol"""
        if symbol in self.feedback_data["reward_adjustments"]:
            return [adj["adjustment"] for adj in self.feedback_data["reward_adjustments"][symbol]]
        return []
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        history = self.feedback_data["feedback_history"]
        
        if not history:
            return {"total_feedback": 0, "correct_rate": 0.0, "symbols": []}
        
        total = len(history)
        correct = sum(1 for entry in history if entry["user_feedback"].lower() == "correct")
        symbols = list(set(entry["symbol"] for entry in history))
        
        return {
            "total_feedback": total,
            "correct_rate": correct / total if total > 0 else 0.0,
            "symbols": symbols,
            "recent_feedback": history[-5:] if history else []
        }

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
        
        # Feedback integration
        self.feedback_memory = FeedbackMemory()
        
    def incorporate_feedback(self, symbol: str, predicted_action: str, user_feedback: str, 
                           confidence: float = 0.0, features: np.ndarray = None):
        """
        Incorporate user feedback into the RL agent
        
        Args:
            symbol: Stock symbol
            predicted_action: Action predicted by RL agent
            user_feedback: User's feedback (correct/incorrect)
            confidence: Confidence score of the prediction
            features: Feature vector used for prediction
        """
        # Add feedback to memory
        reward_adjustment = self.feedback_memory.add_feedback(
            symbol, predicted_action, user_feedback, confidence, 
            features.tolist() if features is not None else None
        )
        
        # Update LinUCB matrices with feedback
        if features is not None and symbol in self.arms:
            # Convert feedback to reward signal
            if user_feedback.lower() == "correct":
                reward = confidence * 0.1
            elif user_feedback.lower() == "incorrect":
                reward = -confidence * 0.1
            else:
                reward = 0.0
            
            # Update LinUCB matrices
            self._update_arm(symbol, features, reward)
            
            logger.info(f"LinUCB updated for {symbol} with reward {reward}")
        
        return reward_adjustment
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        return self.feedback_memory.get_feedback_stats()
    
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
        
        # Validate context
        if len(context) != self.n_features:
            logger.warning(f"Context size mismatch: expected {self.n_features}, got {len(context)}")
            context = np.pad(context, (0, max(0, self.n_features - len(context))), mode='constant')[:self.n_features]
        
        # Handle NaN/inf
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        
        A = self.arms[arm_id]['A']
        b = self.arms[arm_id]['b']
        
        try:
            # Compute A^-1 with regularization for stability
            A_inv = np.linalg.inv(A + np.eye(self.n_features) * 1e-6)
            
            # Theta estimate
            theta = A_inv.dot(b)
            
            # UCB score
            confidence = self.alpha * np.sqrt(context.dot(A_inv).dot(context))
            score = theta.dot(context) + confidence
            
        except np.linalg.LinAlgError:
            logger.warning(f"Matrix inversion failed for {arm_id}, returning default")
            score = 0.0
            confidence = 1.0
        
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
        
        # Validate and clean inputs
        if len(context) != self.n_features:
            context = np.pad(context, (0, max(0, self.n_features - len(context))), mode='constant')[:self.n_features]
        
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        
        # Update matrices
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
            try:
                score, confidence = self.select_action(context, symbol)
                rankings.append((symbol, score, confidence))
            except Exception as e:
                logger.error(f"Error ranking {symbol}: {e}")
                continue
        
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
        
        # Feedback integration
        self.feedback_memory = FeedbackMemory()
        
        logger.info(f"Thompson Sampling Agent initialized with {n_features} features")
    
    def incorporate_feedback(self, symbol: str, predicted_action: str, user_feedback: str, 
                           confidence: float = 0.0, features: np.ndarray = None):
        """
        Incorporate user feedback into the Thompson Sampling agent
        
        Args:
            symbol: Stock symbol
            predicted_action: Action predicted by RL agent
            user_feedback: User's feedback (correct/incorrect)
            confidence: Confidence score of the prediction
            features: Feature vector used for prediction
        """
        # Add feedback to memory
        reward_adjustment = self.feedback_memory.add_feedback(
            symbol, predicted_action, user_feedback, confidence, 
            features.tolist() if features is not None else None
        )
        
        # Update Thompson Sampling parameters with feedback
        if features is not None and symbol in self.arms:
            # Convert feedback to reward signal
            if user_feedback.lower() == "correct":
                reward = confidence * 0.1
            elif user_feedback.lower() == "incorrect":
                reward = -confidence * 0.1
            else:
                reward = 0.0
            
            # Update Thompson Sampling parameters
            self._update_arm(symbol, features, reward)
            
            logger.info(f"Thompson Sampling updated for {symbol} with reward {reward}")
        
        return reward_adjustment
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        return self.feedback_memory.get_feedback_stats()
    
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
        
        # Validate context
        if len(context) != self.n_features:
            context = np.pad(context, (0, max(0, self.n_features - len(context))), mode='constant')[:self.n_features]
        
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        
        B = self.arms[arm_id]['B']
        f = self.arms[arm_id]['f']
        
        try:
            # Compute posterior mean with regularization
            B_inv = np.linalg.inv(B + np.eye(self.n_features) * 1e-6)
            mu = B_inv.dot(f)
            
            # Sample from posterior
            theta_sample = np.random.multivariate_normal(mu, self.v * B_inv)
            
            # Compute score
            score = theta_sample.dot(context)
            confidence = np.sqrt(context.dot(B_inv).dot(context))
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Posterior sampling failed for {arm_id}: {e}")
            score = 0.0
            confidence = 1.0
        
        return float(score), float(confidence)
    
    def update(self, arm_id: str, context: np.ndarray, reward: float):
        """Update arm parameters"""
        if arm_id not in self.arms:
            self._init_arm(arm_id)
        
        # Validate inputs
        if len(context) != self.n_features:
            context = np.pad(context, (0, max(0, self.n_features - len(context))), mode='constant')[:self.n_features]
        
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        
        # Update parameters
        self.arms[arm_id]['B'] += np.outer(context, context)
        self.arms[arm_id]['f'] += reward * context
        
        try:
            B_inv = np.linalg.inv(self.arms[arm_id]['B'] + np.eye(self.n_features) * 1e-6)
            self.arms[arm_id]['mu'] = B_inv.dot(self.arms[arm_id]['f'])
        except np.linalg.LinAlgError:
            pass
        
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
                try:
                    score, confidence = self.select_action(context, symbol)
                    if symbol not in rankings:
                        rankings[symbol] = {'scores': [], 'confidences': []}
                    rankings[symbol]['scores'].append(score)
                    rankings[symbol]['confidences'].append(confidence)
                except Exception as e:
                    logger.error(f"Error sampling {symbol}: {e}")
                    continue
        
        # Average scores
        results = []
        for symbol, data in rankings.items():
            if data['scores']:
                avg_score = np.mean(data['scores'])
                avg_confidence = np.mean(data['confidences'])
                results.append((symbol, float(avg_score), float(avg_confidence)))
        
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
        # Validate inputs
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        next_state = np.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        
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
        
        # Feedback integration
        self.feedback_memory = FeedbackMemory()
        
        logger.info(f"DQN Agent initialized on {self.device}")
    
    def incorporate_feedback(self, symbol: str, predicted_action: str, user_feedback: str, 
                           confidence: float = 0.0, features: np.ndarray = None):
        """
        Incorporate user feedback into the DQN agent
        
        Args:
            symbol: Stock symbol
            predicted_action: Action predicted by RL agent
            user_feedback: User's feedback (correct/incorrect)
            confidence: Confidence score of the prediction
            features: Feature vector used for prediction
        """
        # Add feedback to memory
        reward_adjustment = self.feedback_memory.add_feedback(
            symbol, predicted_action, user_feedback, confidence, 
            features.tolist() if features is not None else None
        )
        
        # For DQN, we can add the feedback as an experience in the replay buffer
        if features is not None:
            # Convert feedback to reward signal
            if user_feedback.lower() == "correct":
                reward = confidence * 0.1
            elif user_feedback.lower() == "incorrect":
                reward = -confidence * 0.1
            else:
                reward = 0.0
            
            # Add to replay buffer for future training
            action_idx = {"long": 0, "short": 1, "hold": 2}.get(predicted_action.lower(), 2)
            experience = Experience(features, action_idx, reward, features, True)
            self.replay_buffer.push(experience)
            
            logger.info(f"DQN feedback recorded for {symbol}: {predicted_action} -> {user_feedback} (reward: {reward})")
        
        return reward_adjustment
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        return self.feedback_memory.get_feedback_stats()
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            evaluate: If True, use greedy policy (no exploration)
            
        Returns:
            Action index
        """
        # Validate state
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
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
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
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
            try:
                q_values = self.get_action_scores(context)
                action = int(q_values.argmax())
                score = float(q_values[action])
                
                # Confidence based on Q-value spread
                q_exp = np.exp(q_values - q_values.max())  # Numerical stability
                confidence = float(q_exp[action] / q_exp.sum())
                
                rankings.append((symbol, score, action, confidence))
            except Exception as e:
                logger.error(f"Error ranking {symbol}: {e}")
                continue
        
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
        
        # Feedback integration
        self.feedback_memory = FeedbackMemory()
        
        logger.info(f"RL Trainer initialized for {agent_type} agent")
        logger.info(f"Feature store contains {len(feature_store)} symbols")
    
    def incorporate_feedback(self, symbol: str, predicted_action: str, user_feedback: str, 
                           confidence: float = 0.0):
        """
        Incorporate user feedback into the RL agent
        
        Args:
            symbol: Stock symbol
            predicted_action: Action predicted by RL agent
            user_feedback: User's feedback (correct/incorrect)
            confidence: Confidence score of the prediction
        """
        # Get features for the symbol
        features = None
        if symbol in self.feature_store:
            df = self.feature_store[symbol]
            if not df.empty:
                feature_cols = self._get_feature_columns(df)
                if feature_cols:
                    features = df[feature_cols].iloc[-1].values
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use agent's feedback method if available
        if hasattr(self.agent, 'incorporate_feedback'):
            return self.agent.incorporate_feedback(symbol, predicted_action, user_feedback, confidence, features)
        else:
            # Fallback to trainer's feedback memory
            return self.feedback_memory.add_feedback(symbol, predicted_action, user_feedback, confidence, 
                                                   features.tolist() if features is not None else None)
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        if hasattr(self.agent, 'get_feedback_stats'):
            return self.agent.get_feedback_stats()
        else:
            return self.feedback_memory.get_feedback_stats()
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract feature columns (exclude OHLCV, metadata, targets, and timestamp columns)"""
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume', 'adj_close',
            'symbol', 'source', 'fetch_timestamp', 'date', 'Date',
            'target', 'target_return', 'target_direction', 'target_binary'
        ]
        return [col for col in df.columns if col not in exclude_cols]
    
    def _compute_reward(self, symbol: str, action: int, horizon: int = 1) -> float:
        """
        Compute reward  based on future returns (FIXED LOGIC)
        
        Args:
            symbol: Symbol identifier
            action: Action taken (for DQN: 0=short, 1=hold, 2=long)
            horizon: Prediction horizon
            
        Returns:
            Reward value
        """
        df = self.feature_store[symbol]
        
        # Use pre-computed target_return
        if 'target_return' not in df.columns:
            logger.warning(f"No target_return for {symbol}")
            return 0.0
        
        # Get the target return (already forward-looking from feature generation)
        # Use iloc[-1] to get the most recent row's target
        if len(df) < 1:
            return 0.0
        
        future_return = df['target_return'].iloc[-1]
        
        # Handle NaN
        if pd.isna(future_return):
            return 0.0
        
        if self.agent_type == "dqn":
            # DQN: align action with return direction
            if action == 2:  # long
                reward = future_return if future_return > 0 else 0  # Reward only correct longs
            elif action == 0:  # short
                reward = -future_return if future_return < 0 else 0  # Reward only correct shorts
            else:  # hold
                # Small reward for correctly holding in sideways markets
                reward = 0.001 if abs(future_return) < 0.01 else -abs(future_return) * 0.5
        else:
            # Bandit: direct return as reward
            reward = future_return
        
        # Scale reward to reasonable range
        return float(reward * 100)
    
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
                    feature_cols = self._get_feature_columns(df)
                    context = df[feature_cols].iloc[-1].values
                    
                    # Handle NaN/inf
                    context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
                    contexts[symbol] = context
            
            if not contexts:
                logger.warning("No valid contexts available")
                break
            
            # Rank and select top-k
            rankings = self.agent.rank_symbols(contexts, top_k=top_k)
            
            # Update based on observed rewards
            round_reward = 0
            for symbol, score, confidence in rankings:
                reward = self._compute_reward(symbol, action=2, horizon=horizon)
                self.agent.update(symbol, contexts[symbol], reward)
                round_reward += reward
            
            cumulative_reward += round_reward
            rewards_per_round.append(round_reward)
            
            if (round_num + 1) % 10 == 0:
                avg_reward = np.mean(rewards_per_round[-10:])
                logger.info(f"Round {round_num + 1}/{n_rounds} - Avg Reward: {avg_reward:.4f}")
        
        stats = {
            'cumulative_reward': cumulative_reward,
            'avg_reward': cumulative_reward / n_rounds if n_rounds > 0 else 0,
            'rewards_per_round': rewards_per_round,
            'total_rounds': n_rounds
        }
        
        self.training_history.append(stats)
        
        logger.info(f"Training complete. Avg reward: {stats['avg_reward']:.4f}")
        
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
            target_update_freq: Frequency to update target network
            horizon: Prediction horizon
            
        Returns:
            Training statistics
        """
        logger.info(f"Training DQN agent for {n_episodes} episodes")
        
        episode_rewards = []
        losses = []
        
        for episode in range(n_episodes):
            episode_reward = 0
            
            # Sample random symbols for this episode
            symbols = list(self.feature_store.keys())
            if not symbols:
                logger.warning("No symbols available for training")
                break
            
            # Random subset of symbols
            episode_symbols = random.sample(symbols, min(max_steps, len(symbols)))
            
            for step, symbol in enumerate(episode_symbols):
                if step >= max_steps:
                    break
                
                try:
                    df = self.feature_store[symbol]
                    if df.empty:
                        continue
                    
                    # Get features
                    feature_cols = self._get_feature_columns(df)
                    if not feature_cols:
                        continue
                    
                    state = df[feature_cols].iloc[-1].values
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Select action
                    action = self.agent.select_action(state)
                    
                    # Compute reward
                    reward = self._compute_reward(symbol, action, horizon)
                    
                    # Store transition
                    next_state = state  # Simplified for this implementation
                    done = step == len(episode_symbols) - 1
                    
                    self.agent.store_transition(state, action, reward, next_state, done)
                    
                    # Train
                    loss = self.agent.train_step()
                    if loss is not None:
                        losses.append(loss)
                    
                    episode_reward += reward
                    
                except Exception as e:
                    logger.error(f"Error in episode {episode}, step {step}: {e}")
                    continue
            
            episode_rewards.append(episode_reward)
            
            # Update target network
            if episode % target_update_freq == 0:
                self.agent.update_target_network()
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(f"Episode {episode + 1}/{n_episodes} - Avg Reward: {avg_reward:.4f}")
        
        stats = {
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'episode_rewards': episode_rewards,
            'losses': losses,
            'total_episodes': n_episodes
        }
        
        self.training_history.append(stats)
        
        logger.info(f"DQN training complete. Avg reward: {stats['avg_reward']:.4f}")
        
        return stats
    
    def evaluate(self, top_k: int = 20) -> Dict:
        """
        Evaluate agent performance on feature store
        
        Args:
            top_k: Number of top symbols to evaluate
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating {self.agent_type} agent")
        
        # Prepare contexts for all symbols
        contexts = {}
        symbol_data = {}
        
        for symbol, df in self.feature_store.items():
            if df.empty:
                continue
            
            try:
                feature_cols = self._get_feature_columns(df)
                if not feature_cols:
                    continue
                
                context = df[feature_cols].iloc[-1].values
                context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
                
                contexts[symbol] = context
                symbol_data[symbol] = df.iloc[-1]
                
            except Exception as e:
                logger.error(f"Error preparing {symbol}: {e}")
                continue
        
        if not contexts:
            logger.warning("No valid contexts for evaluation")
            return {
                'mean_return': 0.0,
                'sharpe_proxy': 0.0,
                'win_rate': 0.0,
                'top_symbols': []
            }
        
        # Rank symbols
        try:
            if hasattr(self.agent, 'action_dim'):
                # DQN agent
                rankings = self.agent.rank_symbols(contexts, top_k=top_k)
            else:
                # Bandit agent
                rankings = self.agent.rank_symbols(contexts, top_k=top_k)
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return {
                'mean_return': 0.0,
                'sharpe_proxy': 0.0,
                'win_rate': 0.0,
                'top_symbols': []
            }
        
        # Evaluate top symbols
        returns = []
        top_symbols = []
        
        for item in rankings[:top_k]:
            try:
                if hasattr(self.agent, 'action_dim'):
                    symbol, score, action_idx, confidence = item
                else:
                    symbol, score, confidence = item
                
                # Get actual return
                if symbol in symbol_data:
                    df = self.feature_store[symbol]
                    if 'target_return' in df.columns:
                        actual_return = df['target_return'].iloc[-1]
                        if not pd.isna(actual_return):
                            returns.append(actual_return)
                            top_symbols.append(symbol)
                
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}")
                continue
        
        if not returns:
            logger.warning("No valid returns for evaluation")
            return {
                'mean_return': 0.0,
                'sharpe_proxy': 0.0,
                'win_rate': 0.0,
                'top_symbols': []
            }
        
        # Calculate metrics
        mean_return = np.mean(returns)
        sharpe_proxy = mean_return / (np.std(returns) + 1e-10)
        win_rate = np.mean([r > 0 for r in returns])
        
        metrics = {
            'mean_return': float(mean_return),
            'sharpe_proxy': float(sharpe_proxy),
            'win_rate': float(win_rate),
            'top_symbols': top_symbols[:10]  # Top 10 symbols
        }
        
        logger.info(f"Evaluation complete:")
        logger.info(f"  Mean Return: {metrics['mean_return']:.4f}")
        logger.info(f"  Sharpe Proxy: {metrics['sharpe_proxy']:.4f}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
        
        return metrics