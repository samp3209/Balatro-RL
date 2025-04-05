from collections import deque
from GameManager import GameManager
import numpy as np
from Inventory import Inventory
from Card import Card
from Enums import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random


class BalatroEnv:
    def __init__(self):
        self.game_manager = GameManager()
        self.game_manager.start_new_game()
        

    def reset(self):
        self.game_manager = GameManager()
        self.game_manager.start_new_game()
        return self._get_play_state()
    

    def step_play(self, action):
        # Process a playing action (select cards to play/discard)
        card_indices = self._convert_action_to_card_indices(action)
        if self.is_discard_action(action):
            success, message = self.game_manager.discard_cards(card_indices)
        else:
            success, message = self.game_manager.play_cards(card_indices)
        
        reward = self._calculate_play_reward()
        done = self.game_manager.game_over or self.game_manager.current_ante_beaten
        next_state = self._get_play_state()
        
        return next_state, reward, done, {"message": message}
    

    def step_strategy(self, action):
        # Process a strategy action (shop purchase, tarot use, etc.)
        # ...
        
        reward = self._calculate_strategy_reward()
        done = self.game_manager.game_over
        next_state = self._get_strategy_state()
        
        return next_state, reward, done, {}
    

    def _get_play_state(self):
        state = {
            "cards_in_hand": self._encode_cards(self.game_manager.current_hand),
            "hand_size": len(self.game_manager.current_hand),
            
            "current_ante": self.game_manager.game.current_ante,
            "current_blind": self.game_manager.game.current_blind,
            "current_score": self.game_manager.current_score,
            "hands_played": self.game_manager.hands_played,
            "max_hands": self.game_manager.max_hands_per_round,
            "discards_used": self.game_manager.discards_used,
            "max_discards": self.game_manager.max_discards_per_round,
            
            "is_boss_blind": self.game_manager.game.is_boss_blind,
            "boss_blind_effect": self._encode_boss_blind_effect(),
            
            "joker_effects": self._encode_joker_effects(),
        }

        return state
    
    def _get_strategy_state(self):
        state = {
            "current_ante": self.game_manager.game.current_ante,
            "current_blind": self.game_manager.game.current_blind,
            "money": self.game_manager.game.inventory.money,
            
            "jokers": self._encode_jokers(self.game_manager.game.inventory.jokers),
            "joker_count": len(self.game_manager.game.inventory.jokers),
            "consumables": self._encode_consumables(),
            
            "shop_items": self._encode_shop_items(),
            
            "planet_levels": self._encode_planet_levels(),
        }

        return state
    
    def _define_play_action_space(self):
        # 0 or 1 (play vs discard)
        # 2^8 possible actions
        return 512 

    def _define_strategy_action_space(self):
        # Buy item from shop 4 slots
        # Sell joker up to 5
        # Use tarot at most 8 card selection
        # skip
        return 26

    def _calculate_play_reward(self):
        score_fraction = self.game_manager.current_score / self.game_manager.game.current_blind
        
        if self.game_manager.hand_result:
            hand_quality = self.game_manager.hand_result.value / 9.0  
            hand_reward = hand_quality * 0.5
        else:
            hand_reward = 0
        
        discard_penalty = -0.05 * self.last_action_was_discard
        
        wasted_hand_penalty = -0.1 if score_fraction < 0.1 else 0
        
        ante_beaten_reward = 1.0 if self.game_manager.current_ante_beaten else 0
        
        game_over_penalty = -1.0 if self.game_manager.game_over else 0
        
        return score_fraction + hand_reward + discard_penalty + wasted_hand_penalty + ante_beaten_reward + game_over_penalty
    
    def _calculate_strategy_reward(self):
        ante_progress = 0.2 * self.game_manager.game.current_ante
        
        joker_quality = sum(j.sell_value for j in self.game_manager.game.inventory.jokers) * 0.02
        
        money_reward = 0.01 * self.game_manager.game.inventory.money
        
        planet_level_reward = 0.05 * sum(self.game_manager.game.inventory.planet_levels.values())
        
        game_over_penalty = -5.0 if self.game_manager.game_over else 0
        
        return ante_progress + joker_quality + money_reward + planet_level_reward + game_over_penalty
    

class PlayingAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95                  # Discount factor
        self.epsilon = 1.0                 # Exploration rate
        self.epsilon_min = 0.01            # Minimum exploration probability
        self.epsilon_decay = 0.995         # Decay rate for exploration
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # For tracking performance metrics
        self.recent_rewards = deque(maxlen=100)
        self.recent_hands = deque(maxlen=100)  # Track hand types played
        self.recent_discards = deque(maxlen=100)  # Track discard frequencies
        
        # Action categories
        self.PLAY_ACTION = 0
        self.DISCARD_ACTION = 1
        
    def _build_model(self):
        """Build a neural network for predicting Q-values"""
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Track rewards for analytics
        self.recent_rewards.append(reward)
    
    def act(self, state, valid_actions=None):
        """Choose an action using epsilon-greedy policy"""
        # Convert state to array if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state).reshape(1, -1)
        else:
            state = state.reshape(1, -1)
        
        # Explore: choose random action
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        # Exploit: choose best action
        act_values = self.model.predict(state, verbose=0)
        
        # Handle valid actions mask if provided
        if valid_actions is not None and len(valid_actions) > 0:
            mask = np.ones(self.action_size) * -1000000  # Large negative value
            for action in valid_actions:
                mask[action] = 0
            act_values = act_values + mask
        
        return np.argmax(act_values[0])
    
    def decode_action(self, action):
        """Decode an action into play/discard and card indices"""
        # First bit determines if this is a play or discard action
        action_type = action // 256  # For 8 cards, we have 2^8=256 possible combinations
        
        # Remaining bits determine which cards to select
        card_mask = action % 256
        
        # Convert to binary representation
        binary = format(card_mask, '08b')  # 8-bit binary representation
        card_indices = [i for i, bit in enumerate(reversed(binary)) if bit == '1']
        
        action_name = "Discard" if action_type == self.DISCARD_ACTION else "Play"
        return action_type, card_indices, f"{action_name} cards {card_indices}"
    
    def replay(self, batch_size):
        """Train the network using experiences from memory"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Extract data
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict current Q values
        targets = self.model.predict(states, verbose=0)
        
        # Get next Q values from target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q values for the actions taken
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        return history.history['loss'][0]
    
    def decay_epsilon(self):
        """Reduce exploration rate over time"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, file_path):
        """Save model to file"""
        self.model.save(file_path)
    
    def load_model(self, file_path):
        """Load model from file"""
        self.model = tf.keras.models.load_model(file_path)
        self.update_target_model()
    
    def get_stats(self):
        """Return performance statistics"""
        if not self.recent_rewards:
            return {"avg_reward": 0}
            
        return {
            "avg_reward": sum(self.recent_rewards) / len(self.recent_rewards),
            "epsilon": self.epsilon,
            "memory_size": len(self.memory)
        }
    
    def track_hand_played(self, hand_type, cards_played):
        """Track what hands the agent is playing"""
        self.recent_hands.append((hand_type, len(cards_played)))
    
    def track_discard(self, cards_discarded):
        """Track discards for analysis"""
        self.recent_discards.append(len(cards_discarded))
    
    def encode_card_state(self, cards):
        """Encode cards into a feature vector"""
        # This would create a flattened representation of cards
        # that captures their rank, suit, and enhancements
        
        # Example implementation (simplified)
        card_features = []
        
        for card in cards:
            # Normalize rank (2-14) to 0-1 range
            rank_feature = (card.rank.value - 2) / 12
            
            # One-hot encode suit (4 suits)
            suit_features = [0, 0, 0, 0]
            if card.suit == Suit.HEARTS:
                suit_features[0] = 1
            elif card.suit == Suit.DIAMONDS:
                suit_features[1] = 1
            elif card.suit == Suit.CLUBS:
                suit_features[2] = 1
            elif card.suit == Suit.SPADES:
                suit_features[3] = 1
            
            # Enhancement features
            enhancement_feature = card.enhancement.value / len(CardEnhancement)
            
            # Additional card state
            is_face = 1.0 if card.face else 0.0
            is_debuffed = 1.0 if card.debuffed else 0.0
            
            # Combine all features for this card
            card_feature_vector = [rank_feature] + suit_features + [enhancement_feature, is_face, is_debuffed]
            card_features.extend(card_feature_vector)
        
        return np.array(card_features)
    
    def evaluate_all_possible_plays(self, hand, evaluator):
        """
        Evaluate all possible card combinations to find the best hand
        This can be used to supplement RL training with expert demonstrations
        """
        best_score = 0
        best_cards = []
        best_hand_type = None
        
        # Generate all possible combinations of cards (power set)
        n = len(hand)
        for i in range(1, 2**n):
            # Convert number to binary to determine which cards to include
            binary = bin(i)[2:].zfill(n)
            selected_cards = [hand[j] for j in range(n) if binary[j] == '1']
            
            # Skip if fewer than 5 cards (minimum for most poker hands)
            if len(selected_cards) < 5:
                continue
                
            # Evaluate this combination
            hand_type, _, scoring_cards = evaluator.evaluate_hand(selected_cards)
            
            # Calculate a score (e.g., hand type value * number of scoring cards)
            score = hand_type.value * len(scoring_cards)
            
            if score > best_score:
                best_score = score
                best_cards = selected_cards
                best_hand_type = hand_type
        
        return best_cards, best_hand_type, best_score
    
    def adaptive_exploration(self, state, valid_actions, episode):
        """
        More sophisticated exploration strategy that adapts based on 
        training progress and state characteristics
        """
        # Base epsilon from standard decay
        current_epsilon = self.epsilon
        
        # Modify epsilon based on episode number (explore more in early training)
        episode_factor = max(0.1, min(1.0, 5000 / (episode + 5000)))
        
        hand_quality = self._estimate_hand_quality(state)
        quality_factor = max(0.5, 1.5 - hand_quality)
        
        adaptive_epsilon = current_epsilon * episode_factor * quality_factor
        
        if np.random.rand() <= adaptive_epsilon:
            return random.choice(valid_actions)
        
        state_array = np.array(state).reshape(1, -1)
        act_values = self.model.predict(state_array, verbose=0)
        
        mask = np.ones(self.action_size) * -1000000
        for action in valid_actions:
            mask[action] = 0
        act_values = act_values + mask
        
        return np.argmax(act_values[0])
    
    def _estimate_hand_quality(self, state):
        """Estimate the quality of the current hand (helper for adaptive_exploration)"""
        # This would extract and analyze card information from the state
        # to estimate how good the current hand is
        # Return a value between 0 (poor hand) and 1 (excellent hand)
        return 0.5  # Placeholder implementation
    
    def prioritized_experience_replay(self, batch_size, alpha=0.6, beta=0.4):
        """
        Prioritized Experience Replay implementation for more efficient learning
        - Samples experiences with higher TD-errors more frequently
        - Uses importance sampling to correct bias
        """
        # This would be a more advanced implementation than the basic replay
        # Placeholder for the full implementation
        pass

class StrategyAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000) 
        self.gamma = 0.99                 
        self.epsilon = 1.0                 
        self.epsilon_min = 0.05            
        self.epsilon_decay = 0.998         
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.recent_rewards = deque(maxlen=100)
        self.recent_actions = deque(maxlen=100)
        
        self.action_map = {
            0: "Buy Shop Item 0",
            1: "Buy Shop Item 1",
            2: "Buy Shop Item 2",
            3: "Buy Shop Item 3",
            
            4: "Sell Joker 0",
            5: "Sell Joker 1",
            6: "Sell Joker 2",
            7: "Sell Joker 3",
            8: "Sell Joker 4",
            
            9: "Use Tarot 0 with no cards",
            10: "Use Tarot 0 with lowest cards",
            11: "Use Tarot 0 with highest cards",
            12: "Use Tarot 1 with no cards",
            13: "Use Tarot 1 with lowest cards",
            14: "Use Tarot 1 with highest cards",
            
            15: "Skip (Do Nothing)"
        }
    
    def _build_model(self):
        """Build a neural network model for deep Q-learning."""
        
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        self.recent_rewards.append(reward)
        self.recent_actions.append(action)
    
    def act(self, state, valid_actions=None):
        """Choose an action based on the current state"""
        state_array = np.array(state).reshape(1, -1)
        
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None:
                return random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state_array, verbose=0)
        
        if valid_actions is not None:
            mask = np.ones(self.action_size) * -1000000
            mask[valid_actions] = 0
            act_values = act_values + mask
        
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train the agent with experiences from memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        targets = self.model.predict(states, verbose=0)
        
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
    
    def act_with_explanation(self, state, valid_actions=None):
        """Choose an action and provide explanation (for debugging)"""
        action = self.act(state, valid_actions)
        explanation = self.action_map.get(action, f"Unknown action {action}")
        
        state_array = np.array(state).reshape(1, -1)
        q_values = self.model.predict(state_array, verbose=0)[0]
        
        return action, explanation, q_values
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, file_path):
        """Save the model to disk"""
        self.model.save(file_path)
    
    def load_model(self, file_path):
        """Load the model from disk"""
        self.model = tf.keras.models.load_model(file_path)
        self.update_target_model()
    
    def get_stats(self):
        """Return current performance stats"""
        if not self.recent_rewards:
            return {"avg_reward": 0, "action_distribution": {}}
        
        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
        
        action_counts = {}
        for action in self.recent_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        action_distribution = {
            self.action_map.get(action, f"Action {action}"): count / len(self.recent_actions) * 100
            for action, count in action_counts.items()
        }
        
        return {
            "avg_reward": avg_reward,
            "action_distribution": action_distribution,
            "epsilon": self.epsilon
        }



def train_agents():
    env = BalatroEnv()
    play_agent = PlayingAgent(state_size=len(env._get_play_state()), action_size=env._define_play_action_space())
    strategy_agent = StrategyAgent(state_size=len(env._get_strategy_state()), action_size=env._define_strategy_action_space())
    
    num_episodes = 10000
    batch_size = 64
    
    for e in range(num_episodes):
        play_state = env.reset()
        play_total_reward = 0
        strategy_total_reward = 0
        
        while True:
            # Playing phase
            play_action = play_agent.act(play_state)
            next_play_state, play_reward, done, _ = env.step_play(play_action)
            play_agent.remember(play_state, play_action, play_reward, next_play_state, done)
            play_state = next_play_state
            play_total_reward += play_reward
            
            if done:
                if env.game_manager.current_ante_beaten:
                    strategy_state = env._get_strategy_state()
                    strategy_action = strategy_agent.act(strategy_state)
                    next_strategy_state, strategy_reward, done, _ = env.step_strategy(strategy_action)
                    strategy_agent.remember(strategy_state, strategy_action, strategy_reward, next_strategy_state, done)
                    strategy_total_reward += strategy_reward
                    
                    if not done:
                        play_state = env._get_play_state()
                        continue
                
                break
            
        if len(play_agent.memory) > batch_size:
            play_agent.replay(batch_size)
        
        if len(strategy_agent.memory) > batch_size:
            strategy_agent.replay(batch_size)
            
        if play_agent.epsilon > play_agent.epsilon_min:
            play_agent.epsilon *= play_agent.epsilon_decay
            
        if strategy_agent.epsilon > strategy_agent.epsilon_min:
            strategy_agent.epsilon *= strategy_agent.epsilon_decay
        
        print(f"Episode: {e+1}/{num_episodes}, Play Reward: {play_total_reward}, Strategy Reward: {strategy_total_reward}")