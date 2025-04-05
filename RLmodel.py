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
    def __init__(self, config=None):
        self.game_manager = GameManager()
        self.game_manager.start_new_game()
        self.config = {
        'simplified': False,  # Simplified rules for initial learning
        'full_features': False  # All game features enabled
        }
    
        # Update with provided config
        if config:
            self.config.update(config)
        
        self.game_manager = GameManager()
        self.game_manager.start_new_game()
        
        # Modify game settings based on config
        if self.config['simplified']:
            # Apply simplified settings
            self.game_manager.game.is_boss_blind = False
            self.game_manager.max_hands_per_round = 4

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
    
    def _encode_cards(self, cards):
        encoded = []
        for card in cards:
            # Normalize rank (2-14) -> 0-1
            rank_feature = (card.rank.value - 2) / 12
            
            # One-hot encode suit
            suit_features = [0, 0, 0, 0]
            suit_features[card.suit.value - 1] = 1
            
            # Enhancement feature
            enhancement_feature = card.enhancement.value / len(CardEnhancement)
            
            # Additional state data
            is_scored = 1.0 if card.scored else 0.0
            is_face = 1.0 if card.face else 0.0
            is_debuffed = 1.0 if hasattr(card, 'debuffed') and card.debuffed else 0.0
            
            card_features = [rank_feature] + suit_features + [enhancement_feature, is_scored, is_face, is_debuffed]
            encoded.extend(card_features)
        
        # Pad to fixed length if needed
        return np.array(encoded)


    def _encode_boss_blind_effect(self):
        """Encode the boss blind effect as a feature vector."""
        # One-hot encode the boss blind effect
        encoded = [0] * (len(BossBlindEffect) + 1)  # +1 for "no effect"
        
        if not self.game_manager.game.is_boss_blind:
            encoded[0] = 1  # No effect
        else:
            effect_value = self.game_manager.game.active_boss_blind_effect.value
            encoded[effect_value] = 1
            
        return encoded

    def _encode_joker_effects(self):
        """Encode joker effects as a feature vector."""
        # This is a simplified encoding - you may want more detailed features
        joker_count = len(self.game_manager.game.inventory.jokers)
        
        # Track special jokers that have important effects
        has_mr_bones = 0
        has_green_joker = 0
        has_socks_buskin = 0
        boss_blind_counters = 0
        
        for joker in self.game_manager.game.inventory.jokers:
            if joker.name == "Mr. Bones":
                has_mr_bones = 1
            elif joker.name == "Green Joker":
                has_green_joker = 1
            elif joker.name == "Socks and Buskin":
                has_socks_buskin = 1
            elif joker.name == "Rocket" and hasattr(joker, "boss_blind_defeated"):
                boss_blind_counters = joker.boss_blind_defeated
        
        return [joker_count, has_mr_bones, has_green_joker, has_socks_buskin, boss_blind_counters]


    def _encode_jokers(self, jokers):
        """
        Encode jokers into a feature vector for the neural network.
        
        Args:
            jokers: List of joker objects
            
        Returns:
            numpy array of encoded joker features
        """
        max_jokers = 5 
        features_per_joker = 6  
        
        encoded = []
        
        for joker in jokers:
            # Rarity encoding
            rarity_feature = 0.0
            if joker.rarity == 'base':
                rarity_feature = 0.33
            elif joker.rarity == 'uncommon':
                rarity_feature = 0.66
            elif joker.rarity == 'rare':
                rarity_feature = 1.0
            
            # Sell value (normalized)
            sell_value = joker.sell_value / 5.0
            
            has_mult_effect = 1.0 if joker.mult_effect > 0 else 0.0
            has_chips_effect = 1.0 if joker.chips_effect > 0 else 0.0
            
            has_boss_counter = 1.0 if hasattr(joker, 'boss_blind_defeated') and joker.boss_blind_defeated > 0 else 0.0
            is_retrigger = 1.0 if joker.retrigger else 0.0
            
            joker_features = [rarity_feature, sell_value, has_mult_effect, 
                            has_chips_effect, has_boss_counter, is_retrigger]
            
            encoded.extend(joker_features)
        
        padding_needed = max_jokers - len(jokers)
        if padding_needed > 0:
            encoded.extend([0.0] * (padding_needed * features_per_joker))
        
        return np.array(encoded)

    def _encode_consumables(self):
        """
        Encode consumables (tarot and planet cards) into a feature vector.
        
        Returns:
            numpy array of encoded consumable features
        """
        max_consumables = 2 
        features_per_consumable = 5
        
        encoded = []
        
        tarot_indices = self.game_manager.game.inventory.get_consumable_tarot_indices()
        planet_indices = self.game_manager.game.inventory.get_consumable_planet_indices()
        
        for idx in range(len(self.game_manager.game.inventory.consumables)):
            if idx >= max_consumables:
                break
                
            consumable = self.game_manager.game.inventory.consumables[idx]
            
            is_planet = 1.0 if consumable.type == ConsumableType.PLANET else 0.0
            
            # Cards required (normalized)
            cards_required = 0.0
            if consumable.type == ConsumableType.TAROT and hasattr(consumable.item, 'selected_cards_required'):
                cards_required = consumable.item.selected_cards_required / 3.0  # Assume max is 3
            
            # Basic effects (simplified)
            affects_cards = 0.0
            affects_money = 0.0
            affects_game = 0.0
            
            if consumable.type == ConsumableType.TAROT:
                tarot_type = consumable.item.tarot_type
                
                # Determine effect type based on tarot
                if tarot_type in [TarotType.THE_MAGICIAN, TarotType.THE_EMPRESS, 
                                TarotType.THE_HIEROPHANT, TarotType.THE_LOVERS,
                                TarotType.THE_CHARIOT, TarotType.JUSTICE, 
                                TarotType.STRENGTH, TarotType.THE_DEVIL, 
                                TarotType.THE_TOWER, TarotType.THE_STAR,
                                TarotType.THE_MOON, TarotType.THE_SUN,
                                TarotType.THE_WORLD]:
                    affects_cards = 1.0
                
                if tarot_type in [TarotType.THE_HERMIT, TarotType.TEMPERANCE]:
                    affects_money = 1.0
                    
                if tarot_type in [TarotType.THE_FOOL, TarotType.THE_HIGH_PRIESTESS,
                                TarotType.THE_EMPEROR, TarotType.WHEEL_OF_FORTUNE,
                                TarotType.JUDGEMENT]:
                    affects_game = 1.0
            
            elif consumable.type == ConsumableType.PLANET:
                affects_game = 1.0  # All planets affect the game scoring
            
            consumable_features = [is_planet, cards_required, affects_cards, 
                                affects_money, affects_game]
            
            encoded.extend(consumable_features)
        
        # Pad with zeros if fewer than max consumables
        padding_needed = max_consumables - len(self.game_manager.game.inventory.consumables)
        if padding_needed > 0:
            encoded.extend([0.0] * (padding_needed * features_per_consumable))
        
        return np.array(encoded)

    def _encode_shop_items(self):
        """
        Encode shop items into a feature vector.
        
        Returns:
            numpy array of encoded shop item features
        """
        # Assuming shop is accessible through game_manager
        # If not, you'll need to adjust this based on your code structure
        # For example, you might need a self.shop attribute
        
        # For the purpose of this code, I'll assume a shop exists and has items
        max_shop_items = 4  # Standard shop size
        features_per_item = 6  # Number of features per shop item
        
        encoded = []
        
        # Check if we have a shop reference
        shop = None
        if hasattr(self, 'shop'):
            shop = self.shop
        elif hasattr(self.game_manager, 'shop'):
            shop = self.game_manager.shop
        
        if shop is None:
            # No shop reference, return zeros
            return np.zeros(max_shop_items * features_per_item)
        
        for i in range(max_shop_items):
            # Check if slot has an item
            has_item = 1.0 if i < len(shop.items) and shop.items[i] is not None else 0.0
            
            # Default values
            item_type = 0.0
            price = 0.0
            can_afford = 0.0
            is_joker = 0.0
            is_consumable = 0.0
            
            if has_item:
                shop_item = shop.items[i]
                
                # Item type (normalized)
                item_type = shop_item.item_type.value / len(ShopItemType)
                
                # Price and affordability
                price = shop.get_item_price(i) / 10.0  # Normalize (assume max price 10)
                can_afford = 1.0 if self.game_manager.game.inventory.money >= price else 0.0
                
                # Item category
                is_joker = 1.0 if shop_item.item_type == ShopItemType.JOKER else 0.0
                is_consumable = 1.0 if shop_item.item_type in [ShopItemType.TAROT, ShopItemType.PLANET] else 0.0
            
            item_features = [has_item, item_type, price, can_afford, is_joker, is_consumable]
            encoded.extend(item_features)
        
        return np.array(encoded)

    def _encode_planet_levels(self):
        """
        Encode planet levels into a feature vector.
        
        Returns:
            numpy array of encoded planet level features
        """
        # Get planet levels
        encoded = []
        
        # Process each planet type
        for planet_type in PlanetType:
            # Get current level
            level = self.game_manager.game.inventory.planet_levels.get(planet_type, 1)
            
            # Normalize
            normalized_level = (level - 1) / 9.0  # Assuming max level is 10
            
            encoded.append(normalized_level)
        
        return np.array(encoded)

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

    def prioritized_replay(self, batch_size, alpha=0.6, beta=0.4):
        if len(self.memory) < batch_size:
            return
        
        # Calculate TD errors for all experiences in memory
        td_errors = []
        for state, action, reward, next_state, done in self.memory:
            state_array = np.array(state).reshape(1, -1)
            next_state_array = np.array(next_state).reshape(1, -1)
            
            current_q = self.model.predict(state_array, verbose=0)[0][action]
            if done:
                target_q = reward
            else:
                next_q = np.max(self.target_model.predict(next_state_array, verbose=0)[0])
                target_q = reward + self.gamma * next_q
            
            td_error = abs(target_q - current_q)
            td_errors.append(td_error)
        
        # Convert errors to priorities and probabilities
        priorities = np.power(np.array(td_errors) + 1e-6, alpha)
        probs = priorities / np.sum(priorities)
        
        # Sample according to priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # Get sampled experiences
        states = np.array([self.memory[i][0] for i in indices])
        actions = np.array([self.memory[i][1] for i in indices])
        rewards = np.array([self.memory[i][2] for i in indices])
        next_states = np.array([self.memory[i][3] for i in indices])
        dones = np.array([self.memory[i][4] for i in indices])
        
        # Calculate importance sampling weights
        weights = np.power(len(self.memory) * probs[indices], -beta)
        weights /= np.max(weights)  # Normalize
        
        # Update model with importance sampling weights
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Use custom loss with importance sampling weights
        history = self.model.fit(states, targets, epochs=1, verbose=0, 
                                sample_weight=weights)
        
        return history.history['loss'][0]


def get_valid_play_actions(self):
    """Return valid play actions based on current game state"""
    valid_actions = []
    
    # If we can still play hands this round
    if self.game_manager.hands_played < self.game_manager.max_hands_per_round:
        # Actions to play cards
        for i in range(min(256, 2**len(self.game_manager.current_hand))):
            valid_actions.append(i)  # Play action
    
    # If we can still discard cards
    if self.game_manager.discards_used < self.game_manager.max_discards_per_round:
        # Actions to discard cards
        for i in range(min(256, 2**len(self.game_manager.current_hand))):
            valid_actions.append(i + 256)  # Discard action
    
    return valid_actions

def get_valid_strategy_actions(self):
    """Return valid strategy actions based on current game state"""
    valid_actions = []
    
    # Check which shop items we can afford
    for i in range(4):  # 4 shop slots
        if self.shop.items[i] is not None and self.game_manager.game.inventory.money >= self.shop.get_item_price(i):
            valid_actions.append(i)  # Buy shop item
    
    # Check if we can sell jokers
    for i in range(len(self.game_manager.game.inventory.jokers)):
        valid_actions.append(i + 4)  # Sell joker
    
    # Check if we can use tarot cards
    tarot_indices = self.game_manager.game.inventory.get_consumable_tarot_indices()
    for i, tarot_idx in enumerate(tarot_indices):
        if i < 2:  # Limit to first 2 tarots for simplicity
            valid_actions.append(9 + i*3)  # Use tarot with no cards
            valid_actions.append(10 + i*3)  # Use tarot with lowest cards
            valid_actions.append(11 + i*3)  # Use tarot with highest cards
    
    # Always valid to skip
    valid_actions.append(15)  # Skip action
    
    return valid_actions

def train_with_curriculum():
    # Phase 1: Train with simplified game (no boss blinds, fixed deck)
    print("Phase 1: Learning basic card play...")
    play_agent, _ = train_agents(episodes=2000, game_config={'simplified': True})
    
    # Phase 2: Include boss blinds and shop interaction
    print("Phase 2: Adding boss blinds and shop strategy...")
    play_agent, strategy_agent = train_agents(episodes=3000, 
                                             game_config={'simplified': False},
                                             play_agent=play_agent)
    
    # Phase 3: Full game with enhanced exploration
    print("Phase 3: Full game training...")
    play_agent, strategy_agent = train_agents(episodes=5000,
                                             game_config={'full_features': True},
                                             play_agent=play_agent,
                                             strategy_agent=strategy_agent)
    
    return play_agent, strategy_agent





def train_agents(episodes=10000, batch_size=64, game_config=None, save_interval=500, play_agent=None, strategy_agent=None, log_interval=100):
    if game_config is None:
        game_config = {}
    
    # Print what configuration is being used
    print(f"Training with config: {game_config}")
    
    env = BalatroEnv()
    
    # Use provided agents or create new ones
    if play_agent is None:
        play_agent = PlayingAgent(state_size=len(env._get_play_state()), 
                                  action_size=env._define_play_action_space())
    
    if strategy_agent is None:  
        strategy_agent = StrategyAgent(state_size=len(env._get_strategy_state()), 
                                      action_size=env._define_strategy_action_space())
    
    # Initialize state sizes
    play_state = env._get_play_state()
    play_state_size = len(np.array(play_state).flatten()) if isinstance(play_state, (list, np.ndarray)) else len(play_state)
    
    strategy_state = env._get_strategy_state()
    strategy_state_size = len(np.array(strategy_state).flatten()) if isinstance(strategy_state, (list, np.ndarray)) else len(strategy_state)
    
    # Initialize agents if not provided
    if play_agent is None:
        play_agent = PlayingAgent(state_size=play_state_size, 
                                  action_size=env._define_play_action_space())
    
    if strategy_agent is None:
        strategy_agent = StrategyAgent(state_size=strategy_state_size, 
                                      action_size=env._define_strategy_action_space())
    
    # Training stats
    play_rewards = []
    strategy_rewards = []
    ante_progression = []
    
    for e in range(episodes):
        play_state = env.reset()
        play_total_reward = 0
        strategy_total_reward = 0
        
        # Track game progress
        max_ante_reached = 1
        game_steps = 0
        
        done = False
        while not done and game_steps < 1000:  # Safety limit
            game_steps += 1
            
            # Playing phase
            valid_play_actions = env.get_valid_play_actions()  # Implement this helper method
            play_action = play_agent.act(play_state, valid_actions=valid_play_actions)
            next_play_state, play_reward, done, info = env.step_play(play_action)
            
            # Remember experience
            play_agent.remember(play_state, play_action, play_reward, next_play_state, done)
            play_state = next_play_state
            play_total_reward += play_reward
            
            # Track performance data
            if hasattr(env.game_manager, 'hand_result') and env.game_manager.hand_result:
                play_agent.track_hand_played(env.game_manager.hand_result, env.game_manager.played_cards)
            
            max_ante_reached = max(max_ante_reached, env.game_manager.game.current_ante)
            
            # Strategy phase (after beating an ante)
            if env.game_manager.current_ante_beaten and not done:
                strategy_state = env._get_strategy_state()
                valid_strategy_actions = env.get_valid_strategy_actions()  # Implement this helper
                strategy_action = strategy_agent.act(strategy_state, valid_strategy_actions)
                
                next_strategy_state, strategy_reward, strategy_done, _ = env.step_strategy(strategy_action)
                strategy_agent.remember(strategy_state, strategy_action, strategy_reward, 
                                       next_strategy_state, strategy_done)
                
                strategy_total_reward += strategy_reward
                done = strategy_done
                
                # Reset play state after strategy phase
                if not done:
                    play_state = env._get_play_state()
        
        # Learn from experiences
        if len(play_agent.memory) > 64:
            play_loss = play_agent.replay(64)
        
        if len(strategy_agent.memory) > 64:
            strategy_loss = strategy_agent.replay(64)
        
        # Decay exploration rates
        play_agent.decay_epsilon()
        strategy_agent.decay_epsilon()
        
        # Record stats
        play_rewards.append(play_total_reward)
        strategy_rewards.append(strategy_total_reward)
        ante_progression.append(max_ante_reached)
        
        # Logging and saving
        if (e + 1) % log_interval == 0:
            avg_play_reward = sum(play_rewards[-log_interval:]) / log_interval
            avg_strategy_reward = sum(strategy_rewards[-log_interval:]) / log_interval
            avg_ante = sum(ante_progression[-log_interval:]) / log_interval
            
            print(f"Episode {e+1}/{episodes}")
            print(f"  Play Agent: reward={avg_play_reward:.2f}, epsilon={play_agent.epsilon:.3f}")
            print(f"  Strategy Agent: reward={avg_strategy_reward:.2f}, epsilon={strategy_agent.epsilon:.3f}")
            print(f"  Average max ante: {avg_ante:.2f}")
            
            # Print additional stats
            play_stats = play_agent.get_stats()
            print(f"  Play stats: {play_stats}")
        
        if (e + 1) % save_interval == 0:
            print(f"Saving models at episode {e+1}")
            play_agent.save_model(f"play_agent_ep{e+1}.h5")
            strategy_agent.save_model(f"strategy_agent_ep{e+1}.h5")
            
            # Also save final models with a fixed name for easier loading
            play_agent.save_model("play_agent_latest.h5")
            strategy_agent.save_model("strategy_agent_latest.h5")
    
    # Save final models
    play_agent.save_model("play_agent_final.h5")
    strategy_agent.save_model("strategy_agent_final.h5")
    
    return play_agent, strategy_agent



def evaluate_agents(play_agent, strategy_agent, episodes=100):
    """Evaluate agent performance without exploration"""
    env = BalatroEnv()
    
    # Save original epsilon values
    play_epsilon = play_agent.epsilon
    strategy_epsilon = strategy_agent.epsilon
    
    # Disable exploration for evaluation
    play_agent.epsilon = 0
    strategy_agent.epsilon = 0
    
    results = {
        'play_rewards': [],
        'strategy_rewards': [],
        'max_antes': [],
        'hands_played': [],
        'win_rate': 0,
        'average_score': 0
    }
    
    for e in range(episodes):
        play_state = env.reset()
        play_total_reward = 0
        strategy_total_reward = 0
        
        max_ante = 1
        hands_played = 0
        game_won = False
        
        done = False
        while not done:
            # Playing phase
            play_action = play_agent.act(play_state)
            next_play_state, play_reward, done, _ = env.step_play(play_action)
            play_state = next_play_state
            play_total_reward += play_reward
            
            hands_played += 1
            max_ante = max(max_ante, env.game_manager.game.current_ante)
            
            # Strategy phase
            if env.game_manager.current_ante_beaten and not done:
                strategy_state = env._get_strategy_state()
                strategy_action = strategy_agent.act(strategy_state)
                
                next_strategy_state, strategy_reward, done, _ = env.step_strategy(strategy_action)
                strategy_total_reward += strategy_reward
                
                if not done:
                    play_state = env._get_play_state()
        
        # Record results
        results['play_rewards'].append(play_total_reward)
        results['strategy_rewards'].append(strategy_total_reward)
        results['max_antes'].append(max_ante)
        results['hands_played'].append(hands_played)
        
        # Consider a win if player reached ante 8
        if max_ante >= 8:
            game_won = True
        results['win_rate'] += 1 if game_won else 0
    
    # Calculate averages
    results['win_rate'] = results['win_rate'] / episodes * 100
    results['average_score'] = sum(results['max_antes']) / episodes
    
    # Restore original epsilon values
    play_agent.epsilon = play_epsilon
    strategy_agent.epsilon = strategy_epsilon
    
    return results


if __name__ == "__main__":
    # Train from scratch
    play_agent, strategy_agent = train_with_curriculum()
    
    # Or load pre-trained models and continue training
    # play_agent = PlayingAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    # play_agent.load_model("play_agent_latest.h5")
    # strategy_agent = StrategyAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    # strategy_agent.load_model("strategy_agent_latest.h5")
    
    # play_agent, strategy_agent = train_agents(episodes=5000, 
    #                                          play_agent=play_agent,
    #                                          strategy_agent=strategy_agent)
    
    # Evaluate the trained agents
    results = evaluate_agents(play_agent, strategy_agent)
    print("Evaluation Results:")
    print(f"  Win Rate: {results['win_rate']:.2f}%")
    print(f"  Average Max Ante: {results['average_score']:.2f}")
    print(f"  Average Hands Played: {sum(results['hands_played'])/len(results['hands_played']):.2f}")