from collections import deque
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from GameManager import GameManager
from Inventory import Inventory
from Card import Card
from Enums import *
from Shop import Shop, ShopItem, ShopItemType, initialize_shops_for_game, FixedShop

class BalatroEnv:
    def __init__(self, config=None):
        self.config = {
            'simplified': False,  # Simplified rules for initial learning
            'full_features': False  # All game features enabled
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        self.game_manager = GameManager()
        self.game_manager.start_new_game()
        self.last_action_was_discard = False
        
        self.all_shops = initialize_shops_for_game()
        self.current_shop = None
        self.pending_tarots = []
        
        self.episode_step = 0
        self.episode_max_blind = 0
        self.max_steps_per_episode = 500  # Safety limit
        self.in_shop_phase = False
        
        if self.config['simplified']:
            self.game_manager.game.is_boss_blind = False
            self.game_manager.max_hands_per_round = 4

    def reset(self):
        """Reset the environment to initial state"""
        self.game_manager = GameManager()
        self.game_manager.start_new_game()
        self.last_action_was_discard = False
        self.episode_step = 0
        self.episode_max_blind = 0
        self.pending_tarots = []
        self.in_shop_phase = False
        
        self.update_shop()
        
        print(f"\n===== STARTING NEW EPISODE =====")
        print(f"Current Ante: {self.game_manager.game.current_ante}, Blind: {self.game_manager.game.current_blind}")
        print(f"Money: ${self.game_manager.game.inventory.money}")
        
        return self._get_play_state()
    
    def update_shop(self):
        """Update the shop for the current ante"""
        current_ante = self.game_manager.game.current_ante
        ante_number = ((current_ante - 1) // 3) + 1
        
        blind_type_map = {
            0: "boss_blind",
            1: "small_blind", 
            2: "medium_blind"
        }
        
        blind_type = blind_type_map[current_ante % 3]
        
        if ante_number in self.all_shops and blind_type in self.all_shops[ante_number]:
            self.current_shop = self.all_shops[ante_number][blind_type]
            print(f"Updated shop for Ante {current_ante} ({blind_type})")
        else:
            self.current_shop = Shop()
            print(f"Created default shop for Ante {current_ante}")

    def step(self, action):
        """
        Unified step function that handles both play and shop phases
        """
        self.episode_step += 1
        
        # Check for episode termination due to step limit
        if self.episode_step >= self.max_steps_per_episode:
            print(f"Episode terminated due to step limit ({self.max_steps_per_episode})")
            return self._get_current_state(), -5.0, True, {"message": "Step limit reached"}
        
        # Handle shop phase
        if self.in_shop_phase:
            return self._handle_shop_action(action)
        # Handle play phase
        else:
            return self._handle_play_action(action)
    
    def _handle_play_action(self, action):
        """Handle card playing/discarding actions"""
        is_discard = self.is_discard_action(action)
        
        # Check if we've already used all our discards
        if is_discard and self.game_manager.discards_used >= self.game_manager.max_discards_per_round:
            print(f"WARNING: Attempted to discard but already used max discards: {self.game_manager.discards_used}/{self.game_manager.max_discards_per_round}")
            is_discard = False
            action = action % 256
        
        card_indices = self._convert_action_to_card_indices(action)
        
        self.last_action_was_discard = is_discard
        
        if self.episode_step % 10 == 0:
            action_type = "Discard" if is_discard else "Play"
            print(f"Step {self.episode_step}: {action_type} action with indices {card_indices}")
            print(f"  Hand size: {len(self.game_manager.current_hand)}, Score: {self.game_manager.current_score}/{self.game_manager.game.current_blind}")
        
        success = False
        message = "Action failed"
        
        if is_discard:
            success, message = self.game_manager.discard_cards(card_indices)
        else:
            success, message = self.game_manager.play_cards(card_indices)
        
        reward = self._calculate_play_reward(success)
        
        done = self.game_manager.game_over
        
        # Check if we've beaten the ante
        if self.game_manager.current_ante_beaten and not done:
            print(f"\n***** ANTE {self.game_manager.game.current_ante} BEATEN! *****")
            print(f"Score: {self.game_manager.current_score}/{self.game_manager.game.current_blind}")
            print(f"Moving to shop phase")
            self.in_shop_phase = True
            # Update the shop for the current ante
            self.update_shop()
        
        # Get the next state
        next_state = self._get_current_state()
        
        info = {
            "message": message,
            "success": success,
            "shop_phase": self.in_shop_phase,
            "current_ante": self.game_manager.game.current_ante,
            "current_blind": self.game_manager.game.current_blind,
            "current_score": self.game_manager.current_score
        }
        
        return next_state, reward, done, info
    
    def _handle_shop_action(self, action):
        """Handle shop actions"""
        reward = 0
        done = self.game_manager.game_over
        info = {"message": "Unknown shop action"}
        
        # Actions 0-3: Buy shop item
        if action < 4:
            slot = action
            if self.current_shop and slot < len(self.current_shop.items) and self.current_shop.items[slot] is not None:
                item_price = self.current_shop.get_item_price(slot)
                
                if self.game_manager.game.inventory.money >= item_price:
                    item_name = getattr(self.current_shop.items[slot], 'get_name', lambda: "Item")()
                    success = self.current_shop.buy_item(slot, self.game_manager.game.inventory)
                    
                    if success:
                        reward += 0.5  # Reward for successful purchase
                        info['message'] = f"Bought {item_name} for ${item_price}"
                        print(f"Bought {item_name} for ${item_price}")
                        
                        # Handle special case for tarot cards
                        if hasattr(self.current_shop.items[slot], 'item_type') and \
                        self.current_shop.items[slot].item_type == ShopItemType.TAROT:
                            self.pending_tarots.append(item_name)
                else:
                    info['message'] = f"Not enough money to buy item (costs ${item_price})"
        
        # Actions 4-8: Sell joker
        elif action < 9:
            joker_idx = action - 4
            if joker_idx < len(self.game_manager.game.inventory.jokers):
                joker_name = self.game_manager.game.inventory.jokers[joker_idx].name
                sell_value = self.current_shop.sell_item("joker", joker_idx, self.game_manager.game.inventory)
                if sell_value > 0:
                    reward += 0.2  # Reward for selling
                    info['message'] = f"Sold {joker_name} for ${sell_value}"
                    print(f"Sold {joker_name} for ${sell_value}")
        
        # Actions 9-14: Use tarot cards with different strategies
        elif action < 15:
            tarot_idx = (action - 9) // 3
            selection_strategy = (action - 9) % 3
            
            tarot_indices = self.game_manager.game.inventory.get_consumable_tarot_indices()
            if tarot_indices and tarot_idx < len(tarot_indices):
                actual_idx = tarot_indices[tarot_idx]
                tarot_name = self.game_manager.game.inventory.consumables[actual_idx].item.name
                
                # Get card selection based on strategy
                selected_indices = []
                if selection_strategy > 0 and self.game_manager.current_hand:
                    if selection_strategy == 1:  # Lowest cards
                        cards_by_value = [(i, card.rank.value) for i, card in enumerate(self.game_manager.current_hand)]
                        cards_by_value.sort(key=lambda x: x[1])
                        selected_indices = [idx for idx, _ in cards_by_value[:3]]
                    else:  # Highest cards
                        cards_by_value = [(i, card.rank.value) for i, card in enumerate(self.game_manager.current_hand)]
                        cards_by_value.sort(key=lambda x: x[1], reverse=True)
                        selected_indices = [idx for idx, _ in cards_by_value[:3]]
                
                success, message = self.game_manager.use_tarot(actual_idx, selected_indices)
                if success:
                    reward += 0.8  # Higher reward for using tarot
                    info['message'] = message
                    print(f"Used {tarot_name}: {message}")
        
        # Action 15: End shop phase / advance to next ante
        elif action == 15:
            print("Ending shop phase and advancing to next ante")
            success = self.game_manager.next_ante()
            
            if success:
                reward += 2.0  # Reward for advancing to next ante
                self.in_shop_phase = False  # Return to play phase
                
                # Handle any pending tarots
                if self.pending_tarots and self.game_manager.current_hand:
                    self._use_pending_tarots()
                
                info['message'] = f"Advanced to Ante {self.game_manager.game.current_ante}, Blind: {self.game_manager.game.current_blind}"
                print(info['message'])
            else:
                info['message'] = "Failed to advance to next ante"
                print(info['message'])
                
        next_state = self._get_current_state()
        return next_state, reward, done, info
    
    def _use_pending_tarots(self):
        """Use tarot cards that were purchased from the shop"""
        if not self.pending_tarots:
            return
        
        print("\n=== Using Tarot Cards From Shop ===")
        
        for tarot_name in self.pending_tarots:
            tarot_indices = self.game_manager.game.inventory.get_consumable_tarot_indices()
            tarot_index = None
            
            for idx in tarot_indices:
                consumable = self.game_manager.game.inventory.consumables[idx]
                if hasattr(consumable.item, 'name') and consumable.item.name.lower() == tarot_name.lower():
                    tarot_index = idx
                    break
            
            if tarot_index is None:
                print(f"Could not find tarot {tarot_name} in inventory")
                continue
                
            tarot = self.game_manager.game.inventory.consumables[tarot_index].item
            cards_required = tarot.selected_cards_required
            
            if cards_required > len(self.game_manager.current_hand):
                print(f"Not enough cards to use {tarot_name}, needs {cards_required}")
                continue
                
            selected_indices = []
            
            if cards_required > 0:
                card_values = [(i, card.rank.value) for i, card in enumerate(self.game_manager.current_hand)]
                card_values.sort(key=lambda x: x[1])
                selected_indices = [idx for idx, _ in card_values[:cards_required]]
            
            success, message = self.game_manager.use_tarot(tarot_index, selected_indices)
            if success:
                print(f"Used {tarot_name}: {message}")
            else:
                print(f"Failed to use {tarot_name}: {message}")
        
        self.pending_tarots = []
    
    def _get_current_state(self):
        """Get the current state based on whether we're in shop phase or play phase"""
        if self.in_shop_phase:
            return self._get_strategy_state()
        else:
            return self._get_play_state()
    
    def _get_play_state(self):
        """Get the current state as a flat numpy array of floats with FIXED SIZE for play actions"""
        state_features = []
        
        # Game state information
        state_features.append(float(self.game_manager.game.current_ante))
        state_features.append(float(self.game_manager.game.current_blind))
        state_features.append(float(self.game_manager.current_score))
        state_features.append(float(self.game_manager.hands_played))
        state_features.append(float(self.game_manager.max_hands_per_round))
        state_features.append(float(self.game_manager.discards_used))
        state_features.append(float(self.game_manager.max_discards_per_round))
        state_features.append(1.0 if self.game_manager.game.is_boss_blind else 0.0)
        
        # Card information (for up to 8 cards in hand)
        for i in range(8):
            if i < len(self.game_manager.current_hand):
                card = self.game_manager.current_hand[i]
                
                # Normalized rank value (1-14) -> (0-1)
                rank_value = float(card.rank.value) / 14.0
                state_features.append(rank_value)
                
                # One-hot encoding of suit
                suit_features = [0.0, 0.0, 0.0, 0.0]
                if card.suit == Suit.HEARTS:
                    suit_features[0] = 1.0
                elif card.suit == Suit.DIAMONDS:
                    suit_features[1] = 1.0
                elif card.suit == Suit.CLUBS:
                    suit_features[2] = 1.0
                elif card.suit == Suit.SPADES:
                    suit_features[3] = 1.0
                state_features.extend(suit_features)
                
                # Normalized enhancement value
                enhancement_value = float(card.enhancement.value) / 12.0
                state_features.append(enhancement_value)
                
                # Card properties
                state_features.append(1.0 if card.face else 0.0)
                state_features.append(1.0 if getattr(card, 'debuffed', False) else 0.0)
            else:
                # Padding for non-existent cards
                state_features.extend([0.0] * 8)
        
        # Ensure we have a consistent state size
        assert len(state_features) == 72, f"Expected 72 features, got {len(state_features)}"
        
        return np.array(state_features, dtype=np.float32)
        
    def _get_strategy_state(self):
        """Get the current strategy state as a flat numpy array"""
        state_features = []
        
        # Game state information
        state_features.append(float(self.game_manager.game.current_ante))
        state_features.append(float(self.game_manager.game.current_blind))
        state_features.append(float(self.game_manager.game.inventory.money))
        state_features.append(float(len(self.game_manager.game.inventory.jokers)))
        state_features.append(float(len(self.game_manager.game.inventory.consumables)))
        state_features.append(1.0 if self.game_manager.game.is_boss_blind else 0.0)
        
        # Boss blind effect (one-hot encoded)
        boss_effect = [0.0] * len(BossBlindEffect)
        if self.game_manager.game.is_boss_blind and self.game_manager.game.active_boss_blind_effect:
            boss_effect[self.game_manager.game.active_boss_blind_effect.value - 1] = 1.0
        state_features.extend(boss_effect)
        
        # Joker information (up to 5 jokers)
        joker_features = [0.0] * 10
        for i, joker in enumerate(self.game_manager.game.inventory.jokers[:5]):
            if joker:
                joker_features[i*2] = joker.sell_value / 5.0  # Normalized sell value
                joker_features[i*2+1] = 1.0 if joker.rarity == "uncommon" else 0.5  # Rarity
        state_features.extend(joker_features)
        
        # Shop information (up to 4 items)
        if self.current_shop:
            for i in range(4):
                if i < len(self.current_shop.items) and self.current_shop.items[i] is not None:
                    state_features.append(1.0)  # Item exists
                    state_features.append(float(self.current_shop.get_item_price(i)) / 10.0)  # Normalized price
                    
                    # Item type (one-hot encoded)
                    item_type_features = [0.0] * len(ShopItemType)
                    item_type_features[self.current_shop.items[i].item_type.value - 1] = 1.0
                    state_features.extend(item_type_features)
                else:
                    # Empty slot
                    state_features.append(0.0)
                    state_features.append(0.0)
                    state_features.extend([0.0] * len(ShopItemType))
        else:
            # No shop available
            for _ in range(4 * (2 + len(ShopItemType))):
                state_features.append(0.0)
        
        return np.array(state_features, dtype=np.float32)
    
    def _define_play_action_space(self):
        """Define the action space size for playing cards"""
        # 0-255: Play card combinations
        # 256-511: Discard card combinations
        return 512 

    def _define_strategy_action_space(self):
        """Define the action space size for strategy (shop) actions"""
        # 0-3: Buy shop item
        # 4-8: Sell joker
        # 9-14: Use tarot cards with different strategies
        # 15: Skip/Next ante
        return 16

    def _calculate_play_reward(self, success=True):
        """Calculate a reward for the play agent"""
        if not success:
            return -0.5  # Penalty for invalid actions
        
        # Base reward based on progress toward beating the blind
        score_progress = self.game_manager.current_score / self.game_manager.game.current_blind
        
        # Reward based on hand quality
        hand_quality_reward = 0
        if self.game_manager.hand_result:
            hand_quality_map = {
                HandType.HIGH_CARD: -1.0,
                HandType.PAIR: -0.1,
                HandType.TWO_PAIR: 1.5,
                HandType.THREE_OF_A_KIND: 3.0,
                HandType.STRAIGHT: 7.0,
                HandType.FLUSH: 7.0,
                HandType.FULL_HOUSE: 10.0,
                HandType.FOUR_OF_A_KIND: 15.0,
                HandType.STRAIGHT_FLUSH: 20.0
            }
            hand_quality_reward = hand_quality_map.get(self.game_manager.hand_result, 0.1)
        
        # Reward for playing more cards (better hands require 5+ cards)
        cards_played = len(self.game_manager.played_cards)
        cards_bonus = 0
        if cards_played >= 5:
            cards_bonus = 2.0
        else:
            cards_bonus = 0.1 * cards_played
        
        # Penalty for frequent discarding
        discard_penalty = -0.5 if self.last_action_was_discard else 0
        
        # Reward for beating an ante
        ante_beaten_reward = 10.0 if self.game_manager.current_ante_beaten else 0
        
        # Penalty for game over
        game_over_penalty = -10.0 if self.game_manager.game_over else 0
        
        # Total reward
        total_reward = score_progress + hand_quality_reward + cards_bonus + discard_penalty + ante_beaten_reward + game_over_penalty
        
        # Log rewards occasionally for debugging
        if self.episode_step % 10 == 0:
            print(f"Reward breakdown: score_progress={score_progress:.2f}, hand_quality={hand_quality_reward:.2f}, " +
                f"cards_bonus={cards_bonus:.2f}, discard_penalty={discard_penalty:.2f}, " +
                f"ante_beaten={ante_beaten_reward:.2f}, game_over={game_over_penalty:.2f}")
            print(f"Hand: {self.game_manager.hand_result}, Cards played: {cards_played}, Total reward: {total_reward:.2f}")
        
        return total_reward
    
    def get_valid_play_actions(self):
        """Return valid play actions with guidance toward good poker hands"""
        valid_actions = []
        
        # If we can still play hands this round
        if self.game_manager.hands_played < self.game_manager.max_hands_per_round:
            # First, check if we have any good poker hands available
            best_hand_info = self.game_manager.get_best_hand_from_current()
            
            if best_hand_info:
                best_hand, best_cards = best_hand_info
                
                # Convert the best hand to an action
                if best_hand.value >= HandType.PAIR.value:  # It's a decent hand
                    recommended_indices = []
                    for card in best_cards:
                        for i, hand_card in enumerate(self.game_manager.current_hand):
                            if hand_card.rank == card.rank and hand_card.suit == card.suit:
                                recommended_indices.append(i)
                                break
                    
                    # Convert indices to action number
                    if recommended_indices:
                        action = self._indices_to_action(recommended_indices, is_discard=False)
                        valid_actions.append(action)
                        
                        # Add this as the first action, with higher probability of being chosen
                        for _ in range(3):  # Add multiple times to increase probability
                            valid_actions.append(action)
            
            # Add all possible play actions
            for i in range(min(256, 2**len(self.game_manager.current_hand))):
                if self._is_valid_play_action(i):
                    valid_actions.append(i)  # Play action
        
        # If we can still discard cards
        if self.game_manager.discards_used < self.game_manager.max_discards_per_round:
            # Only add discard actions if we have a poor hand
            best_hand_info = self.game_manager.get_best_hand_from_current()
            if not best_hand_info or best_hand_info[0].value <= HandType.PAIR.value:
                # Actions to discard cards
                for i in range(min(256, 2**len(self.game_manager.current_hand))):
                    if self._is_valid_discard_action(i):
                        valid_actions.append(i + 256)  # Discard action
        
        return valid_actions

    def get_valid_strategy_actions(self):
        """Return valid strategy actions based on current game state"""
        valid_actions = []
        
        # Check which shop items we can afford
        if self.current_shop:
            for i in range(4):  # 4 shop slots
                if i < len(self.current_shop.items) and self.current_shop.items[i] is not None and \
                self.game_manager.game.inventory.money >= self.current_shop.get_item_price(i):
                    valid_actions.append(i)  # Buy shop item
        
        # Check if we can sell jokers (limit to 5 jokers max)
        for i in range(min(5, len(self.game_manager.game.inventory.jokers))):
            valid_actions.append(i + 4)  # Sell joker
        
        # Check if we can use tarot cards
        tarot_indices = self.game_manager.game.inventory.get_consumable_tarot_indices()
        for i, tarot_idx in enumerate(tarot_indices):
            if i < 2:  # Limit to first 2 tarots for simplicity
                valid_actions.append(9 + i*3)  # Use tarot with no cards
                valid_actions.append(10 + i*3)  # Use tarot with lowest cards
                valid_actions.append(11 + i*3)  # Use tarot with highest cards
        
        # Always valid to end shop phase
        valid_actions.append(15)
        
        # If no valid actions, can always end shop
        if not valid_actions:
            valid_actions.append(15)
        
        return valid_actions
    
    def _is_valid_play_action(self, action):
        """Check if a play action is valid (has at least one card selected)"""
        # Prevent empty plays
        card_indices = self._convert_action_to_card_indices(action)
        return len(card_indices) > 0

    def _is_valid_discard_action(self, action):
        """Check if a discard action is valid (has at least one card selected)"""
        # Prevent empty discards
        card_indices = self._convert_action_to_card_indices(action)
        return len(card_indices) > 0

    def _convert_action_to_card_indices(self, action):
        """
        Convert an action integer into a list of card indices to play or discard
        
        Args:
            action: Integer representing the action
            
        Returns:
            List of card indices to play/discard
        """
        # Get the card selection part of the action (for up to 8 cards)
        card_mask = action % 256
        
        # Convert to binary representation to determine which cards to select
        binary = format(card_mask, '08b')  # 8-bit binary representation
        
        # Select cards where the corresponding bit is 1
        card_indices = [i for i, bit in enumerate(reversed(binary)) if bit == '1']
        
        return card_indices

    def _indices_to_action(self, indices, is_discard=False):
        """Convert a list of card indices to an action number"""
        action = 0
        for idx in indices:
            action |= (1 << idx)
        
        if is_discard:
            action += 256
            
        return action

    def is_discard_action(self, action):
        """
        Check if an action is a discard (vs play) action
        
        Args:
            action: Integer representing the action
            
        Returns:
            Boolean indicating if this is a discard action
        """
        return action >= 256


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
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Track rewards for analytics
        self.recent_rewards.append(reward)
    
    def replay(self, batch_size):
        """Train the network using experiences from memory"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Pre-process states and next_states to ensure consistent shapes
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for experience in minibatch:
            state, action, reward, next_state, done = experience
            
            # Convert to numpy arrays of consistent shape if needed
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)
            
            # Ensure all states have the same shape
            if len(state.shape) > 1:
                state = state.flatten()
                
            if len(next_state.shape) > 1:
                next_state = next_state.flatten()
                
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert to numpy arrays with consistent shapes
        # Find the maximum length to pad all states to the same size
        max_state_length = max(len(state) for state in states)
        max_next_state_length = max(len(next_state) for next_state in next_states)
        
        # Pad states if needed
        for i in range(len(states)):
            if len(states[i]) < max_state_length:
                states[i] = np.pad(states[i], (0, max_state_length - len(states[i])), 'constant')
            if len(next_states[i]) < max_next_state_length:
                next_states[i] = np.pad(next_states[i], (0, max_next_state_length - len(next_states[i])), 'constant')
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
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
        
    def act(self, state, valid_actions=None):
        """Choose an action using epsilon-greedy policy"""
        # Ensure state is properly formatted
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
            
        if len(state.shape) == 1:
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
            mask = np.full(self.action_size, -1e6)  # Large negative value
            for action in valid_actions:
                mask[action] = 0
            act_values = act_values + mask
        
        return np.argmax(act_values[0])
    
class StrategyAgent:
    """Agent responsible for making decisions during the shop/strategy phase"""
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
            
            15: "Skip (End shop/Advance to next ante)"
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
        """Store experience in replay memory"""
        # Ensure consistent state format
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        
        # Flatten multi-dimensional states for consistency
        if len(state.shape) > 1:
            state = state.flatten()
        if len(next_state.shape) > 1:
            next_state = next_state.flatten()
            
        self.memory.append((state, action, reward, next_state, done))
        
        # Track rewards for analytics
        self.recent_rewards.append(reward)
        self.recent_actions.append(action)
    
    def act(self, state, valid_actions=None):
        """Choose an action based on the current state"""
        state_array = np.array(state).reshape(1, -1)
        
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state_array, verbose=0)
        
        if valid_actions is not None and len(valid_actions) > 0:
            mask = np.full(self.action_size, -1e6)  # Large negative value
            for action in valid_actions:
                mask[action] = 0
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
        """Save model to file"""
        self.model.save(file_path)
    
    def load_model(self, file_path):
        """Load model from file"""
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


def train_agents(episodes=10000, batch_size=64, game_config=None, save_interval=500, play_agent=None, strategy_agent=None, log_interval=100):
    """
    Train both play and strategy agents
    
    Args:
        episodes: Number of episodes to train
        batch_size: Batch size for replay
        game_config: Configuration for the game environment
        save_interval: How often to save model checkpoints
        play_agent: Existing play agent (if None, create new)
        strategy_agent: Existing strategy agent (if None, create new)
        log_interval: How often to log training progress
        
    Returns:
        Tuple of (play_agent, strategy_agent)
    """
    if game_config is None:
        game_config = {}
    
    # Print what configuration is being used
    print(f"Training with config: {game_config}")
    
    env = BalatroEnv(config=game_config)
    
    # Initialize play agent
    initial_play_state = env._get_play_state()
    play_state_size = len(initial_play_state)
    if play_agent is None:
        print(f"Creating new PlayingAgent with state_size={play_state_size}")
        play_agent = PlayingAgent(state_size=play_state_size, 
                                action_size=env._define_play_action_space())
    
    # Initialize strategy agent
    strategy_state = env._get_strategy_state()
    strategy_state_size = len(strategy_state)
    if strategy_agent is None:  
        print(f"Creating new StrategyAgent with state_size={strategy_state_size}")
        strategy_agent = StrategyAgent(state_size=strategy_state_size, 
                                     action_size=env._define_strategy_action_space())
    
    # Training stats
    play_rewards = []
    strategy_rewards = []
    ante_progression = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        # Track game progress
        max_ante_reached = 1
        game_steps = 0
        
        done = False
        while not done and game_steps < env.max_steps_per_episode:
            game_steps += 1
            
            # Get an agent-appropriate action based on the current phase
            valid_actions = []
            if env.in_shop_phase:
                # Strategy phase (shop)
                valid_actions = env.get_valid_strategy_actions()
                action = strategy_agent.act(state, valid_actions=valid_actions)
                if game_steps % 20 == 0:  # Occasional logging
                    action_name = strategy_agent.action_map.get(action, f"Strategy action {action}")
                    print(f"Step {game_steps}: {action_name}")
            else:
                # Play phase (cards)
                valid_actions = env.get_valid_play_actions()
                action = play_agent.act(state, valid_actions=valid_actions)
                if game_steps % 20 == 0:  # Occasional logging
                    action_type, card_indices, action_desc = play_agent.decode_action(action) if hasattr(play_agent, 'decode_action') else (None, [], f"Action {action}")
                    print(f"Step {game_steps}: {action_desc}")
            
            # Take a step in the environment
            next_state, reward, done, info = env.step(action)
            
            # Track maximum ante reached
            max_ante_reached = max(max_ante_reached, env.game_manager.game.current_ante)
            
            # Store experience in appropriate agent's memory
            if env.in_shop_phase:
                strategy_agent.remember(state, action, reward, next_state, done)
                strategy_rewards.append(reward)
            else:
                play_agent.remember(state, action, reward, next_state, done)
                play_rewards.append(reward)
            
            total_reward += reward
            state = next_state
        
        # Train models if enough experiences have been collected
        if len(play_agent.memory) > batch_size:
            play_loss = play_agent.replay(batch_size)
        
        if len(strategy_agent.memory) > batch_size:
            strategy_loss = strategy_agent.replay(batch_size)
        
        play_agent.decay_epsilon()
        strategy_agent.decay_epsilon()
        
        ante_progression.append(max_ante_reached)
        
        if (e + 1) % log_interval == 0:
            # Calculate average play and strategy rewards
            recent_play_rewards = play_rewards[-min(len(play_rewards), log_interval*100):]
            recent_strategy_rewards = strategy_rewards[-min(len(strategy_rewards), log_interval*100):]
            
            avg_play_reward = sum(recent_play_rewards) / max(1, len(recent_play_rewards))
            avg_strategy_reward = sum(recent_strategy_rewards) / max(1, len(recent_strategy_rewards))
            avg_ante = sum(ante_progression[-log_interval:]) / max(1, len(ante_progression[-log_interval:]))
            
            print(f"\n===== Episode {e+1}/{episodes} =====")
            print(f"  Play Agent: reward={avg_play_reward:.2f}, epsilon={play_agent.epsilon:.3f}")
            print(f"  Strategy Agent: reward={avg_strategy_reward:.2f}, epsilon={strategy_agent.epsilon:.3f}")
            print(f"  Average max ante: {avg_ante:.2f}")
            print(f"  Total steps this episode: {game_steps}")
            
            play_stats = play_agent.get_stats()
            strategy_stats = strategy_agent.get_stats()
            
            print(f"  Play agent stats: {play_stats}")
            print(f"  Strategy agent stats: {strategy_stats}")
            
            # Print game state details
            print(f"  Current ante: {env.game_manager.game.current_ante}")
            print(f"  Current blind: {env.game_manager.game.current_blind}")
            print(f"  Current money: ${env.game_manager.game.inventory.money}")
            print(f"  Jokers: {len(env.game_manager.game.inventory.jokers)}")
            print(f"  Is boss blind: {env.game_manager.game.is_boss_blind}")
        
        if (e + 1) % save_interval == 0:
            print(f"Saving models at episode {e+1}")
            play_agent.save_model(f"play_agent_ep{e+1}.h5")
            strategy_agent.save_model(f"strategy_agent_ep{e+1}.h5")
            
            play_agent.save_model("play_agent_latest.h5")
            strategy_agent.save_model("strategy_agent_latest.h5")
    
    play_agent.save_model("play_agent_final.h5")
    strategy_agent.save_model("strategy_agent_final.h5")
    
    return play_agent, strategy_agent


def evaluate_agent(play_agent, strategy_agent=None, num_episodes=10, config=None):
    """
    Evaluate agent performance without exploration
    
    Args:
        play_agent: Playing agent to evaluate
        strategy_agent: Strategy agent to evaluate (if None, will use default strategy)
        num_episodes: Number of episodes to evaluate
        config: Game configuration
        
    Returns:
        Dictionary of evaluation results
    """
    env = BalatroEnv(config=config)
    
    # Save original epsilon values
    play_epsilon = play_agent.epsilon
    strategy_epsilon = strategy_agent.epsilon if strategy_agent else 0
    
    # Disable exploration for evaluation
    play_agent.epsilon = 0
    if strategy_agent:
        strategy_agent.epsilon = 0
    
    results = {
        'total_rewards': [],
        'max_antes': [],
        'hands_played': [],
        'win_rate': 0,
        'average_score': 0,
        'max_score': 0
    }
    
    for e in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        max_ante = 1
        hands_played = 0
        game_won = False
        step = 0
        
        done = False
        while not done and step < env.max_steps_per_episode:
            step += 1
            
            # Choose action based on current phase
            if env.in_shop_phase:
                if strategy_agent:
                    valid_actions = env.get_valid_strategy_actions()
                    action = strategy_agent.act(state, valid_actions=valid_actions)
                else:
                    # Default strategy if no agent provided
                    action = 15  # Skip/Advance to next ante
            else:
                valid_actions = env.get_valid_play_actions()
                action = play_agent.act(state, valid_actions=valid_actions)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            
            # Track stats
            hands_played = env.game_manager.hands_played
            max_ante = max(max_ante, env.game_manager.game.current_ante)
        
        # Record results
        results['total_rewards'].append(total_reward)
        results['max_antes'].append(max_ante)
        results['hands_played'].append(hands_played)
        
        # Consider a win if player reached ante 8 or higher
        if max_ante >= 8:
            game_won = True
        results['win_rate'] += 1 if game_won else 0
        
        print(f"Evaluation episode {e+1}: Max ante {max_ante}, Total reward: {total_reward:.2f}")
    
    # Calculate aggregate statistics
    results['win_rate'] = results['win_rate'] / num_episodes * 100
    results['average_score'] = sum(results['max_antes']) / num_episodes
    results['max_score'] = max(results['max_antes'])
    
    # Restore original epsilon values
    play_agent.epsilon = play_epsilon
    if strategy_agent:
        strategy_agent.epsilon = strategy_epsilon
    
    return results


def add_demonstration_examples(play_agent, num_examples=100):
    """
    Add expert demonstration examples to the agent's memory
    
    This helps jumpstart learning by showing examples of good play
    """
    env = BalatroEnv(config={'simplified': True})
    
    for i in range(num_examples):
        state = env.reset()
        done = False
        
        while not done:
            # Get the best hand according to poker rules
            best_hand_info = env.game_manager.get_best_hand_from_current()
            
            if best_hand_info:
                # Convert the best hand to card indices
                best_hand, best_cards = best_hand_info
                indices = []
                
                for card in best_cards:
                    for i, hand_card in enumerate(env.game_manager.current_hand):
                        if hand_card.rank == card.rank and hand_card.suit == card.suit:
                            indices.append(i)
                            break
                
                # Convert indices to action number
                action = 0
                for idx in indices:
                    action |= (1 << idx)
                
                # Execute the action and get next state, reward
                next_state, reward, done, _ = env.step(action)
                
                # Add this as a demonstration example
                play_agent.remember(state, action, reward, next_state, done)
                
                state = next_state
            else:
                # If no good hand, try discarding
                if env.game_manager.discards_used < env.game_manager.max_discards_per_round:
                    # Discard low cards
                    indices = []
                    low_cards = sorted([(i, card.rank.value) for i, card in enumerate(env.game_manager.current_hand)], 
                                     key=lambda x: x[1])[:3]
                    indices = [idx for idx, _ in low_cards]
                    
                    action = 0
                    for idx in indices:
                        action |= (1 << idx)
                    
                    # Make it a discard action
                    action += 256
                    
                    next_state, reward, done, _ = env.step(action)
                    play_agent.remember(state, action, reward, next_state, done)
                    
                    state = next_state
                else:
                    # If can't discard, just play all cards
                    action = (1 << len(env.game_manager.current_hand)) - 1
                    next_state, reward, done, _ = env.step(action)
                    play_agent.remember(state, action, reward, next_state, done)
                    
                    state = next_state
                    
            # Handle shop phase if it comes up
            if env.in_shop_phase and not done:
                # Just advance to next ante
                action = 15  # Skip/advance to next ante
                next_state, reward, done, _ = env.step(action)
                state = next_state
    
    print(f"Added {len(play_agent.memory)} demonstration examples to memory")


def train_with_curriculum():
    """
    Enhanced curriculum learning approach for training agents
    
    The curriculum gradually increases in difficulty:
    1. Basic card play with simplified rules
    2. Adding shop interactions and boss blinds
    3. Full game with enhanced exploration
    
    Returns:
        Tuple of (play_agent, strategy_agent)
    """
    # Create the base environment
    env = BalatroEnv(config={'simplified': True})
    
    # Initialize agents with correct state sizes
    play_state_size = len(env._get_play_state())
    play_agent = PlayingAgent(state_size=play_state_size, 
                             action_size=env._define_play_action_space())
    
    strategy_state_size = len(env._get_strategy_state())
    strategy_agent = StrategyAgent(state_size=strategy_state_size,
                                  action_size=env._define_strategy_action_space())
    
    # Add demonstration examples to jumpstart learning
    print("Adding demonstration examples...")
    add_demonstration_examples(play_agent, num_examples=200)
    
    # Phase 1: Train with simplified game (no boss blinds, fixed deck)
    print("\n===== PHASE 1: Learning basic card play =====")
    play_agent, strategy_agent = train_agents(episodes=500, 
                                             game_config={'simplified': True}, 
                                             play_agent=play_agent,
                                             strategy_agent=strategy_agent,
                                             log_interval=20)
    
    # Evaluate after Phase 1
    results = evaluate_agent(play_agent, strategy_agent, num_episodes=20, config={'simplified': True})
    print(f"\nAfter Phase 1: average score: {results['average_score']}, max ante: {results['max_score']}")
    
    if results['max_score'] < 2:
        print("Phase 1 performance too low. Retraining with more demonstrations...")
        add_demonstration_examples(play_agent, num_examples=300)
        play_agent, strategy_agent = train_agents(episodes=500, 
                                                game_config={'simplified': True}, 
                                                play_agent=play_agent,
                                                strategy_agent=strategy_agent,
                                                log_interval=20)
    
    # Phase 2: Include boss blinds and shop interaction
    print("\n===== PHASE 2: Adding boss blinds and shop strategy =====")
    play_agent, strategy_agent = train_agents(episodes=1000, 
                                             game_config={'simplified': False},
                                             play_agent=play_agent,
                                             strategy_agent=strategy_agent,
                                             log_interval=20)
    
    # Phase 3: Full game with enhanced exploration
    print("\n===== PHASE 3: Full game training =====")
    play_agent, strategy_agent = train_agents(episodes=2000,
                                             game_config={'full_features': True},
                                             play_agent=play_agent,
                                             strategy_agent=strategy_agent,
                                             log_interval=20)
    
    # Final evaluation
    results = evaluate_agent(play_agent, strategy_agent, num_episodes=50, config={'full_features': True})
    
    print("\n===== FINAL EVALUATION =====")
    print(f"Win rate: {results['win_rate']:.1f}%")
    print(f"Average max ante: {results['average_score']:.2f}")
    print(f"Maximum ante reached: {results['max_score']}")
    
    return play_agent, strategy_agent


if __name__ == "__main__":
    print("Starting Balatro RL agent training...")
    
    # Train agents using curriculum learning
    # This gradually increases complexity for better learning
    play_agent, strategy_agent = train_with_curriculum()
    
    # Save final models
    play_agent.save_model("play_agent_final.h5")
    strategy_agent.save_model("strategy_agent_final.h5")
    
    # Comprehensive evaluation
    results = evaluate_agent(play_agent, strategy_agent, num_episodes=50)
    print("\n===== FINAL EVALUATION =====")
    print(f"Win rate: {results['win_rate']:.2f}%")
    print(f"Average ante: {results['average_score']:.2f}")
    print(f"Maximum ante: {results['max_score']}")
    print(f"Average total reward: {sum(results['total_rewards'])/len(results['total_rewards']):.2f}")
    
    print("\nTraining complete!")