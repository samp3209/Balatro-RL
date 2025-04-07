from collections import deque
import itertools
from GameManager import GameManager
import numpy as np
from Inventory import Inventory
from Card import Card
from Enums import *
from Shop import Shop, ShopItem, ShopItemType, initialize_shops_for_game, FixedShop
from HandEvaluator import *


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random


class BalatroEnv:
    def __init__(self, config=None):
        self.game_manager = GameManager()
        self.config = {
            'simplified': False,  # Simplified rules for initial learning
            'full_features': False  # All game features enabled
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        self.game_manager = GameManager()
        self.last_action_was_discard = False
        
        self.all_shops = initialize_shops_for_game()
        self.current_shop = None
        self.pending_tarots = []
        
        self.episode_step = 0
        self.episode_max_blind = 0
        
        # Initialize the game and shop
        self.start_new_game()

    def start_new_game(self):
        """Initialize game and shop consistently, similar to GameTest.py"""
        self.game_manager.start_new_game()
        
        if self.config['simplified']:
            self.game_manager.game.is_boss_blind = False
            self.game_manager.max_hands_per_round = 4
            
        # Initialize shop for the starting ante
        self.update_shop()
        
        print(f"\n===== STARTING NEW GAME =====")
        print(f"Current Ante: {self.game_manager.game.current_ante}, Blind: {self.game_manager.game.current_blind}")
        print(f"Money: ${self.game_manager.game.inventory.money}")

    def reset(self):
        """Reset the environment to initial state with a completely fresh game"""
        self.last_action_was_discard = False
        self.episode_step = 0
        self.episode_max_blind = 0
        self.pending_tarots = []
        
        self.game_manager = GameManager()
        
        self.game_manager.start_new_game()
        
        if self.config.get('add_bootstrap', False):
            from JokerCreation import create_joker
            joker = create_joker("Bootstraps")
            if joker:
                self.game_manager.game.inventory.add_joker(joker)
                print(f"Added Bootstraps to inventory")
        
        self.game_manager.game.inventory.money = 100
        
        self.update_shop()
        
        return self._get_play_state()
    
    def update_shop(self):
        """Update the shop for the current ante, similar to get_shop_for_current_ante in GameTest.py"""
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
            print(f"Updated shop for Round {current_ante} ({blind_type})")
        else:
            self.current_shop = Shop()
            print(f"Created default shop for Round {current_ante}")

    def step_strategy(self, action):
        """Process a strategy action with improved reward signals and anti-loop measures"""
        # Make sure we have the current shop
        if self.current_shop is None:
            self.update_shop()
        
        reward = 0
        done = self.game_manager.game_over
        info = {"message": "Unknown action"}
        
        # Shopping actions (slots 0-3)
        if action < 4:
            slot = action
            if self.current_shop and slot < len(self.current_shop.items) and self.current_shop.items[slot] is not None:
                item = self.current_shop.items[slot]
                price = self.current_shop.get_item_price(slot)
                
                if self.game_manager.game.inventory.money >= price:
                    item_name = "Unknown Item"
                    
                    # Get proper item name and type for better rewards
                    if hasattr(item, 'item_type'):
                        if item.item_type == ShopItemType.JOKER:
                            item_name = item.item.name if hasattr(item.item, 'name') else "Joker"
                            # Much higher reward for jokers (they're critical for success)
                            reward = 5.0
                            
                        elif item.item_type == ShopItemType.PLANET:
                            item_name = item.item.name if hasattr(item.item, 'name') else "Planet"
                            # Good reward for planets
                            reward = 3.0
                            
                        elif item.item_type == ShopItemType.TAROT:
                            item_name = item.item.name if hasattr(item.item, 'name') else "Tarot"
                            # Decent reward for tarots
                            reward = 2.0
                            
                        elif item.item_type == ShopItemType.BOOSTER:
                            item_name = str(item.item) if hasattr(item, 'item') else "Booster"
                            # Small reward for boosters
                            reward = 1.0
                    
                    success = self.current_shop.buy_item(slot, self.game_manager.game.inventory)
                    if success:
                        info['message'] = f"Bought {item_name} for ${price}"
                    else:
                        reward = 0  # No reward if purchase fails
                else:
                    # Negative reward for trying to buy something we can't afford
                    reward = -0.5
                    info['message'] = f"Not enough money to buy item (costs ${price})"
        elif action < 9:  # Sell joker
            joker_idx = action - 4
            if joker_idx < len(self.game_manager.game.inventory.jokers):
                joker_name = self.game_manager.game.inventory.jokers[joker_idx].name
                sell_value = self.current_shop.sell_item("joker", joker_idx, self.game_manager.game.inventory)
                if sell_value > 0:
                    reward += 0.2  # Reward for selling
                    info['message'] = f"Sold {joker_name} for ${sell_value}"
                    print(f"Sold {joker_name} for ${sell_value}")
        
        elif action < 15:  # Use tarot cards with different strategies
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
        
        # IMPORTANT: Advance to next ante (action 15) with proper reward
        elif action == 15 and self.game_manager.current_ante_beaten:
            # Add an incentive to move to the next ante
            joker_count = len(self.game_manager.game.inventory.jokers)
            reward = 10.0 + joker_count * 2.0  # Strong reward based on joker count
            
            # Additional reward for having saved money
            money = self.game_manager.game.inventory.money
            reward += min(money / 20.0, 5.0)  # Up to 5.0 additional reward
            
            # Execute the next ante action
            success = self.game_manager.next_ante()
            
            if success:
                info['message'] = f"Advanced to Ante {self.game_manager.game.current_ante}"
                print(f"Successfully advanced to Ante {self.game_manager.game.current_ante}")
                
                # Deal a new hand for the new ante
                if not self.game_manager.current_hand:
                    self.game_manager.deal_new_hand()
            else:
                reward = 0  # No reward if failed to advance
                info['message'] = "Failed to advance to next ante"
                print("Failed to advance to next ante")
        
        # Return updated game state
        next_state = self._get_strategy_state()
        return next_state, reward, done, info

    def step_play(self, action):
        """Process a playing action with stricter enforcement of rules"""
        self.episode_step += 1
        
        # Check if we've reached a failed state (max hands played, not enough score, no reset)
        if (self.game_manager.hands_played >= self.game_manager.max_hands_per_round and 
            self.game_manager.current_score < self.game_manager.game.current_blind and 
            not self.game_manager.current_ante_beaten):
            
            # Set game over to true - we've failed this ante
            print(f"\n***** GAME OVER: Failed to beat the ante *****")
            print(f"Score: {self.game_manager.current_score}/{self.game_manager.game.current_blind}")
            print(f"Max hands played: {self.game_manager.hands_played}/{self.game_manager.max_hands_per_round}")
            
            # In GameTest.py, this is handled by checking for Mr. Bones joker first
            mr_bones_index = None
            for i, joker in enumerate(self.game_manager.game.inventory.jokers):
                if joker.name == "Mr. Bones":
                    mr_bones_index = i
                    break
                    
            if mr_bones_index is not None and self.game_manager.current_score >= (self.game_manager.game.current_blind * 0.25):
                # Mr. Bones saves the game but gets removed
                self.game_manager.current_ante_beaten = True
                removed_joker = self.game_manager.game.inventory.remove_joker(mr_bones_index)
                print(f"Mr. Bones saved you and vanished!")
                
                # Now go to shop phase
                next_state = self._get_play_state()
                return next_state, 0.5, False, {"message": "Mr. Bones saved you!", "shop_phase": True}
            else:
                # Game over
                self.game_manager.game_over = True
                next_state = self._get_play_state()
                return next_state, -5.0, True, {"message": "GAME OVER: Failed to beat the ante"}
        
        is_discard = self.is_discard_action(action)
        
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
        
        if is_discard:
            success, message = self.game_manager.discard_cards(card_indices)
        else:
            success, message = self.game_manager.play_cards(card_indices)
        
        reward = self._calculate_play_reward()
        
        done = self.game_manager.game_over
        shop_phase = self.game_manager.current_ante_beaten and not done
        
        if shop_phase:
            print(f"\n***** Round {self.game_manager.game.current_ante} BEATEN! *****")
            print(f"Score: {self.game_manager.current_score}/{self.game_manager.game.current_blind}")
            print(f"Moving to shop phase")
        
        next_state = self._get_play_state()
        
        info = {
            "message": message,
            "shop_phase": shop_phase, 
            "current_ante": self.game_manager.game.current_ante,
            "current_blind": self.game_manager.game.current_blind,
            "current_score": self.game_manager.current_score
        }
        
        return next_state, reward, done, info
    

    def _get_play_state(self):
        """Get the current state as a flat numpy array of floats with FIXED SIZE"""
        state_features = []
        
        state_features.append(float(self.game_manager.game.current_ante))
        state_features.append(float(self.game_manager.game.current_blind))
        state_features.append(float(self.game_manager.current_score))
        state_features.append(float(self.game_manager.hands_played))
        state_features.append(float(self.game_manager.max_hands_per_round))
        state_features.append(float(self.game_manager.discards_used))
        state_features.append(float(self.game_manager.max_discards_per_round))
        state_features.append(1.0 if self.game_manager.game.is_boss_blind else 0.0)
        
        has_pair = 0.0
        has_two_pair = 0.0
        has_three_kind = 0.0
        has_straight_potential = 0.0
        has_flush_potential = 0.0
        
        if best_hand_info := self.game_manager.get_best_hand_from_current():
            hand_type, _ = best_hand_info
            if hand_type.value >= HandType.PAIR.value:
                has_pair = 1.0
            if hand_type.value >= HandType.TWO_PAIR.value:
                has_two_pair = 1.0
            if hand_type.value >= HandType.THREE_OF_A_KIND.value:
                has_three_kind = 1.0
            if hand_type.value >= HandType.STRAIGHT.value:
                has_straight_potential = 1.0
            if hand_type.value >= HandType.FLUSH.value:
                has_flush_potential = 1.0
        
        state_features.extend([has_pair, has_two_pair, has_three_kind, 
                            has_straight_potential, has_flush_potential])



        for i in range(8):
            if i < len(self.game_manager.current_hand):
                card = self.game_manager.current_hand[i]
                
                rank_value = float(card.rank.value) / 14.0
                state_features.append(rank_value)
                
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
                
                enhancement_value = float(card.enhancement.value) / 12.0  # Normalize
                state_features.append(enhancement_value)
                
                state_features.append(1.0 if card.face else 0.0)
                state_features.append(1.0 if getattr(card, 'debuffed', False) else 0.0)
            else:
                state_features.extend([0.0] * 8)
        
        assert len(state_features) == 77, f"Expected 77 features, got {len(state_features)}"
        
        return np.array(state_features, dtype=np.float32)
        
    def _get_strategy_state(self):
        """Get the current strategy state as a flat numpy array"""
        state_features = []
        
        state_features.append(float(self.game_manager.game.current_ante))
        state_features.append(float(self.game_manager.game.current_blind))
        state_features.append(float(self.game_manager.game.inventory.money))
        state_features.append(float(len(self.game_manager.game.inventory.jokers)))
        state_features.append(float(len(self.game_manager.game.inventory.consumables)))
        state_features.append(1.0 if self.game_manager.game.is_boss_blind else 0.0)
        
        boss_effect = [0.0] * len(BossBlindEffect)
        if self.game_manager.game.is_boss_blind and self.game_manager.game.active_boss_blind_effect:
            boss_effect[self.game_manager.game.active_boss_blind_effect.value - 1] = 1.0
        state_features.extend(boss_effect)
        
        joker_features = [0.0] * 10
        for i, joker in enumerate(self.game_manager.game.inventory.jokers[:5]):
            if joker:
                joker_features[i*2] = joker.sell_value / 5.0
                joker_features[i*2+1] = 1.0 if joker.rarity == "uncommon" else 0.5
        state_features.extend(joker_features)
        
        if hasattr(self, 'shop'):
            for i in range(4):
                if i < len(self.shop.items) and self.shop.items[i] is not None:
                    state_features.append(1.0)
                    state_features.append(float(self.shop.get_item_price(i)) / 10.0) 
                    
                    item_type_features = [0.0] * len(ShopItemType)
                    item_type_features[self.shop.items[i].item_type.value - 1] = 1.0
                    state_features.extend(item_type_features)
                else:
                    state_features.append(0.0)
                    state_features.append(0.0)
                    state_features.extend([0.0] * len(ShopItemType))
        else:
            for _ in range(4 * (2 + len(ShopItemType))):
                state_features.append(0.0)
        print(f"Strategy state size: {len(state_features)}")

        return np.array(state_features, dtype=np.float32)
    
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
        """Improved reward calculation focused on consistent progress"""
        # Base score progress
        score_progress = min(1.0, self.game_manager.current_score / self.game_manager.game.current_blind)
        
        # Scale progress reward and add bonus for beating the blind
        progress_reward = score_progress * 5.0
        if self.game_manager.current_ante_beaten:
            progress_reward += 10.0
        
        # Much stronger hand quality rewards
        hand_quality_reward = 0
        if self.game_manager.hand_result:
            hand_map = {
                HandType.HIGH_CARD: -2.0,
                HandType.PAIR: 1.0,
                HandType.TWO_PAIR: 3.0,
                HandType.THREE_OF_A_KIND: 6.0,
                HandType.STRAIGHT: 10.0,
                HandType.FLUSH: 10.0,
                HandType.FULL_HOUSE: 15.0,
                HandType.FOUR_OF_A_KIND: 25.0,
                HandType.STRAIGHT_FLUSH: 40.0
            }
            hand_quality_reward = hand_map.get(self.game_manager.hand_result, 0.2)
        
        # Bigger incentive for playing real poker hands (5+ cards)
        cards_played = len(self.game_manager.played_cards)
        if cards_played >= 5:
            cards_bonus = 3.0 + (cards_played - 5)  # Bigger bonus for more cards
        else:
            cards_bonus = 0.2 * cards_played
        
        # Strong penalty for game over
        game_over_penalty = -25.0 if self.game_manager.game_over else 0
        
        total_reward = progress_reward + hand_quality_reward + cards_bonus + game_over_penalty
        
        # Special cases
        if self.game_manager.hand_result and self.game_manager.hand_result.value >= HandType.FULL_HOUSE.value:
            total_reward *= 1.5  # Bonus multiplier for excellent hands
        
        return total_reward
    
    def _calculate_strategy_reward(self):
        ante_progress = 0.2 * self.game_manager.game.current_ante
        
        joker_quality = sum(j.sell_value for j in self.game_manager.game.inventory.jokers) * 0.02
        
        money_reward = 0.01 * self.game_manager.game.inventory.money
        
        planet_level_reward = 0.05 * sum(self.game_manager.game.inventory.planet_levels.values())
        
        game_over_penalty = -5.0 if self.game_manager.game_over else 0
        
        return ante_progress + joker_quality + money_reward + planet_level_reward + game_over_penalty
    
    def get_normalized_state(self, is_shop_phase=False):
        """
        Get a state representation with consistent dimensionality regardless of phase
        """
        if is_shop_phase:
            raw_state = self._get_strategy_state()
        else:
            raw_state = self._get_play_state()
            
        if not isinstance(raw_state, np.ndarray):
            raw_state = np.array(raw_state, dtype=np.float32)
        
        fixed_size = 77
        normalized_state = np.zeros(fixed_size, dtype=np.float32)
        
        normalized_state[:len(raw_state)] = raw_state
        
        return normalized_state


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
                cards_required = consumable.item.selected_cards_required / 3.0
            
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

    def get_valid_play_actions(self):
        """Return valid play actions with proper handling for end-of-round cases"""
        valid_actions = []
        
        # Check if we're in a terminal state (max hands played, not enough score)
        hands_limit_reached = self.game_manager.hands_played >= self.game_manager.max_hands_per_round
        ante_beaten = self.game_manager.current_ante_beaten
        
        # If we've reached max hands but haven't beaten the ante, no valid actions
        # (this gets handled by the step_play method to set game_over=True)
        if hands_limit_reached and not ante_beaten:
            print("No valid actions: max hands played and ante not beaten")
            # Return a dummy action (play all cards) so the agent has something to do
            # The step_play method will handle this as a game over state
            all_cards_action = (1 << len(self.game_manager.current_hand)) - 1
            return [all_cards_action]
        
        # If we can still play hands this round
        if not hands_limit_reached:
            # First, check if we have any good poker hands available
            best_hand_info = self.game_manager.get_best_hand_from_current()
            
            if best_hand_info:
                best_hand, best_cards = best_hand_info
                
                # Convert the best hand to an action
                if best_hand.value >= HandType.PAIR.value:  # It's a decent hand
                    recommended_indices = []
                    for card in best_cards:
                        for i, hand_card in enumerate(self.game_manager.current_hand):
                            if hasattr(hand_card, 'rank') and hasattr(card, 'rank') and \
                            hasattr(hand_card, 'suit') and hasattr(card, 'suit') and \
                            hand_card.rank == card.rank and hand_card.suit == card.suit:
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
        
        # If no valid actions found (shouldn't normally happen)
        if not valid_actions and len(self.game_manager.current_hand) > 0:
            print("Warning: No valid actions found with cards in hand. Defaulting to play all.")
            all_cards_action = (1 << len(self.game_manager.current_hand)) - 1
            valid_actions.append(all_cards_action)
        
        return valid_actions

    def _is_valid_play_action(self, action):
        """Check if a play action is valid (has at least one card selected)"""
        # Prevent empty plays
        card_indices = self._convert_action_to_card_indices(action)
        
        if len(card_indices) == 0:
            return False
        
        # Check if we've already reached max hands
        if self.game_manager.hands_played >= self.game_manager.max_hands_per_round:
            return False
        
        return True

    def _is_valid_discard_action(self, action):
        """Check if a discard action is valid (has at least one card selected)"""
        # Prevent empty discards
        card_indices = self._convert_action_to_card_indices(action)
        
        if len(card_indices) == 0:
            return False
        
        # Check if we've already used max discards
        if self.game_manager.discards_used >= self.game_manager.max_discards_per_round:
            return False
        
        return True

    def _indices_to_action(self, indices, is_discard=False):
        """Convert a list of card indices to an action number"""
        action = 0
        for idx in indices:
            action |= (1 << idx)
        
        if is_discard:
            action += 256
            
        return action

    def get_valid_strategy_actions(self):
        """Return valid strategy actions based on current game state"""
        valid_actions = []
        
        # Create shop if it doesn't exist yet
        if not hasattr(self, 'current_shop') or self.current_shop is None:
            self.update_shop()
        
        # Check which shop items we can afford
        for i in range(4):  # 4 shop slots
            if i < len(self.current_shop.items) and self.current_shop.items[i] is not None and \
            self.game_manager.game.inventory.money >= self.current_shop.get_item_price(i):
                valid_actions.append(i)  # Buy shop item
        
        # Check if we can sell jokers
        for i in range(len(self.game_manager.game.inventory.jokers)):
            if i < 5:  # Limit to max 5 jokers
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
        
        # If no valid actions, can always skip
        if not valid_actions:
            valid_actions.append(15)
        
        return valid_actions

    def _convert_action_to_card_indices(self, action):
        """
        Convert an action integer into a list of card indices to play or discard
        
        Args:
            action: Integer representing the action
            
        Returns:
            List of card indices to play/discard
        """
        # First determine if this is a play or discard action (first bit)
        action_type = action // 256  # For 8 cards, we have 2^8=256 possible combinations
        
        # Get the card selection part of the action
        card_mask = action % 256
        
        # Convert to binary representation to determine which cards to select
        binary = format(card_mask, '08b')  # 8-bit binary representation
        
        # Select cards where the corresponding bit is 1
        card_indices = [i for i, bit in enumerate(reversed(binary)) if bit == '1']
        
        return card_indices

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
        """Build a more sophisticated neural network for predicting Q-values"""
        model = Sequential()
        
        # First layer with more units
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        
        # Add an intermediate layer
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        
        # Add another intermediate layer
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        
        # Output layer
        model.add(Dense(self.action_size, activation='linear'))
        
        # Use a lower learning rate
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.0005))
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
        """Choose an action with robust input validation"""
        # Validate and fix input state
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
            
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # Check state dimensions
        if state.shape[1] != self.state_size:
            print(f"WARNING: Play agent received incorrect state size. Expected {self.state_size}, got {state.shape[1]}")
            if state.shape[1] < self.state_size:
                # Pad with zeros
                padded_state = np.zeros((1, self.state_size), dtype=np.float32)
                padded_state[0, :state.shape[1]] = state[0, :]
                state = padded_state
            else:
                # Truncate
                state = state[:, :self.state_size]
        
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        
        if valid_actions is not None and len(valid_actions) > 0:
            mask = np.full(self.action_size, -1e6)
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
        """Improved learning strategy with better experience prioritization"""
        if len(self.memory) < batch_size:
            return
        
        # Prioritize experiences that led to significant rewards
        priorities = []
        for _, _, reward, _, _ in self.memory:
            # Prioritize experiences with large positive or negative rewards
            priority = abs(reward) + 0.1  # Small base priority for all experiences
            priorities.append(priority)
        
        # Normalize priorities
        priorities = np.array(priorities) / sum(priorities)
        
        # Sample according to priorities
        indices = np.random.choice(len(self.memory), batch_size, p=priorities)
        
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
    
    def prioritized_replay(self, batch_size):
        """Prioritized Experience Replay implementation"""
        if len(self.memory) < batch_size:
            return
        
        # Calculate TD errors for prioritization
        td_errors = []
        for state, action, reward, next_state, done in self.memory:
            state_array = np.array(state).reshape(1, -1)
            next_state_array = np.array(next_state).reshape(1, -1)
            
            current_q = self.model.predict(state_array, verbose=0)[0][action]
            
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * np.max(self.target_model.predict(next_state_array, verbose=0)[0])
            
            # TD error is the difference between target and current Q-value
            td_error = abs(target_q - current_q)
            td_errors.append(td_error)
        
        # Convert errors to probabilities with prioritization
        probabilities = np.array(td_errors) ** 0.6  # Alpha parameter
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample according to these probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        # Get the samples
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            states.append(self.memory[idx][0])
            actions.append(self.memory[idx][1])
            rewards.append(self.memory[idx][2])
            next_states.append(self.memory[idx][3])
            dones.append(self.memory[idx][4])
        
        # Convert to arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Update Q-values
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)

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
        
        # Action mapping for debugging
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
        """Build a deeper neural network for predicting Q-values"""
        model = Sequential()
        
        # Input layer and first hidden layer
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        
        # Deeper network with intermediate layers
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        
        # Output layer
        model.add(Dense(self.action_size, activation='linear'))
        
        # Use Adam optimizer with a smaller learning rate
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory with robust state size handling"""
        # Convert to numpy arrays if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        
        # Flatten multi-dimensional states
        if len(state.shape) > 1:
            state = state.flatten()
        if len(next_state.shape) > 1:
            next_state = next_state.flatten()
        
        # Handle size mismatch by padding or truncating
        if len(state) != self.state_size:
            padded_state = np.zeros(self.state_size, dtype=np.float32)
            min_size = min(len(state), self.state_size)
            padded_state[:min_size] = state[:min_size]
            state = padded_state
            
        if len(next_state) != self.state_size:
            padded_next_state = np.zeros(self.state_size, dtype=np.float32)
            min_size = min(len(next_state), self.state_size)
            padded_next_state[:min_size] = next_state[:min_size]
            next_state = padded_next_state
        
        # Store the padded states in memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Track rewards for analytics
        self.recent_rewards.append(reward)
    
    def act(self, state, valid_actions=None):
        """Choose an action with robust state size handling"""
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
            
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        if state.shape[1] != self.state_size:
            print(f"WARNING: Strategy agent handling state size mismatch: got {state.shape[1]}, expected {self.state_size}")
            
            padded_state = np.zeros((1, self.state_size), dtype=np.float32)
            
            min_size = min(state.shape[1], self.state_size)
            padded_state[0, :min_size] = state[0, :min_size]
            
            state = padded_state
        
        # Exploration-exploitation logic
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None:
                return random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        
        if valid_actions is not None:
            mask = np.ones(self.action_size) * -1000000
            for action in valid_actions:
                mask[action] = 0
            act_values = act_values + mask
        
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train the agent with experiences from memory with size adaptation"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Get expected input size from model
        expected_size = 77  # Hardcoded based on the error message
        
        # Pre-process states and next_states for consistent shapes
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in minibatch:
            # Convert to numpy if needed
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)
            
            # Flatten multi-dimensional states
            if len(state.shape) > 1:
                state = state.flatten()
            if len(next_state.shape) > 1:
                next_state = next_state.flatten()
            
            # Handle size mismatch by padding
            if len(state) != expected_size:
                padded_state = np.zeros(expected_size, dtype=np.float32)
                padded_state[:len(state)] = state
                state = padded_state
                
            if len(next_state) != expected_size:
                padded_next_state = np.zeros(expected_size, dtype=np.float32)
                padded_next_state[:len(next_state)] = next_state
                next_state = padded_next_state
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Get target values
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update targets for actions taken
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
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
        """Load model from file with compatibility check"""
        try:
            loaded_model = tf.keras.models.load_model(file_path)
            
            expected_input_shape = (None, self.state_size)
            actual_input_shape = loaded_model.layers[0].input_shape
            
            if actual_input_shape[1] != self.state_size:
                print(f"Warning: Loaded model expects input dimension {actual_input_shape[1]}, " +
                    f"but current state size is {self.state_size}. Creating new model.")
                return
                
            self.model = loaded_model
            self.update_target_model()
            print(f"Successfully loaded model from {file_path}")
        except Exception as e:
            print(f"Error loading model: {e}. Creating new model.")
    
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


def get_valid_strategy_actions(self):
    """Return valid strategy actions based on current game state"""
    valid_actions = []
    
    # Create shop if it doesn't exist yet
    if self.current_shop is None:
        self.update_shop()
    
    # Check which shop items we can afford
    for i in range(4):  # 4 shop slots
        if (i < len(self.current_shop.items) and 
            self.current_shop.items[i] is not None and 
            self.game_manager.game.inventory.money >= self.current_shop.get_item_price(i)):
            valid_actions.append(i)  # Buy shop item
    
    # Check if we can sell jokers
    joker_count = len(self.game_manager.game.inventory.jokers)
    if joker_count > 0:
        for i in range(min(joker_count, 5)):
            valid_actions.append(i + 4)  # Sell joker
    
    # Check if we can use tarot cards
    tarot_indices = self.game_manager.game.inventory.get_consumable_tarot_indices()
    for i, tarot_idx in enumerate(tarot_indices):
        if i < 2:  # Limit to first 2 tarots for simplicity
            valid_actions.append(9 + i*3)  # Use tarot with no cards
            valid_actions.append(10 + i*3)  # Use tarot with lowest cards
            valid_actions.append(11 + i*3)  # Use tarot with highest cards
    
    # Always valid to advance to next ante
    valid_actions.append(15)  # Skip action (next ante)
    
    # If no valid actions, can always skip
    if not valid_actions:
        valid_actions.append(15)
    
    return valid_actions

def handle_shop_strategy(self, env):
    """
    Automated shop strategy based on GameTest.py logic
    This can be used to provide demonstration examples
    """
    inventory = env.game_manager.game.inventory
    valid_actions = []
    
    joker_count = len(inventory.jokers)
    
    # Buy affordable jokers if we have space
    if joker_count < 5:
        for i in range(4):
            if (i < len(self.current_shop.items) and 
                self.current_shop.items[i] is not None and 
                self.current_shop.items[i].item_type == ShopItemType.JOKER and
                inventory.money >= self.current_shop.get_item_price(i)):
                valid_actions.append(i)
    
    # Buy planets (auto-use)
    for i in range(4):
        if (i < len(self.current_shop.items) and 
            self.current_shop.items[i] is not None and 
            self.current_shop.items[i].item_type == ShopItemType.PLANET and
            inventory.money >= self.current_shop.get_item_price(i)):
            valid_actions.append(i)
    
    # Buy tarot cards
    for i in range(4):
        if (i < len(self.current_shop.items) and 
            self.current_shop.items[i] is not None and 
            self.current_shop.items[i].item_type == ShopItemType.TAROT and
            inventory.money >= self.current_shop.get_item_price(i)):
            valid_actions.append(i)
    
    # Sell a joker if we have too many
    if joker_count > 4 and random.random() < 0.5:
        # Find lowest value joker
        min_value_idx = min(range(joker_count), key=lambda i: inventory.jokers[i].sell_value)
        valid_actions.append(min_value_idx + 4)
    
    # If no valid actions or we've already bought what we want
    if not valid_actions:
        return 15  # Skip to next ante
    
    # Choose an action
    return random.choice(valid_actions)

def use_pending_tarots(self):
    """
    Use tarot cards that were purchased from the shop, similar to GameTest.py
    """
    if not self.pending_tarots:
        return False
    
    print("\n=== Using Tarot Cards From Shop ===")
    used_any = False
    
    for tarot_name in self.pending_tarots.copy():
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
        cards_required = tarot.selected_cards_required if hasattr(tarot, 'selected_cards_required') else 0
        
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
            self.pending_tarots.remove(tarot_name)
            used_any = True
        else:
            print(f"Failed to use {tarot_name}: {message}")
    
    return used_any

def handle_pack_opening(self, pack_type, item_index):
    """
    Handle opening a booster pack and selecting items from it, similar to GameTest.py
    
    Args:
        pack_type: The type of pack (Standard, Celestial, Arcana, etc.)
        item_index: The index of the item in the shop
    """
    from JokerCreation import create_joker
    from Tarot import create_tarot_by_name
    from Planet import create_planet_by_name
    from Card import Card
    from Enums import Suit, Rank, CardEnhancement
    
    if self.current_shop.items[item_index] is None:
        print("Error: No item at this index")
        return False
    
    shop_item = self.current_shop.items[item_index]
    pack_contents = None
    
    # Get pack contents if available
    if hasattr(shop_item, 'contents'):
        pack_contents = shop_item.contents
    else:
        # Try to get contents from predefined packs
        if "STANDARD" in pack_type.upper():
            # Example standard pack contents - should be replaced with actual logic
            pack_contents = ["A ♥", "K ♠", "Q ♦", "J ♣", "10 ♥"]
    
    if not pack_contents:
        print(f"Error: No contents found for {pack_type}")
        return False
    
    print(f"\n=== Opening {pack_type} ===")
    print("Pack contents:")
    for i, item in enumerate(pack_contents):
        print(f"{i}: {item}")
    
    # Process based on pack type
    if "STANDARD" in pack_type.upper():
        # Simple AI: randomly select a card from the pack
        selected_idx = random.randint(0, len(pack_contents) - 1)
        card_string = pack_contents[selected_idx]
        
        # Process the card (similar to GameTest.py but simplified)
        parts = card_string.split()
        
        # Map strings to proper Rank enums
        rank_map = {
            "A": Rank.ACE, 
            "2": Rank.TWO,
            "3": Rank.THREE,
            "4": Rank.FOUR,
            "5": Rank.FIVE,
            "6": Rank.SIX,
            "7": Rank.SEVEN,
            "8": Rank.EIGHT,
            "9": Rank.NINE,
            "10": Rank.TEN,
            "J": Rank.JACK, 
            "Q": Rank.QUEEN, 
            "K": Rank.KING
        }
        
        # Map strings to proper Suit enums
        suit_map = {
            "heart": Suit.HEARTS, 
            "hearts": Suit.HEARTS, 
            "♥": Suit.HEARTS,
            "diamond": Suit.DIAMONDS, 
            "diamonds": Suit.DIAMONDS, 
            "♦": Suit.DIAMONDS,
            "club": Suit.CLUBS, 
            "clubs": Suit.CLUBS, 
            "♣": Suit.CLUBS,
            "spade": Suit.SPADES, 
            "spades": Suit.SPADES, 
            "♠": Suit.SPADES
        }
        
        # Extract rank and suit
        rank_str = parts[0] if parts else "A"
        suit_str = parts[-1].lower() if len(parts) > 1 else "hearts"
        
        rank = rank_map.get(rank_str, Rank.ACE)
        suit = suit_map.get(suit_str, Suit.HEARTS)
        
        # Create and add card
        try:
            card = Card(suit, rank)
            self.game_manager.game.inventory.add_card_to_deck(card)
            print(f"Added {card_string} to deck")
            return True
        except Exception as e:
            print(f"Error processing card: {e}")
            return False
            
    elif "CELESTIAL" in pack_type.upper():
        # Handle celestial packs (planets)
        selected_idx = random.randint(0, len(pack_contents) - 1)
        planet_name = pack_contents[selected_idx]
        
        try:
            planet = create_planet_by_name(planet_name)
            if planet and hasattr(planet, 'planet_type'):
                planet_type = planet.planet_type
                current_level = self.game_manager.game.inventory.planet_levels.get(planet_type, 1)
                self.game_manager.game.inventory.planet_levels[planet_type] = current_level + 1
                
                print(f"Used {planet_name} planet to upgrade to level {current_level + 1}")
                return True
            else:
                print(f"Failed to process planet {planet_name}")
                return False
        except Exception as e:
            print(f"Error processing planet: {e}")
            return False
    
    # Handle other pack types if needed
    return False


def train_with_curriculum():
    """Enhanced curriculum learning approach with proper state size handling"""
    # Create the base environment
    env = BalatroEnv(config={'simplified': True})  # No bootstraps
    
    # Get accurate state sizes for both agents
    play_state = env._get_play_state()
    play_state_size = len(play_state)
    print(f"Play state size: {play_state_size}")
    
    strategy_state = env._get_strategy_state()
    strategy_state_size = len(strategy_state)
    print(f"Strategy state size: {strategy_state_size}")
    
    # Create the play agent with correct size
    play_agent = PlayingAgent(state_size=play_state_size, 
                             action_size=env._define_play_action_space())
    
    # Add demonstration examples to jumpstart learning
    print("Adding demonstration examples...")
    add_demonstration_examples(play_agent, num_examples=200)
    
    # Phase 1: Train with simplified game (no shop/strategy agent yet)
    print("\n===== PHASE 1: LEARNING BASIC CARD PLAY =====")
    for episode in range(500):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            valid_actions = env.get_valid_play_actions()
            action = play_agent.act(state, valid_actions)
            next_state, reward, done, info = env.step_play(action)
            
            play_agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Handle shop phase by skipping it entirely during Phase 1
            if info.get('shop_phase', False) and not done:
                # Skip the shop phase entirely by advancing to next ante
                success = env.game_manager.next_ante()
                if success:
                    # Get a new state after ante advancement
                    state = env._get_play_state()
        
        # Train the agent
        if len(play_agent.memory) > 64:
            play_agent.replay(64)
        
        play_agent.decay_epsilon()
        
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1}/500, Reward: {total_reward:.2f}, Epsilon: {play_agent.epsilon:.3f}")
    
    # Phase 2: Train a separate strategy agent
    print("\n===== PHASE 2: ADDING SHOP STRATEGY =====")
    # Create strategy agent with correct size
    strategy_agent = StrategyAgent(state_size=strategy_state_size,
                                  action_size=env._define_strategy_action_space())
    
    for episode in range(1000):
        state = env.reset()
        play_state = state  # Initial state is play state
        done = False
        play_total_reward = 0
        strategy_total_reward = 0
        in_shop_phase = False
        
        while not done:
            if not in_shop_phase:
                # Play phase
                valid_actions = env.get_valid_play_actions()
                action = play_agent.act(play_state, valid_actions)
                next_play_state, reward, done, info = env.step_play(action)
                
                play_agent.remember(play_state, action, reward, next_play_state, done)
                play_state = next_play_state
                play_total_reward += reward
                
                # Check if we need to enter shop phase
                if info.get('shop_phase', False) and not done:
                    in_shop_phase = True
                    env.update_shop()
                    strategy_state = env._get_strategy_state()  # Get strategy state
            else:
                # Shop phase
                valid_actions = env.get_valid_strategy_actions()
                action = strategy_agent.act(strategy_state, valid_actions)
                next_strategy_state, reward, done, info = env.step_strategy(action)
                
                strategy_agent.remember(strategy_state, action, reward, next_strategy_state, done)
                strategy_state = next_strategy_state
                strategy_total_reward += reward
                
                # Check if we need to exit shop phase
                if action == 15 or not env.game_manager.current_ante_beaten or done:
                    in_shop_phase = False
                    # Get play state again
                    if not done:
                        play_state = env._get_play_state()
        
        # Train both agents
        if len(play_agent.memory) > 64:
            play_agent.replay(64)
        
        if len(strategy_agent.memory) > 64:
            strategy_agent.replay(64)
        
        play_agent.decay_epsilon()
        strategy_agent.decay_epsilon()
        
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1}/1000, Play Reward: {play_total_reward:.2f}, " +
                  f"Strategy Reward: {strategy_total_reward:.2f}")
    
    # Phase 3: Full game with all features
    print("\n===== PHASE 3: FULL GAME TRAINING (2000 episodes) =====")
    env = BalatroEnv(config={'full_features': True})
    
    # Save final models
    play_agent.save_model("play_agent_curriculum_final.h5")
    strategy_agent.save_model("strategy_agent_curriculum_final.h5")
    
    # Final evaluation
    results = evaluate_agents(play_agent, strategy_agent, episodes=50)
    print("\n===== CURRICULUM TRAINING COMPLETE =====")
    print(f"Win rate: {results['win_rate']:.2f}%")
    print(f"Average max ante: {results['average_score']:.2f}")
    
    return play_agent, strategy_agent


def add_demonstration_examples(play_agent, num_examples=300):
    """Add expert demonstration examples to the agent's memory with better poker hand recognition"""
    env = BalatroEnv(config={'simplified': True})
    hand_evaluator = HandEvaluator()  # Create a hand evaluator instance
    
    print(f"Adding {num_examples} demonstration examples...")
    examples_added = 0
    
    for _ in range(num_examples):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 50:
            steps += 1
            
            # Evaluate all possible hands from current cards
            potential_hands = []
            current_hand = env.game_manager.current_hand
            
            # Try all combinations of 5 or more cards
            for r in range(5, len(current_hand) + 1):
                for combo in itertools.combinations(range(len(current_hand)), r):
                    cards = [current_hand[i] for i in combo]
                    hand_type, _, scoring_cards = hand_evaluator.evaluate_hand(cards)
                    hand_value = hand_type.value
                    potential_hands.append((list(combo), hand_value, hand_type))
            
            # Sort by hand value (descending)
            potential_hands.sort(key=lambda x: x[1], reverse=True)
            
            # If we have a good hand, play it
            if potential_hands and potential_hands[0][1] >= HandType.TWO_PAIR.value:
                indices = potential_hands[0][0]
                action = 0
                for idx in indices:
                    action |= (1 << idx)
                
                next_state, reward, done, info = env.step_play(action)
                
                # Enhance reward for good hands
                enhanced_reward = reward * 1.5  # Boost the reward
                
                # Add this experience to memory
                play_agent.remember(state, action, enhanced_reward, next_state, done)
                examples_added += 1
                
                state = next_state
            
            # If no good hand and we can discard, do so
            elif env.game_manager.discards_used < env.game_manager.max_discards_per_round:
                # Find the lowest cards to discard
                cards_with_values = [(i, card.rank.value) for i, card in enumerate(current_hand)]
                cards_with_values.sort(key=lambda x: x[1])  # Sort by rank value
                
                # Discard up to 3 of the lowest cards
                indices = [idx for idx, _ in cards_with_values[:min(3, len(cards_with_values))]]
                
                action = 0
                for idx in indices:
                    action |= (1 << idx)
                
                # Make it a discard action
                action += 256
                
                next_state, reward, done, info = env.step_play(action)
                play_agent.remember(state, action, reward, next_state, done)
                examples_added += 1
                
                state = next_state
            
            # Otherwise, play the best hand we have
            else:
                if potential_hands:
                    indices = potential_hands[0][0]  # Best hand we have
                    action = 0
                    for idx in indices:
                        action |= (1 << idx)
                else:
                    # Play all cards as last resort
                    action = (1 << len(current_hand)) - 1
                
                next_state, reward, done, info = env.step_play(action)
                play_agent.remember(state, action, reward, next_state, done)
                examples_added += 1
                
                state = next_state
            
            # Handle shop phase if needed - just skip it for demonstration
            if info.get('shop_phase', False) and not done:
                next_state, _, done, _ = env.step_strategy(15)  # Skip action
                state = next_state
    
    print(f"Successfully added {examples_added} demonstration examples to memory")
    return examples_added

def train_agents(episodes=10000, batch_size=64, game_config=None, save_interval=500, play_agent=None, strategy_agent=None, log_interval=100):
    if game_config is None:
        game_config = {}

    # Print what configuration is being used
    print(f"Training with config: {game_config}")
    
    env = BalatroEnv(config=game_config)
    
    # Get accurate state sizes for validation
    initial_play_state = env._get_play_state()
    play_state_size = len(initial_play_state)
    
    # Initialize play agent if not provided
    if play_agent is None:
        print(f"Creating new PlayingAgent with state_size={play_state_size}")
        play_agent = PlayingAgent(state_size=play_state_size, 
                                action_size=env._define_play_action_space())
    elif play_agent.state_size != play_state_size:
        print(f"Warning: Play agent state size mismatch. Expected {play_state_size}, got {play_agent.state_size}")
        print("Creating new play agent with correct size.")
        play_agent = PlayingAgent(state_size=play_state_size, 
                                action_size=env._define_play_action_space())
    
    # Only initialize strategy agent if provided or needed
    if strategy_agent is not None:
        strategy_state = env._get_strategy_state()
        strategy_state_size = len(strategy_state)
        
        if strategy_agent.state_size != strategy_state_size:
            print(f"Warning: Strategy agent state size mismatch. Expected {strategy_state_size}, got {strategy_agent.state_size}")
            print("Creating new strategy agent with correct size.")
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
        shop_steps = 0  # Track steps in shop phase separately
        
        done = False
        in_shop_phase = False
        
        while not done and game_steps < 500:
            if not in_shop_phase:
                valid_play_actions = env.get_valid_play_actions()
                play_action = play_agent.act(state, valid_actions=valid_play_actions)
                next_play_state, play_reward, done, info = env.step_play(play_action)
                
                # Get normalized next state
                next_state = env.get_normalized_state(is_shop_phase=False)
                
                # Remember play experience
                play_agent.remember(state, play_action, play_reward, next_state, done)
                state = next_state
                play_total_reward += play_reward
                
                # Check if we need to enter shop phase
                if info.get('shop_phase', False) and not done:
                    in_shop_phase = True
                    env.update_shop()
                    # Important: Get the normalized state for shop phase
                    state = env.get_normalized_state(is_shop_phase=True)
            
            # SHOP PHASE
            elif strategy_agent is not None:
                # Now we're in the strategy phase
                valid_strategy_actions = env.get_valid_strategy_actions()
                strategy_action = strategy_agent.act(state, valid_actions=valid_strategy_actions)
                
                # Execute the action
                next_strategy_state, strategy_reward, strategy_done, strategy_info = env.step_strategy(strategy_action)
                
                # Get normalized next state
                next_state = env.get_normalized_state(is_shop_phase=True)
                
                # Remember strategy experience
                strategy_agent.remember(state, strategy_action, strategy_reward, 
                                      next_state, strategy_done)
                
                strategy_total_reward += strategy_reward
                done = strategy_done
                
                # Check if we need to exit shop phase
                if strategy_action == 15 or not env.game_manager.current_ante_beaten or done:
                    in_shop_phase = False
                    
                    # Important: Get normalized state for play phase
                    if not done:
                        state = env.get_normalized_state(is_shop_phase=False)
                else:
                    state = next_state
        
        # Training after episode completes
        if len(play_agent.memory) > batch_size:
            play_agent.replay(batch_size)
        
        if strategy_agent is not None and len(strategy_agent.memory) > batch_size:
            strategy_agent.replay(batch_size)
        
        play_agent.decay_epsilon()
        if strategy_agent is not None:
            strategy_agent.decay_epsilon()
        
        play_rewards.append(play_total_reward)
        if strategy_agent is not None:
            strategy_rewards.append(strategy_total_reward)
        ante_progression.append(max_ante_reached)
        
        if (e + 1) % log_interval == 0:
            avg_play_reward = sum(play_rewards[-log_interval:]) / log_interval
            avg_ante = sum(ante_progression[-log_interval:]) / log_interval
            
            print(f"\nEpisode {e+1}/{episodes}")
            print(f"  Play Agent: reward={avg_play_reward:.2f}, epsilon={play_agent.epsilon:.3f}")
            
            if strategy_agent is not None:
                avg_strategy_reward = sum(strategy_rewards[-log_interval:]) / log_interval
                print(f"  Strategy Agent: reward={avg_strategy_reward:.2f}, epsilon={strategy_agent.epsilon:.3f}")
            
            print(f"  Average max ante: {avg_ante:.2f}")
            
            play_stats = play_agent.get_stats()
            print(f"  Play stats: {play_stats}")
        
        if (e + 1) % save_interval == 0:
            print(f"Saving models at episode {e+1}")
            play_agent.save_model(f"play_agent_ep{e+1}.h5")
            if strategy_agent is not None:
                strategy_agent.save_model(f"strategy_agent_ep{e+1}.h5")
            
            play_agent.save_model("play_agent_latest.h5")
            if strategy_agent is not None:
                strategy_agent.save_model("strategy_agent_latest.h5")
    
    play_agent.save_model("play_agent_final.h5")
    if strategy_agent is not None:
        strategy_agent.save_model("strategy_agent_final.h5")
    
    return play_agent, strategy_agent

def train_with_separate_agents():
    """Train with fully isolated state handling and force progression"""
    env = BalatroEnv(config={'simplified': True})
    
    # Create agents with correct dimensions
    play_state_size = len(env._get_play_state())
    play_action_size = env._define_play_action_space()
    
    strategy_state_size = len(env._get_strategy_state())
    strategy_action_size = env._define_strategy_action_space()
    
    print(f"Play agent: state_size={play_state_size}, action_size={play_action_size}")
    print(f"Strategy agent: state_size={strategy_state_size}, action_size={strategy_action_size}")
    
    # Create fresh agents
    play_agent = PlayingAgent(state_size=play_state_size, action_size=play_action_size)
    strategy_agent = StrategyAgent(state_size=strategy_state_size, action_size=strategy_action_size)
    
    # Isolate act methods to ensure proper state handling
    def play_act(state, valid_actions=None):
        # Ensure state has play dimensions
        if isinstance(state, np.ndarray) and len(state.shape) == 2 and state.shape[1] != play_state_size:
            print(f"WARNING: Play agent received wrong state size: {state.shape[1]}, expected {play_state_size}")
            # Get a fresh play state as fallback
            state = env._get_play_state()
            state = np.array(state, dtype=np.float32).reshape(1, -1)
        
        # Call original act method
        return play_agent.act(state, valid_actions)
    
    def strategy_act(state, valid_actions=None):
        # Ensure state has strategy dimensions
        if isinstance(state, np.ndarray) and len(state.shape) == 2 and state.shape[1] != strategy_state_size:
            print(f"WARNING: Strategy agent received wrong state size: {state.shape[1]}, expected {strategy_state_size}")
            # Get a fresh strategy state as fallback
            state = env._get_strategy_state()
            state = np.array(state, dtype=np.float32).reshape(1, -1)
        
        # Call original act method
        return strategy_agent.act(state, valid_actions)
    
    # Create separate memory buffers
    play_memory = []
    strategy_memory = []
    
    # Training loop
    for episode in range(3000):
        env.reset()
        done = False
        play_total_reward = 0
        strategy_total_reward = 0
        
        # Track shop phase to detect loops
        shop_steps = 0
        max_shop_steps = 20  # Maximum number of shop actions before forcing skip
        
        while not done:
            # PLAY PHASE
            # Always get a fresh play state
            play_state = env._get_play_state()
            play_state = np.array(play_state, dtype=np.float32).reshape(1, -1)
            
            valid_actions = env.get_valid_play_actions()
            play_action = play_act(play_state, valid_actions)
            _, play_reward, done, info = env.step_play(play_action)
            
            # Get next play state
            next_play_state = env._get_play_state()
            next_play_state = np.array(next_play_state, dtype=np.float32).reshape(1, -1)
            
            # Store play experience
            play_memory.append((play_state[0], play_action, play_reward, next_play_state[0], done))
            play_total_reward += play_reward
            
            # SHOP PHASE
            if info.get('shop_phase', False) and not done:
                env.update_shop()
                shop_done = False
                shop_steps = 0  # Reset shop step counter
                
                while not shop_done and not done:
                    shop_steps += 1
                    
                    # Always get a fresh strategy state
                    strategy_state = env._get_strategy_state()
                    
                    # Force correct shape
                    if not isinstance(strategy_state, np.ndarray):
                        strategy_state = np.array(strategy_state, dtype=np.float32)
                    strategy_state = strategy_state.reshape(1, -1)
                    
                    valid_actions = env.get_valid_strategy_actions()
                    
                    # IMPORTANT: Force the "Skip" action in specific cases to prevent infinite loops
                    force_skip = False
                    
                    # Case 1: Early episodes, encourage exploration of the game flow
                    if episode < 10:
                        force_skip = True
                    
                    # Case 2: Detect shop phase loops by limiting shop steps
                    if shop_steps >= max_shop_steps:
                        print(f"Forcing shop exit after {shop_steps} steps to prevent infinite loop")
                        force_skip = True
                    
                    # Case 3: Random chance to skip (increasing with shop_steps)
                    if shop_steps > 5 and random.random() < (shop_steps / 50):
                        force_skip = True
                    
                    if force_skip:
                        strategy_action = 15  # Skip action (advance to next ante)
                    else:
                        strategy_action = strategy_act(strategy_state, valid_actions)
                    
                    _, strategy_reward, done, info = env.step_strategy(strategy_action)
                    
                    # Get next strategy state
                    next_strategy_state = env._get_strategy_state()
                    if not isinstance(next_strategy_state, np.ndarray):
                        next_strategy_state = np.array(next_strategy_state, dtype=np.float32)
                    next_strategy_state = next_strategy_state.reshape(1, -1)
                    
                    # Store strategy experience (with higher reward for Skip in loop conditions)
                    if force_skip and shop_steps >= max_shop_steps:
                        # Provide extra reward for Skip action when it breaks a loop
                        strategy_reward += 2.0
                    
                    strategy_memory.append((strategy_state[0], strategy_action, strategy_reward, next_strategy_state[0], done))
                    strategy_total_reward += strategy_reward
                    
                    # Always exit shop phase after skip action
                    if strategy_action == 15 or done:
                        shop_done = True
        
        # Train play agent
        if len(play_memory) >= 32:
            # Sample batch
            minibatch = random.sample(play_memory, 32)
            
            # Convert to arrays
            states = np.array([exp[0] for exp in minibatch])
            actions = np.array([exp[1] for exp in minibatch])
            rewards = np.array([exp[2] for exp in minibatch])
            next_states = np.array([exp[3] for exp in minibatch])
            dones = np.array([exp[4] for exp in minibatch])
            
            # Get target values
            targets = play_agent.model.predict(states, verbose=0)
            next_q_values = play_agent.target_model.predict(next_states, verbose=0)
            
            # Update targets
            for i in range(len(minibatch)):
                if dones[i]:
                    targets[i, actions[i]] = rewards[i]
                else:
                    targets[i, actions[i]] = rewards[i] + play_agent.gamma * np.max(next_q_values[i])
            
            # Train model
            play_agent.model.fit(states, targets, epochs=1, verbose=0)
        
        # Train strategy agent
        if len(strategy_memory) >= 16:
            # Sample batch
            minibatch = random.sample(strategy_memory, 16)
            
            # Convert to arrays
            states = np.array([exp[0] for exp in minibatch])
            actions = np.array([exp[1] for exp in minibatch])
            rewards = np.array([exp[2] for exp in minibatch])
            next_states = np.array([exp[3] for exp in minibatch])
            dones = np.array([exp[4] for exp in minibatch])
            
            # Get target values
            targets = strategy_agent.model.predict(states, verbose=0)
            next_q_values = strategy_agent.target_model.predict(next_states, verbose=0)
            
            # Update targets
            for i in range(len(minibatch)):
                if dones[i]:
                    targets[i, actions[i]] = rewards[i]
                else:
                    targets[i, actions[i]] = rewards[i] + strategy_agent.gamma * np.max(next_q_values[i])
            
            # Train model
            strategy_agent.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decay exploration
        play_agent.epsilon *= play_agent.epsilon_decay
        strategy_agent.epsilon *= strategy_agent.epsilon_decay
        
        # Cap epsilon at minimum
        play_agent.epsilon = max(play_agent.epsilon, play_agent.epsilon_min)
        strategy_agent.epsilon = max(strategy_agent.epsilon, strategy_agent.epsilon_min)
        
        # Log progress
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/3000")
            print(f"  Play reward: {play_total_reward:.2f}")
            print(f"  Strategy reward: {strategy_total_reward:.2f}")
            print(f"  Play agent epsilon: {play_agent.epsilon:.3f}")
            print(f"  Strategy agent epsilon: {strategy_agent.epsilon:.3f}")
            print(f"  Max ante: {env.game_manager.game.current_ante}")
            print(f"  Play memory size: {len(play_memory)}")
            print(f"  Strategy memory size: {len(strategy_memory)}")
        
        # Save models
        if (episode + 1) % 500 == 0:
            play_agent.save_model(f"play_agent_ep{episode+1}.h5")
            strategy_agent.save_model(f"strategy_agent_ep{episode+1}.h5")
            
            # Trim memories to prevent excessive growth
            if len(play_memory) > 10000:
                play_memory = play_memory[-10000:]
            if len(strategy_memory) > 5000:
                strategy_memory = strategy_memory[-5000:]
    
    # Final save
    play_agent.save_model("play_agent_final.h5")
    strategy_agent.save_model("strategy_agent_final.h5")
    
    return play_agent, strategy_agent

def play_replay(agent, batch_size):
    """Custom replay for play agent to avoid shape mismatch issues"""
    if len(agent.memory) < batch_size:
        return
    
    minibatch = random.sample(agent.memory, batch_size)
    
    # Extract experience components
    states = np.array([exp[0] for exp in minibatch])
    actions = np.array([exp[1] for exp in minibatch])
    rewards = np.array([exp[2] for exp in minibatch])
    next_states = np.array([exp[3] for exp in minibatch])
    dones = np.array([exp[4] for exp in minibatch])
    
    # Validate shapes
    if states.shape[1] != agent.state_size:
        print(f"WARNING: Play state shape mismatch: expected {agent.state_size}, got {states.shape[1]}")
        return
    
    # Get predictions
    targets = agent.model.predict(states, verbose=0)
    next_q_values = agent.target_model.predict(next_states, verbose=0)
    
    # Update targets for actions taken
    for i in range(batch_size):
        if dones[i]:
            targets[i, actions[i]] = rewards[i]
        else:
            targets[i, actions[i]] = rewards[i] + agent.gamma * np.max(next_q_values[i])
    
    # Train the model
    agent.model.fit(states, targets, epochs=1, verbose=0)

def strategy_replay(agent, batch_size):
    """Custom replay for strategy agent to avoid shape mismatch issues"""
    if len(agent.memory) < batch_size:
        return
    
    minibatch = random.sample(agent.memory, batch_size)
    
    # Extract experience components
    states = np.array([exp[0] for exp in minibatch])
    actions = np.array([exp[1] for exp in minibatch])
    rewards = np.array([exp[2] for exp in minibatch])
    next_states = np.array([exp[3] for exp in minibatch])
    dones = np.array([exp[4] for exp in minibatch])
    
    # Validate shapes
    if states.shape[1] != agent.state_size:
        print(f"WARNING: Strategy state shape mismatch: expected {agent.state_size}, got {states.shape[1]}")
        return
    
    # Get predictions
    targets = agent.model.predict(states, verbose=0)
    next_q_values = agent.target_model.predict(next_states, verbose=0)
    
    # Update targets for actions taken
    for i in range(batch_size):
        if dones[i]:
            targets[i, actions[i]] = rewards[i]
        else:
            targets[i, actions[i]] = rewards[i] + agent.gamma * np.max(next_q_values[i])
    
    # Train the model
    agent.model.fit(states, targets, epochs=1, verbose=0)


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
        env.reset()
        play_state = env._get_play_state()
        play_total_reward = 0
        strategy_total_reward = 0
        
        max_ante = 1
        hands_played = 0
        game_won = False
        game_steps = 0
        
        done = False
        
        # Game loop
        while not done and game_steps < 500:
            game_steps += 1
            
            # PLAY PHASE
            valid_play_actions = env.get_valid_play_actions()
            play_action = play_agent.act(play_state, valid_actions=valid_play_actions)
            next_play_state, play_reward, done, info = env.step_play(play_action)
            
            play_state = next_play_state
            play_total_reward += play_reward
            
            hands_played += 1
            max_ante = max(max_ante, env.game_manager.game.current_ante)
            
            # Check if we need to enter shop phase
            if info.get('shop_phase', False) and not done:
                strategy_state = env._get_strategy_state()
                shop_done = False
                
                # Shop phase loop
                while not shop_done and not done and game_steps < 500:
                    game_steps += 1
                    valid_strategy_actions = env.get_valid_strategy_actions()
                    strategy_action = strategy_agent.act(strategy_state, valid_strategy_actions)
                    
                    next_strategy_state, strategy_reward, strategy_done, _ = env.step_strategy(strategy_action)
                    strategy_state = next_strategy_state
                    strategy_total_reward += strategy_reward
                    done = strategy_done
                    
                    # Check if we need to exit shop phase
                    if strategy_action == 15 or not env.game_manager.current_ante_beaten or done:
                        shop_done = True
                
                # Get fresh play state after shop
                if not done:
                    play_state = env._get_play_state()
        
        # Record results
        results['play_rewards'].append(play_total_reward)
        results['strategy_rewards'].append(strategy_total_reward)
        results['max_antes'].append(max_ante)
        results['hands_played'].append(hands_played)
        
        # Consider a win if player reached ante 8 or higher
        if max_ante >= 8:
            game_won = True
        results['win_rate'] += 1 if game_won else 0
        
        if (e + 1) % 10 == 0:
            print(f"Evaluated {e + 1}/{episodes} episodes. Current max ante: {max_ante}")
    
    # Calculate averages
    results['win_rate'] = results['win_rate'] / episodes * 100
    results['average_score'] = sum(results['max_antes']) / episodes
    
    # Restore original epsilon values
    play_agent.epsilon = play_epsilon
    strategy_agent.epsilon = strategy_epsilon
    
    return results

def evaluate_agent(play_agent, num_episodes=10, config=None):
    """Evaluate an agent's performance"""
    env = BalatroEnv(config=config)
    
    results = {
        'avg_score': 0,
        'max_ante': 0,
        'win_rate': 0
    }
    
    total_score = 0
    max_ante_reached = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_score = 0
        
        while not done:
            valid_actions = env.get_valid_play_actions()
            action = play_agent.act(state, valid_actions=valid_actions)
            next_state, reward, done, info = env.step_play(action)
            
            state = next_state
            episode_score = env.game_manager.current_score
            
            # Handle shop phase if needed
            if info.get('shop_phase', False) and not done:
                # Just skip shop for evaluation
                next_state, _, done, _ = env.step_strategy(15)  # Skip action
                state = next_state
        
        total_score += episode_score
        max_ante_reached = max(max_ante_reached, env.game_manager.game.current_ante)
        
        if env.game_manager.game.current_ante >= 8:
            results['win_rate'] += 1
    
    results['avg_score'] = total_score / num_episodes
    results['max_ante'] = max_ante_reached
    results['win_rate'] = (results['win_rate'] / num_episodes) * 100
    
    return results

def train_with_game_phases():
    """
    Train agents with clear separation between gameplay and shop phases,
    similar to the structure in GameTest.py
    """
    env = BalatroEnv()
    
    # Initialize agents
    play_state_size = len(env._get_play_state())
    play_agent = PlayingAgent(state_size=play_state_size, action_size=env._define_play_action_space())
    
    strategy_state_size = len(env._get_strategy_state())
    strategy_agent = StrategyAgent(state_size=strategy_state_size, action_size=env._define_strategy_action_space())
    
    # Training parameters
    episodes = 5000
    batch_size = 64
    log_interval = 50
    save_interval = 500
    
    # Training stats
    play_rewards = []
    strategy_rewards = []
    max_antes = []
    games_won = 0
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        max_ante = 1
        game_steps = 0
        
        play_episode_memory = []  # Store experiences for batch update
        strategy_episode_memory = []  # Store experiences for batch update
        
        done = False
        show_shop_next = False  # Flag for shop phase, similar to GameTest.py
        pending_tarots = []
        
        # Game loop - structure similar to GameTest.py
        while not done and game_steps < 500:  # Reduced safety limit to avoid infinite loops
            game_steps += 1
            
            # Check for failure state - max hands played without beating ante
            if (env.game_manager.hands_played >= env.game_manager.max_hands_per_round and 
                env.game_manager.current_score < env.game_manager.game.current_blind and 
                not env.game_manager.current_ante_beaten and
                not show_shop_next):
                print(f"\nEpisode {e+1}: Failed to beat ante {env.game_manager.game.current_ante}")
                print(f"Score: {env.game_manager.current_score}/{env.game_manager.game.current_blind}")
                
                # Set game over
                env.game_manager.game_over = True
                done = True
                
                # Strong negative reward for failing
                total_reward -= 5.0
                
                # Store transition
                play_episode_memory.append((state, 0, -5.0, state, True))
                
                continue  # Skip to next episode
            
            # SHOP PHASE
            if show_shop_next:
                print(f"\n===== SHOP PHASE (Episode {e+1}) =====")
                
                # Make sure shop is updated for current ante
                env.update_shop()
                
                # Get strategy action
                strategy_state = env._get_strategy_state()
                valid_actions = env.get_valid_strategy_actions()
                
                if not valid_actions:
                    # Always include skip action
                    valid_actions = [15]  # Skip to next ante
                
                strategy_action = strategy_agent.act(strategy_state, valid_actions)
                
                # Execute shop action
                next_strategy_state, strategy_reward, strategy_done, strategy_info = env.step_strategy(strategy_action)
                
                # Store experience
                strategy_episode_memory.append((strategy_state, strategy_action, strategy_reward, next_strategy_state, strategy_done))
                
                total_reward += strategy_reward
                done = strategy_done
                
                # Check if we're done with shop (used "Skip" action)
                if strategy_action == 15:
                    show_shop_next = False
                    print(f"Advanced to Ante {env.game_manager.game.current_ante}")
                    
                    # Deal a new hand if needed
                    if not env.game_manager.current_hand:
                        env.game_manager.deal_new_hand()
                
                # Handle any pending tarots
                if pending_tarots and env.game_manager.current_hand:
                    print(f"Using {len(pending_tarots)} pending tarots")
                    for tarot_name in list(pending_tarots):  # Use a copy of the list for iteration
                        # Find the tarot in inventory
                        tarot_indices = env.game_manager.game.inventory.get_consumable_tarot_indices()
                        for idx in tarot_indices:
                            consumable = env.game_manager.game.inventory.consumables[idx]
                            if hasattr(consumable.item, 'name') and consumable.item.name == tarot_name:
                                # Simple strategy: use tarot with no selected cards
                                success, message = env.game_manager.use_tarot(idx, [])
                                if success:
                                    print(f"Used pending tarot {tarot_name}: {message}")
                                    pending_tarots.remove(tarot_name)
                    
                # Force shop exit after 20 steps to avoid getting stuck
                if game_steps % 20 == 0 and show_shop_next:
                    print(f"Forcing shop exit at step {game_steps}")
                    show_shop_next = False
                    
                    # Skip action to advance ante
                    _, _, strategy_done, _ = env.step_strategy(15)
                    done = strategy_done
                    
                    if not done:
                        # Ensure we have a hand
                        if not env.game_manager.current_hand:
                            env.game_manager.deal_new_hand()
                
                # Update state for next iteration
                if not done:
                    state = env._get_play_state()
                
                continue  # Skip to next iteration
            
            # REGULAR GAMEPLAY PHASE
            # Get valid actions
            valid_actions = env.get_valid_play_actions()
            
            if not valid_actions:
                print(f"Episode {e+1}: No valid actions at step {game_steps}")
                # If no valid actions and max hands reached without beating ante
                if (env.game_manager.hands_played >= env.game_manager.max_hands_per_round and 
                    env.game_manager.current_score < env.game_manager.game.current_blind):
                    print("Game over due to max hands without beating ante")
                    env.game_manager.game_over = True
                    done = True
                    play_episode_memory.append((state, 0, -5.0, state, True))
                    continue
                
                # Default to play all cards if we can
                if len(env.game_manager.current_hand) > 0:
                    print("Defaulting to play all cards")
                    action = (1 << len(env.game_manager.current_hand)) - 1
                else:
                    # If no cards to play, must be in a bad state - end episode
                    print("No cards in hand - ending episode")
                    done = True
                    continue
            else:
                # Choose action from valid actions
                action = play_agent.act(state, valid_actions)
            
            # Take action
            next_state, reward, done, info = env.step_play(action)
            
            # Store experience
            play_episode_memory.append((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward
            
            # Track maximum ante reached
            max_ante = max(max_ante, env.game_manager.game.current_ante)
            
            # Check if we need to enter shop phase (ante beaten)
            if info.get('shop_phase', False) and not done:
                print(f"\n===== ANTE {env.game_manager.game.current_ante} BEATEN! =====")
                print(f"Current Score: {env.game_manager.current_score}/{env.game_manager.game.current_blind}")
                show_shop_next = True
        
        # Learn from experiences collected in this episode
        # Update play agent
        for experience in play_episode_memory:
            play_agent.remember(*experience)
        
        if len(play_agent.memory) >= batch_size:
            play_agent.replay(batch_size)
        
        # Update strategy agent
        for experience in strategy_episode_memory:
            strategy_agent.remember(*experience)
            
        if len(strategy_agent.memory) >= batch_size:
            strategy_agent.replay(batch_size)
        
        # Decay exploration rates
        play_agent.decay_epsilon()
        strategy_agent.decay_epsilon()
        
        # Log progress
        play_rewards.append(total_reward)
        max_antes.append(max_ante)
        
        if (e + 1) % log_interval == 0:
            avg_reward = sum(play_rewards[-log_interval:]) / log_interval
            avg_ante = sum(max_antes[-log_interval:]) / log_interval
            
            print(f"\nEpisode {e+1}/{episodes}")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Max Ante: {avg_ante:.2f}")
            print(f"  Play Agent Epsilon: {play_agent.epsilon:.3f}")
            print(f"  Strategy Agent Epsilon: {strategy_agent.epsilon:.3f}")
        
        if (e + 1) % save_interval == 0:
            play_agent.save_model(f"play_agent_ep{e+1}.h5")
            strategy_agent.save_model(f"strategy_agent_ep{e+1}.h5")
            
            print(f"Models saved at episode {e+1}")
    
    return play_agent, strategy_agent



def test_rl_model():
    """
    Run a test of the RL model using the fixed environment to check
    if the shop transition works correctly
    """
    env = BalatroEnv()
    
    # Initialize agents
    play_state_size = len(env._get_play_state())
    play_agent = PlayingAgent(state_size=play_state_size, action_size=env._define_play_action_space())
    
    strategy_state_size = len(env._get_strategy_state())
    strategy_agent = StrategyAgent(state_size=strategy_state_size, action_size=env._define_strategy_action_space())
    
    # Load pre-trained models if they exist
    try:
        play_agent.load_model("play_agent_latest.h5")
        print("Loaded pre-trained play agent model")
    except:
        print("No pre-trained play agent model found, using new model")
    
    try:
        strategy_agent.load_model("strategy_agent_latest.h5")
        print("Loaded pre-trained strategy agent model")
    except:
        print("No pre-trained strategy agent model found, using new model")
    
    # Test parameters
    episodes = 5
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_ante = 1
        
        # Add initial jokers for testing
        from JokerCreation import create_joker
        for joker_name in ["Bootstraps", "Socks and Buskin"]:
            joker = create_joker(joker_name)
            if joker:
                env.game_manager.game.inventory.add_joker(joker)
                print(f"Added {joker_name} to inventory")
        
        # Give some initial money
        env.game_manager.game.inventory.money = 100
        
        done = False
        in_shop_phase = False
        
        print(f"\n===== STARTING TEST EPISODE {e+1} =====")
        print(f"Current Ante: {env.game_manager.game.current_ante}, Blind: {env.game_manager.game.current_blind}")
        print(f"Money: ${env.game_manager.game.inventory.money}")
        
        while not done and steps < 200:
            steps += 1
            
            # Check if we're in shop phase based on ante beaten status
            if env.game_manager.current_ante_beaten and not in_shop_phase and not done:
                print(f"\n***** ANTE {env.game_manager.game.current_ante} BEATEN! Moving to shop *****")
                in_shop_phase = True
            
            # SHOP PHASE
            if in_shop_phase:
                print(f"\n===== SHOP PHASE (Step {steps}) =====")
                
                # Update shop for current ante
                env.update_shop()
                
                # Display shop items
                print("\n=== SHOP ITEMS ===")
                for i in range(len(env.current_shop.items)):
                    if env.current_shop.items[i] is not None:
                        item = env.current_shop.items[i]
                        item_name = "Unknown Item"
                        
                        if hasattr(item, 'item_type'):
                            if item.item_type == ShopItemType.JOKER and hasattr(item.item, 'name'):
                                item_name = item.item.name
                            elif item.item_type in [ShopItemType.TAROT, ShopItemType.PLANET] and hasattr(item.item, 'name'):
                                item_name = item.item.name
                            elif item.item_type == ShopItemType.BOOSTER and hasattr(item, 'item'):
                                item_name = str(item.item)
                                
                        price = env.current_shop.get_item_price(i)
                        print(f"{i}: {item_name} - ${price}")
                
                # Get strategy action
                strategy_state = env._get_strategy_state()
                valid_actions = env.get_valid_strategy_actions()
                strategy_action = strategy_agent.act(strategy_state, valid_actions)
                
                # Take action
                next_strategy_state, strategy_reward, strategy_done, strategy_info = env.step_strategy(strategy_action)
                
                # Display result
                action_name = "Unknown"
                if strategy_action < 4:
                    action_name = f"Buy Item {strategy_action}"
                elif strategy_action < 9:
                    action_name = f"Sell Joker {strategy_action - 4}"
                elif strategy_action < 15:
                    tarot_idx = (strategy_action - 9) // 3
                    selection_type = (strategy_action - 9) % 3
                    action_name = f"Use Tarot {tarot_idx} with Selection {selection_type}"
                elif strategy_action == 15:
                    action_name = "Advance to Next Ante"
                
                print(f"Strategy Action: {action_name}")
                print(f"Strategy Reward: {strategy_reward}")
                print(f"Info: {strategy_info.get('message', 'No message')}")
                
                total_reward += strategy_reward
                done = strategy_done
                
                # Exit shop phase if advancing to next ante or game is done
                if strategy_action == 15 or done:
                    in_shop_phase = False
                    print(f"Exiting shop phase, advancing to Ante {env.game_manager.game.current_ante}")
                    
                    # Make sure we have a hand
                    if not env.game_manager.current_hand:
                        env.game_manager.deal_new_hand()
            
            # REGULAR GAMEPLAY PHASE
            else:
                # Print current hand info
                print(f"\n=== PLAY PHASE (Step {steps}) ===")
                print(f"Hand {env.game_manager.hands_played + 1}/{env.game_manager.max_hands_per_round}")
                print(f"Discards Used: {env.game_manager.discards_used}/{env.game_manager.max_discards_per_round}")
                print(f"Score: {env.game_manager.current_score}/{env.game_manager.game.current_blind}")
                
                # Print current hand
                print("\n=== CURRENT HAND ===")
                for i, card in enumerate(env.game_manager.current_hand):
                    status = ""
                    if card.scored:
                        status += " (scoring)"
                    if hasattr(card, 'debuffed') and card.debuffed:
                        status += " (DEBUFFED)"
                    print(f"{i}: {card}{status}")
                
                # Get valid actions
                valid_actions = env.get_valid_play_actions()
                
                # Choose action
                play_action = play_agent.act(state, valid_actions)
                
                # Decode action
                is_discard = env.is_discard_action(play_action)
                indices = env._convert_action_to_card_indices(play_action)
                
                action_type = "Discard" if is_discard else "Play"
                print(f"\nChosen Action: {action_type} cards at indices {indices}")
                
                # Take action
                next_state, reward, done, info = env.step_play(play_action)
                
                print(f"Reward: {reward}")
                print(f"Info: {info.get('message', 'No message')}")
                
                state = next_state
                total_reward += reward
                
                # Track maximum ante
                max_ante = max(max_ante, env.game_manager.game.current_ante)
        
        print(f"\n===== TEST EPISODE {e+1} COMPLETE =====")
        print(f"Total Steps: {steps}")
        print(f"Max Ante Reached: {max_ante}")
        print(f"Total Reward: {total_reward}")
        print(f"Game Over: {done}")
    
    return env, play_agent, strategy_agent

if __name__ == "__main__":
    # Option 1: Run the test to check for shop transition bugs
    # test_rl_model()
    
    # Option 2: Train using the separate agents approach
    play_agent, strategy_agent = train_with_separate_agents()
    
    # Option 3: Train from scratch with curriculum learning  
    # play_agent, strategy_agent = train_with_curriculum()
    
    # Evaluate agents after training
    results = evaluate_agents(play_agent, strategy_agent)
    print("Evaluation Results:")
    print(f"  Win Rate: {results['win_rate']:.2f}%")
    print(f"  Average Max Ante: {results['average_score']:.2f}")
    print(f"  Average Hands Played: {sum(results['hands_played'])/len(results['hands_played']):.2f}")


"""
if __name__ == "__main__":
    # Initialize the environment to get state and action sizes
    env = BalatroEnv()
    
    # Define state sizes for both agents
    play_state_size = len(env._get_play_state())
    strategy_state_size = len(env._get_strategy_state())
    
    # Define action sizes for both agents
    play_action_size = env._define_play_action_space() 
    strategy_action_size = env._define_strategy_action_space()
    
    print(f"Play agent: state_size={play_state_size}, action_size={play_action_size}")
    print(f"Strategy agent: state_size={strategy_state_size}, action_size={strategy_action_size}")
    
    # Option 1: Run the test to check for shop transition bugs
    # test_rl_model()
    
    # Option 2: Train from scratch with curriculum learning
    play_agent, strategy_agent = train_with_curriculum()
    
    # Option 3: Load existing models and continue training
    # play_agent = PlayingAgent(state_size=play_state_size, action_size=play_action_size)
    # play_agent.load_model("play_agent_latest.h5")
    # strategy_agent = StrategyAgent(state_size=strategy_state_size, action_size=strategy_action_size)
    # strategy_agent.load_model("strategy_agent_latest.h5")
    # play_agent, strategy_agent = train_agents(episodes=5000, 
    #                                          play_agent=play_agent,
    #                                          strategy_agent=strategy_agent)
    
    # Evaluate agents after training
    results = evaluate_agents(play_agent, strategy_agent)
    print("Evaluation Results:")
    print(f"  Win Rate: {results['win_rate']:.2f}%")
    print(f"  Average Max Ante: {results['average_score']:.2f}")
    print(f"  Average Hands Played: {sum(results['hands_played'])/len(results['hands_played']):.2f}")
    """