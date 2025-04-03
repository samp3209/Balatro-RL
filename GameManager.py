from typing import List, Dict, Tuple, Optional
from Enums import *
from Card import Card
from Inventory import Inventory
from Game import Game
from HandEvaluator import HandEvaluator

class GameManager:
    """
    High-level manager for the game that coordinates game flow and provides
    an interface for playing the game.
    """
    
    def __init__(self, seed=None):
        """Initialize the game manager with an optional seed for reproducibility"""
        self.game = Game(seed)
        self.hand_evaluator = HandEvaluator()
        
        # Current game state
        self.current_hand = []
        self.played_cards = []
        self.discarded_cards = []
        self.hand_result = None
        self.contained_hand_types = {}
        self.scoring_cards = []
        
        # Counters and limits
        self.max_hand_size = 8
        self.max_discards = 4
        self.discards_used = 0
        self.hands_played = 0
        self.current_score = 0
        self.current_ante_beaten = False
        
    def start_new_game(self):
        """Start a new game with a fresh deck and initial ante"""
        self.game.initialize_deck()
        self.deal_new_hand()
        
    def deal_new_hand(self):
        """Deal a new hand of cards from the deck"""
        # Clear any existing cards
        self.current_hand = []
        self.played_cards = []
        self.discarded_cards = []
        
        # Deal new cards
        self.current_hand = self.game.deal_hand(self.max_hand_size)
        
        # Reset counters
        self.discards_used = 0
        self.hands_played = 0
        self.current_score = 0
        self.current_ante_beaten = False
        
        # Reset hand result
        self.hand_result = None
        self.contained_hand_types = {}
        self.scoring_cards = []
        
    def play_cards(self, card_indices: List[int]) -> Tuple[bool, str]:
        """
        Play cards from the current hand
        
        Args:
            card_indices: List of indices of cards to play from the current hand
            
        Returns:
            Tuple of (success, message)
        """
        if not card_indices:
            return (False, "No cards selected to play")
            
        if len(self.played_cards) > 0:
            return (False, "Cards have already been played this hand")
            
        if self.hands_played >= 1:
            return (False, "Already played maximum number of hands")
            
        # Get the selected cards
        cards_to_play = []
        for idx in sorted(card_indices, reverse=True):
            if 0 <= idx < len(self.current_hand):
                cards_to_play.append(self.current_hand.pop(idx))
                
        if not cards_to_play:
            return (False, "No valid cards selected")
            
        # Add to played cards
        self.played_cards.extend(cards_to_play)
        
        # Mark cards as played in the game
        self.game.play_cards(cards_to_play)
        
        # Evaluate the hand
        hand_type, contained_hands, scoring_cards = self.hand_evaluator.evaluate_hand(self.played_cards)
        
        # Mark scoring cards
        self.hand_evaluator.mark_scoring_cards(self.played_cards, scoring_cards)
        
        # Store results
        self.hand_result = hand_type
        self.contained_hand_types = contained_hands
        self.scoring_cards = scoring_cards
        
        # Calculate score
        mult, chips, money = self.game.apply_joker_effects(
            self.played_cards, hand_type, contained_hands
        )
        
        # Update score and counters
        self.current_score += chips
        self.game.inventory.money += money
        self.hands_played += 1
        
        # Check if ante is beaten
        if self.current_score >= self.game.current_blind:
            self.current_ante_beaten = True
            
        return (True, f"Played {hand_type.name} for {chips} chips (x{mult} mult)")
    
    def discard_cards(self, card_indices: List[int]) -> Tuple[bool, str]:
        """
        Discard cards from the current hand and draw replacements
        
        Args:
            card_indices: List of indices of cards to discard from the current hand
            
        Returns:
            Tuple of (success, message)
        """
        if not card_indices:
            return (False, "No cards selected to discard")
            
        if self.discards_used >= self.max_discards:
            return (False, "Maximum discards already used")
            
        # Get the selected cards
        cards_to_discard = []
        for idx in sorted(card_indices, reverse=True):
            if 0 <= idx < len(self.current_hand):
                cards_to_discard.append(self.current_hand.pop(idx))
                
        if not cards_to_discard:
            return (False, "No valid cards selected")
            
        # Add to discarded cards
        self.discarded_cards.extend(cards_to_discard)
        
        # Mark cards as discarded in the game
        self.game.discard_cards(cards_to_discard)
        
        # Draw replacements
        replacement_cards = self.game.deal_hand(len(cards_to_discard))
        self.current_hand.extend(replacement_cards)
        
        # Update counter
        self.discards_used += 1
        
        return (True, f"Discarded {len(cards_to_discard)} cards and drew replacements")
    
    def get_best_hand_from_current(self) -> Optional[Tuple[HandType, List[Card]]]:
        """
        Evaluate the current hand to determine the best possible hand
        
        Returns:
            Tuple of (best_hand_type, cards_in_best_hand) or None if hand is empty
        """
        if not self.current_hand:
            return None
            
        # Evaluate all possible hands from current cards
        hand_type, _, scoring_cards = self.hand_evaluator.analyze_hand(self.current_hand)
        
        return (hand_type, scoring_cards)
    
    def get_recommended_play(self) -> Tuple[List[int], str]:
        """
        Get a recommended play based on the current hand
        
        Returns:
            Tuple of (recommended_card_indices, explanation)
        """
        # This is a very basic recommendation that just plays the best hand
        # A real implementation would be much more sophisticated
        best_hand = self.get_best_hand_from_current()
        
        if not best_hand:
            return ([], "No cards to play")
            
        hand_type, cards = best_hand
        
        # Find indices of the cards in the current hand
        indices = []
        for card in cards:
            for i, hand_card in enumerate(self.current_hand):
                if (hand_card.rank.value == card.rank.value and 
                    hand_card.suit == card.suit and
                    i not in indices):
                    indices.append(i)
                    break
                    
        explanation = f"Play {hand_type.name} for the best chance of winning"
        
        return (indices, explanation)
    
    def get_game_state(self) -> Dict:
        """
        Get the current game state as a dictionary
        """
        return {
            'current_ante': self.game.current_ante,
            'current_blind': self.game.current_blind,
            'hands_played': self.hands_played,
            'discards_used': self.discards_used,
            'max_discards': self.max_discards,
            'current_score': self.current_score,
            'ante_beaten': self.current_ante_beaten,
            'money': self.game.inventory.money,
            'hand_size': len(self.current_hand),
            'played_cards_count': len(self.played_cards),
            'discarded_cards_count': len(self.discarded_cards),
            'deck_size': len(self.game.inventory.deck),
            'joker_count': len(self.game.inventory.jokers),
            'consumable_count': len(self.game.inventory.consumables)
        }
        
    def next_ante(self):
        """
        Move to the next ante if the current one is beaten
        Returns True if successful
        """
        if not self.current_ante_beaten:
            return False
            
        # Move to next ante
        self.game.current_ante += 1
        
        # Set the new blind based on ante progression
        blind_progression = {
            1: 300,   # Small blind
            2: 450,   # Medium blind 
            3: 600,  # Boss blind
            4: 800,  # Small blind for ante 2
            5: 1200,  # Medium blind for ante 2
            6: 1600,  # Boss blind for ante 2
            7: 2000, # Small blind for ante 3
            8: 3000,  # Medium blind for ante 3
            9: 4000,
            10: 5000,
            11: 7500,
            12: 10000,
            13: 11000,
            14: 17500,
            15: 22000,
            16: 20000,
            17: 30000,
            18: 40000,
            19: 35000,
            20: 52500,
            21: 70000,
            22: 50000,
            23: 75000,
            24: 100000,
        }
        
        self.game.current_blind = blind_progression.get(self.game.current_ante, 300)
        
        # Reset for the new ante
        self.game.reset_for_new_round()
        self.deal_new_hand()
        self.current_ante_beaten = False
        
        return True
    
    def use_tarot(self, tarot_index: int, selected_card_indices: List[int]) -> Tuple[bool, str]:
        """
        Use a tarot card with selected cards
        
        Args:
            tarot_index: Index of the tarot card in consumables
            selected_card_indices: Indices of selected cards from the current hand
            
        Returns:
            Tuple of (success, message)
        """
        tarot_indices = self.game.inventory.get_consumable_tarot_indices()
        
        if tarot_index not in tarot_indices:
            return (False, "Invalid tarot card selected")
            
        # Get the selected cards
        selected_cards = []
        for idx in selected_card_indices:
            if 0 <= idx < len(self.current_hand):
                selected_cards.append(self.current_hand[idx])
                
        # Create game state dict for tarot effect
        game_state = {
            'money': self.game.inventory.money,
            'last_tarot_used': self.game.inventory.last_tarot,
            'last_planet_used': self.game.inventory.last_planet
        }
        
        # Apply tarot effect
        effect = self.game.inventory.use_tarot(tarot_index, selected_cards, game_state)
        
        if not effect:
            return (False, "Failed to apply tarot effect")
            
        # Process the effect results
        message = effect.get('message', 'Tarot card used successfully')
        
        # Handle possible modifications to cards, money, etc.
        if 'money_gained' in effect and effect['money_gained'] > 0:
            self.game.inventory.money += effect['money_gained']
            message += f" Gained ${effect['money_gained']}."
            
        # Handle card enhancements, deletions, etc.
        # This would be more complex in a complete implementation
        
        return (True, message)
    
    def use_planet(self, planet_index: int) -> Tuple[bool, str]:
        """
        Use a planet card with the current hand type
        
        Args:
            planet_index: Index of the planet card in consumables
            
        Returns:
            Tuple of (success, message)
        """
        planet_indices = self.game.inventory.get_consumable_planet_indices()
        
        if planet_index not in planet_indices:
            return (False, "Invalid planet card selected")
            
        if not self.hand_result:
            return (False, "No hand has been played yet")
            
        # Create game state dict for planet effect
        game_state = {
            'money': self.game.inventory.money,
            'stake_multiplier': self.game.stake_multiplier
        }
        
        # Apply planet effect
        effect = self.game.inventory.use_planet(planet_index, self.hand_result, game_state)
        
        if not effect:
            return (False, "Failed to apply planet effect")
            
        # Process the effect results
        message = effect.get('message', 'Planet card used successfully')
        
        # Update game state based on effect
        if 'mult_bonus' in effect:
            # Recalculate score with the new multiplier
            self.game.stake_multiplier += effect['mult_bonus']
            
        if 'chip_bonus' in effect:
            self.current_score += effect['chip_bonus']
            
        # Check if ante is beaten after applying planet effect
        if self.current_score >= self.game.current_blind:
            self.current_ante_beaten = True
            message += f" Ante beaten! ({self.current_score}/{self.game.current_blind})"
            
        return (True, message)