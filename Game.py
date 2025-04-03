import random
from typing import List, Optional, Tuple, Dict, Set
from Enums import *
from Card import Card
from Hand import hand
from Inventory import Inventory

class Game:
    def __init__(self, seed=None):
        # Initialize with a seed for deterministic randomness
        if seed is not None:
            random.seed(seed)
            
        # Game state
        self.inventory = Inventory()
        self.current_ante = 1
        self.current_blind = 300  # Starting small blind
        self.stake_multiplier = 1
        self.hands_played = 0
        self.hands_discarded = 0
        self.face_cards_discarded_count = 0
        
        # Initialize standard deck
        self.initialize_deck()
        
    def initialize_deck(self):
        """Initialize a standard 52-card deck"""
        self.inventory.initialize_standard_deck()
        self.inventory.shuffle_deck()
        
    def deal_hand(self, count=8) -> List[Card]:
        """Deal a specified number of cards from the deck to the hand"""
        hand = []
        for _ in range(min(count, len(self.inventory.deck))):
            card = self.inventory.deck.pop(0)
            card.in_deck = False
            card.in_hand = True
            hand.append(card)
            
        return hand
    
    def play_cards(self, cards: List[Card]) -> bool:
        """
        Mark specified cards as played
        Returns True if successful
        """
        if not cards:
            return False
            
        for card in cards:
            if card.in_hand and not card.played:
                card.played = True
                card.in_hand = False
                card.played_this_ante = True
                
        return True
    
    def discard_cards(self, cards: List[Card]) -> bool:
        """
        Mark specified cards as discarded
        Returns True if successful
        """
        if not cards:
            return False
            
        for card in cards:
            if card.in_hand and not card.discarded:
                card.discarded = True
                card.in_hand = False
                
                # Track face cards discarded for certain Joker effects
                if card.face:
                    self.face_cards_discarded_count += 1
                
        self.hands_discarded += 1
        return True
    
    def evaluate_played_hand(self, played_cards: List[Card]) -> Tuple[HandType, Dict[str, bool]]:
        """
        Evaluate the played cards to determine the best possible hand type
        Also identify all possible hand types contained within the played cards
        
        Returns:
            Tuple of (best_hand_type, contained_hand_types)
            where contained_hand_types is a dict of {hand_type_name: exists_bool}
        """
        if not played_cards or len(played_cards) < 5:
            return (HandType.HIGH_CARD, {"high_card": True})
        
        # Sort cards by rank for easier evaluation
        sorted_cards = sorted(played_cards, key=lambda card: card.rank.value)
        
        # Track all possible hand types contained in this set of cards
        contained_hands = {
            "high_card": True,  # Always true if we have at least one card
            "pair": False,
            "two_pair": False,
            "three_of_kind": False,
            "straight": False,
            "flush": False,
            "full_house": False,
            "four_of_kind": False,
            "straight_flush": False
        }
        
        # Count occurrences of each rank and suit
        rank_counts = {}
        suit_counts = {}
        
        for card in played_cards:
            # Handle wild cards (count for all suits)
            if card.enhancement == CardEnhancement.WILD:
                for suit in Suit:
                    if suit != Suit.WILD:  # Skip the WILD enum itself
                        suit_counts[suit] = suit_counts.get(suit, 0) + 1
            else:
                suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
                
            rank_counts[card.rank.value] = rank_counts.get(card.rank.value, 0) + 1
        
        # Check for pairs, three of a kind, four of a kind
        pairs = []
        three_of_kinds = []
        four_of_kinds = []
        
        for rank, count in rank_counts.items():
            if count == 2:
                pairs.append(rank)
            elif count == 3:
                three_of_kinds.append(rank)
            elif count == 4:
                four_of_kinds.append(rank)
        
        # Check for flush
        flush = any(count >= 5 for count in suit_counts.values())
        if flush:
            contained_hands["flush"] = True
        
        # Check for straight
        straight = False
        # Create a set of unique ranks
        unique_ranks = set(card.rank.value for card in played_cards)
        
        # Check for A-2-3-4-5 straight (Ace = 1)
        if {14, 2, 3, 4, 5}.issubset(unique_ranks):
            straight = True
        
        # Check for normal straight
        for i in range(2, 11):
            if all(r in unique_ranks for r in range(i, i+5)):
                straight = True
                break
                
        if straight:
            contained_hands["straight"] = True
        
        # Determine best hand type and fill contained hands
        if len(pairs) >= 1:
            contained_hands["pair"] = True
            
        if len(pairs) >= 2:
            contained_hands["two_pair"] = True
            
        if len(three_of_kinds) >= 1:
            contained_hands["three_of_kind"] = True
            
        if len(four_of_kinds) >= 1:
            contained_hands["four_of_kind"] = True
            
        if (len(three_of_kinds) >= 1 and len(pairs) >= 1) or len(three_of_kinds) >= 2:
            contained_hands["full_house"] = True
            
        # Determine the best hand
        best_hand = HandType.HIGH_CARD
        
        if contained_hands["straight_flush"]:
            best_hand = HandType.STRAIGHT_FLUSH
        elif contained_hands["four_of_kind"]:
            best_hand = HandType.FOUR_OF_A_KIND
        elif contained_hands["full_house"]:
            best_hand = HandType.FULL_HOUSE
        elif contained_hands["flush"]:
            best_hand = HandType.STRAIGHT
        elif contained_hands["straight"]:
            best_hand = HandType.STRAIGHT
        elif contained_hands["three_of_kind"]:
            best_hand = HandType.THREE_OF_A_KIND
        elif contained_hands["two_pair"]:
            best_hand = HandType.TWO_PAIR
        elif contained_hands["pair"]:
            best_hand = HandType.PAIR
            
        # Mark scoring cards
        self._mark_scoring_cards(played_cards, best_hand)
            
        return (best_hand, contained_hands)
    
    def _mark_scoring_cards(self, played_cards: List[Card], hand_type: HandType):
        """Mark cards that contribute to the scoring hand"""
        if hand_type == HandType.HIGH_CARD:
            # Find highest card and mark it
            highest_card = max(played_cards, key=lambda card: card.rank.value)
            highest_card.scored = True
            return
            
        # For other hand types, we need more complex logic
        # This is a simplified version, real implementation would be more detailed
        sorted_cards = sorted(played_cards, key=lambda card: card.rank.value, reverse=True)
        
        if hand_type == HandType.PAIR:
            # Find the highest pair
            rank_counts = {}
            for card in played_cards:
                rank_counts[card.rank.value] = rank_counts.get(card.rank.value, 0) + 1
                
            pair_rank = max([rank for rank, count in rank_counts.items() if count >= 2])
            
            # Mark the pair cards
            for card in played_cards:
                if card.rank.value == pair_rank:
                    card.scored = True
        
        # Similar logic for other hand types...
        # This would be expanded for all hand types in a complete implementation
        
    def calculate_hand_score(self, hand_type: HandType) -> Tuple[int, int]:
        """
        Calculate the score for a hand type, applying inventory bonuses
        Returns (multiplier, chips)
        """
        return self.inventory.calculate_hand_value(hand_type, {
            'stake_multiplier': self.stake_multiplier
        })
        
    def apply_joker_effects(self, played_cards: List[Card], hand_type: HandType, contained_hands: Dict[str, bool]) -> Tuple[int, int, int]:
        """
        Apply all joker effects based on the current state
        
        Returns:
            Tuple of (total_mult, base_chips, money_gained)
            The caller should multiply total_mult by base_chips to get the final score
        """
        # First, get base values for the hand
        base_mult, base_chips = self.inventory.calculate_hand_value(hand_type, {
            'stake_multiplier': self.stake_multiplier
        })
        
        # Start with base values
        total_mult = base_mult
        money_gained = 0
        
        print(f"Base values: {base_mult} mult, {base_chips} chips")
        
        # Game state information for jokers
        round_info = {
            'hand_type': hand_type.name.lower(),
            'contained_hands': contained_hands,
            'hands_played': self.hands_played,
            'inventory': self.inventory,
            'max_discards': 4,  # Default max discards
            'face_cards_discarded_count': self.face_cards_discarded_count
        }
        
        # Apply each joker's effect
        for joker in self.inventory.jokers:
            print(f"Applying {joker.name} effect...")
            effect = joker.calculate_effect(
                played_cards, 
                self.hands_discarded, 
                self.inventory.deck, 
                round_info
            )
            
            # Apply effect
            old_mult = total_mult
            
            # First apply additive effects
            total_mult += effect.mult_add
            base_chips += effect.chips  # Add chips directly to the base amount
            
            # Then apply multiplicative effects
            total_mult *= effect.mult_mult
            
            # Add money
            money_gained += effect.money
            
            print(f"  • {joker.name}: +{effect.mult_add} mult, x{effect.mult_mult} mult, +{effect.chips} chips, +${effect.money}")
            print(f"  • Result: {old_mult} → {total_mult} mult, {base_chips} chips")
        
        # Return the total multiplier and base chips
        # The final score will be calculated as total_mult * base_chips by the caller
        return (total_mult, base_chips, money_gained)
        
    def reset_for_new_round(self):
        """Reset game state for a new round"""
        # Reset cards in deck
        for card in self.inventory.deck:
            card.reset_state()
            
        # Reset counters
        self.hands_played = 0
        self.hands_discarded = 0
        self.face_cards_discarded_count = 0
        
        # Reset jokers that need per-round reset
        for joker in self.inventory.jokers:
            if hasattr(joker, 'reset'):
                joker.reset()
                
        # Reshuffle deck
        self.inventory.shuffle_deck()