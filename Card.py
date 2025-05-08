from typing import Tuple
from Enums import *
class Card:
    def __init__(self, suit: Suit, rank: Rank):
        if not isinstance(suit, Suit):
            raise TypeError(f"Expected Suit enum for suit, got {type(suit)}: {suit}")
        
        if not isinstance(rank, Rank):
            raise TypeError(f"Expected Rank enum for rank, got {type(rank)}: {rank}")
            
        self.suit = suit
        self.rank = rank
        
        self.enhancement = CardEnhancement.NONE
        
        self.face = rank.value >= Rank.JACK.value and rank.value < Rank.ACE.value
        self.played_this_ante = False
        self.in_hand = False
        self.played = False
        self.scored = False
        self.discarded = False
        self.in_deck = True
        self.retrigger = False
        self.debuffed = False
    
    def apply_enhancement(self, enhancement: CardEnhancement):
        """
        Apply an enhancement to the card
        """
        self.enhancement = enhancement
    
    def get_chip_bonus(self) -> int:
        """
        Calculate chip bonus based on enhancement
        """
        if self.enhancement == CardEnhancement.FOIL:
            return 50
        return 0
    
    def get_mult_add(self) -> float:
        """
        Calculate mult bonus based on enhancement
        """
        if self.enhancement == CardEnhancement.HOLO:
            return 10
        return 0

    def get_mult_mult(self) -> float:
        """Calculate mult multiplication bonus based on enhancement"""
        if self.enhancement == CardEnhancement.POLY:
            return 1.5
        return 1.0
    





    def is_wild(self) -> bool:
        """Check if this card is a wild card (counts as all suits)"""
        return self.enhancement == CardEnhancement.WILD
    
    def has_no_rank_or_suit(self) -> bool:
        """Check if this card has no rank or suit (Stone cards)"""
        return self.enhancement == CardEnhancement.STONE
    
    def get_gold_money(self) -> int:
        """Get money bonus from gold card if held in hand"""
        if self.enhancement == CardEnhancement.GOLD and self.in_hand:
            return 3
        return 0
    
    def get_lucky_bonus(self) -> Tuple[int, int]:
        """
        Get random bonus from lucky card
        Returns (mult_bonus, money_bonus)
        """
        import random
        mult_bonus = 0
        money_bonus = 0
        
        if self.enhancement == CardEnhancement.LUCKY:
            if random.random() < 0.2:
                mult_bonus = 20
            
            if random.random() < 0.066:
                money_bonus = 20
        
        return (mult_bonus, money_bonus)
    
    def check_glass_break(self) -> bool:
        """
        Check if glass card breaks after use
        Returns True if card breaks and should be removed
        """
        import random
        if self.enhancement == CardEnhancement.GLASS and self.played:
            return random.random() < 0.25
        return False
    
    def reset_state(self):
        """
        Reset card state for a new round
        """
        self.played_this_ante = False
        self.in_hand = False
        self.played = False
        self.scored = False
        self.discarded = False
        self.in_deck = True
        self.retrigger = False
        self.debuffed = False
    
    def __repr__(self):
        """
        String representation of the card
        """
        return f"{self.suit.name} {self.rank.name}"

        

