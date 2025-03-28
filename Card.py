from Enums import *
class Card:
    def __init__(self, suit: Suit, rank: Rank):
        # Suit information
        self.suit = suit
        
        # Rank information
        self.rank = rank
        
        # Enhancements
        self.enhancement = CardEnhancement.NONE
        
        # Card state
        self.face = rank.value >= Rank.JACK.value and rank.value < Rank.ACE.value
        self.played_this_ante = False
        self.in_deck = True
        self.in_hand = False
        self.played = False
        self.scored = False
        self.discarded = False
        self.retrigger = False
    
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
    
    def __repr__(self):
        """
        String representation of the card
        """
        return f"{self.suit.name} {self.rank.name}"

        

