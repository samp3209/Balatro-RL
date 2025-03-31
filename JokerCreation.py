from Joker import *
from enum import Enum, auto
from typing import List, Optional
from Enums import *
import random
from Inventory import *

def create_joker(joker_name: str) -> Optional[Joker]:
    joker_classes = {
        "Green Joker": GreenJoker,
        "Mr. Bones": MrBonesJoker,
        "Delayed Gratification": DelayedGratificationJoker,
        "Clever": CleverJoker,
        "Walkie Talkie": WalkieTalkieJoker,
        "Rocket": RocketJoker
    }
    
    return joker_classes.get(joker_name, lambda: None)()

class GreenJoker(Joker): #NEED to store global played vs discard functionality not just round
    def __init__(self):
        super().__init__("Green Joker", price=4, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        # +1 mult per hand played, -1 mult per discard
        effect.mult_add = round_info.get('hands_played', 0) - discards
        return effect
    

class MrBonesJoker(Joker):
    def __init__(self):
        super().__init__("Mr. Bones", price=6, sell_value=3)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        # Logic for preventing death if chips are at least 25% of required
        # Note: This would likely need to be handled in the main game logic
        return effect

class DelayedGratificationJoker(Joker):
    def __init__(self):
        super().__init__("Delayed Gratification", price=4, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        # Earn $2 per discard if no discards used by end of round
        if discards == 0:
            effect.money = 2 * round_info.get('max_discards', 0)
        return effect

class CleverJoker(Joker): #NEED to handle flush with a two pair in it
    def __init__(self):
        super().__init__("Clever", price=4, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        # +80 Chips if played hand contains a two pair
        if round_info.get('hand_type') == 'two_pair' or round_info.get('hand_type') == 'full_house':
            effect.chips = 80
        return effect

class MadJoker(Joker): #NEED to handle flush with a two pair in it
    def __init__(self):
        super().__init__("Mad", price=3, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info) -> JokerEffect:
        effect = JokerEffect()
        # +10 mult if played hand contains a two pair
        if round_info.get('hand_type') == 'two_pair' or round_info.get('hand_type') == 'full_house':
            effect.mult_add = 10
        return effect
    
class WilyJoker(Joker): #NEED to handle flush with a three of a kind in it
    def __init__(self):
        super().__init__("Wily", price=3, sell_value=1)

    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        # +100 Chips if played hand contains a three of a kind
        if round_info.get('hand_type') == 'three_of_kind' or round_info.get('hand_type') == 'four_of_kind' or round_info.get('hand_type') == 'full_house':
            effect.chips = 100
        return effect
    

class CraftyJoker(Joker):
    def __init__(self):
        super().__init__("Crafty", price=3, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        if round_info.get('hand_type') == 'flush':
            effect.chips = 80
        return effect

class MisprintJoker(Joker):
    def __init__(self):
        super().__init__("Misprint", price=4, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        effect.mult_add = random.randint(0, 23)
        return effect
    
class WrathfulJoker(Joker):
    def __init__(self):
        super().__init__("Wrathful", price=4, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        spade_count = sum(1 for card in hand if card.suit == Suit.SPADES and card.scored == True)
        effect.mult_add = 3 * spade_count
        return effect
    
class SmileyJoker(Joker):
    def __init__(self):
        super().__init__("Smiley", price=5, sell_value=2)

    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        face_count = sum(1 for card in hand if card.rank in [Rank.JACK, Rank.QUEEN, Rank.King] and card.scored == True)
        effect.mult_add = 5 * face_count
        return effect

class EvenSteven(Joker):
    def __init__(self):
        super().__init__("EvenSteven", price = 4, sell_value=2)
    
    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        even_count = sum(1 for card in hand if card.rank in [Rank.TEN, Rank.EIGHT, Rank.SIX, Rank.FOUR, Rank.TWO] and card.scored == True)
        effect.mult_add = even_count * 4
        return effect

class BlueJoker(Joker):
    def __init__(self):
        super().__init__("Blue", price =4, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        deck_count = sum(1 for card in deck)
        effect.chips = deck_count *2
        return effect

class WalkieTalkieJoker(Joker):
    def __init__(self):
        super().__init__("Walkie Talkie", price=4, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        # Each played 10 or 4 gives +10 chips and +4 mult when scored
        ten_four_count = sum(1 for card in hand if card.rank in [Rank.TEN, Rank.FOUR] and card.scored == True)
        effect.chips = 10 * ten_four_count
        effect.mult_add = 4 * ten_four_count
        return effect

class RocketJoker(Joker):
    def __init__(self):
        super().__init__("Rocket", price=6, sell_value=3)
        self.boss_blind_defeated = 0
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        # Earn $1 at end of round, payout increases by $2 when boss blind is defeated
        effect.money = 1 + (2 * self.boss_blind_defeated)
        return effect
    
class RedCardJoker(Joker):
    def __init__(self):
        super().__init__("Red Card", price=4, sell_value=2)
    
    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        skip_count = Inventory.booster_skip
        effect.mult_add  = 4 * skip_count
        return effect
    
class BannerJoker(Joker):
    def __init__(self):
        super().__init__("Banner", price=4, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        effect.chips = 30*discards
        return effect
    
class TheDuoJoker(Joker): # NEED functionality to check if pair is in flush 
    def __init__(self):
        super().__init__("The Duo", price=7, sell_value=3)
    
    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        if round_info.get("hand_type") in ["pair", "two_pair", "full_house", "three_of_kind", "four_of_kind"]:
            effect.mult_mult = 2
        return effect

class GluttonousJoker(Joker):
    def __init__(self):
        super().__init__("Gluttonous", price=4, sell_value=2)
    
    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        club_count = sum(1 for card in hand if card.suit == Suit.CLUBS and card.scored == True)
        effect.mult_add = club_count * 3
        return effect

class FortuneTellerJoker(Joker):
    def __init__(self):
        super().__init__("Fortune Teller", price=4, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        effect.mult_add = Inventory.tarot_used
        return effect

class BusinessCardJoker(Joker):
    def __init__(self):
        super().__init__("BusinessCard", price=4, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info):
        rng = random.randint(0,1)
        effect = JokerEffect()
        if rng == 1:
            effect.money = 2
        
        return effect
    
class BaseballJoker(Joker):
    def __init__(self):
        super().__init__("BaseballJoker", price=6, sell_value=3)

    def calculate_effect(self, hand, discards, deck, round_info):
        effect = JokerEffect()
        effect.mult_mult = 1.5 * Inventory.uncommon_joker_count
        return effect
    
