from Joker import *
from enum import Enum, auto
from typing import List, Optional
from Enums import *
import random
from Tarot import create_random_tarot



def create_joker(joker_name: str) -> Optional[Joker]:
    joker_classes = {
        "Green Joker": GreenJoker,
        "Mr. Bones": MrBonesJoker,
        "Delayed Gratification": DelayedGratificationJoker,
        "Clever": CleverJoker,
        "Walkie Talkie": WalkieTalkieJoker,
        "Rocket": RocketJoker,
        "Cartomancer": CartomancerJoker,
        "Cloud 9": Cloud9Joker,
        "Mad": MadJoker,
        "Wily": WilyJoker,
        "Smiley": SmileyJoker,
        "Crafty": CraftyJoker,
        "Bootstraps": BootstrapsJoker,
        "Splash": SplashJoker,
        "Misprint": MisprintJoker,
        "Wrathful": WrathfulJoker,
        "Red Card": RedCardJoker,
        "Blue": BlueJoker,
        "Even Steven": EvenSteven,
        "Banner": BannerJoker,
        "Brainstorm": BrainstormJoker,
        "Baseball Card": BaseballJoker,
        "Socks and Buskin": SocksAndBuskinJoker,
        "8 Ball": EightBallJoker,
        "The Duo": TheDuoJoker,
        "Gluttonous": GluttonousJoker,
        "Fortune Teller": FortuneTellerJoker,
        "Faceless": FacelessJoker,
        "Business Card": BusinessCardJoker,
        "Black Board": BlackBoardJoker,
        "Photograph": PhotographJoker,
        "Square": SquareJoker,
    }
    
    return joker_classes.get(joker_name, lambda: None)()

class GreenJoker(Joker): #NEED to store global played vs discard functionality not just round
    def __init__(self):
        super().__init__("Green Joker", price=4, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict, Inventory=None) -> JokerEffect:
        effect = JokerEffect()
        effect.mult_add = max(0, round_info.get('hands_played', 0) - discards)
        return effect
    

class MrBonesJoker(Joker):
    def __init__(self):
        super().__init__("Mr. Bones", price=6, sell_value=3)
        self.activated = False
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        return effect

class DelayedGratificationJoker(Joker):
    def __init__(self):
        super().__init__("Delayed Gratification", price=4, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
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
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):

        effect = JokerEffect()
        if round_info.get('hand_type') == 'two_pair' or round_info.get('hand_type') == 'full_house':
            effect.mult_add = 10
        return effect
    
class WilyJoker(Joker): #NEED to handle flush with a three of a kind in it
    def __init__(self):
        super().__init__("Wily", price=3, sell_value=1)

    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        if round_info.get('hand_type') == 'three_of_kind' or round_info.get('hand_type') == 'four_of_kind' or round_info.get('hand_type') == 'full_house':
            effect.chips = 100
        return effect
    

class CraftyJoker(Joker):
    def __init__(self):
        super().__init__("Crafty", price=3, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        if round_info.get('hand_type') == 'flush':
            effect.chips = 80
        return effect

class MisprintJoker(Joker):
    def __init__(self):
        super().__init__("Misprint", price=4, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        effect.mult_add = random.randint(0, 23)
        return effect

class SquareJoker(Joker):
    def __init__(self):
        super().__init__("Square", price=4, sell_value=2)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        
        if len(hand) == 4:
            effect.chips = 4
            effect.triggered_effects.append("Square activated: +4 chips for playing exactly 4 cards")
            
        return effect
    
class WrathfulJoker(Joker):
    def __init__(self):
        super().__init__("Wrathful", price=4, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        spade_count = sum(1 for card in hand if card.suit == Suit.SPADES and card.scored == True)
        effect.mult_add = 3 * spade_count
        return effect
    
class SmileyJoker(Joker):
    def __init__(self):
        super().__init__("Smiley", price=5, sell_value=2)

    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        face_count = sum(1 for card in hand if card.rank in [Rank.JACK, Rank.QUEEN, Rank.KING] and card.scored == True)
        effect.mult_add = 5 * face_count
        return effect

class EvenSteven(Joker):
    def __init__(self):
        super().__init__("EvenSteven", price = 4, sell_value=2)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        even_count = sum(1 for card in hand if card.rank in [Rank.TEN, Rank.EIGHT, Rank.SIX, Rank.FOUR, Rank.TWO] and card.scored == True)
        effect.mult_add = even_count * 4
        return effect

class BlueJoker(Joker):
    def __init__(self):
        super().__init__("Blue", price =4, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        deck_count = sum(1 for card in deck)
        effect.chips = deck_count *2
        return effect

class WalkieTalkieJoker(Joker):
    def __init__(self):
        super().__init__("Walkie Talkie", price=4, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
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
        effect.money = 1 + (2 * self.boss_blind_defeated)
        return effect
    
class RedCardJoker(Joker):
    def __init__(self):
        super().__init__("Red Card", price=4, sell_value=2)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        skip_count = Inventory.booster_skip
        effect.mult_add  = 4 * skip_count
        return effect
    
class BannerJoker(Joker):
    def __init__(self):
        super().__init__("Banner", price=4, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        effect.chips = 30*discards
        return effect
    
class TheDuoJoker(Joker): # NEED functionality to check if pair is in flush 
    def __init__(self):
        super().__init__("The Duo", price=7, sell_value=3)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        if round_info.get("hand_type") in ["pair", "two_pair", "full_house", "three_of_kind", "four_of_kind"]:
            effect.mult_mult = 2
        return effect

class GluttonousJoker(Joker):
    def __init__(self):
        super().__init__("Gluttonous", price=4, sell_value=2)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        club_count = sum(1 for card in hand if card.suit == Suit.CLUBS and card.scored == True)
        effect.mult_add = club_count * 3
        return effect

class FortuneTellerJoker(Joker):
    def __init__(self):
        super().__init__("Fortune Teller", price=4, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        effect.mult_add = Inventory.tarot_used
        return effect

class BusinessCardJoker(Joker):
    def __init__(self):
        super().__init__("BusinessCard", price=4, sell_value=1)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        rng = random.randint(0,1)
        effect = JokerEffect()
        if rng == 1:
            effect.money = 2
        
        return effect
    
class BaseballJoker(Joker):
    def __init__(self):
        super().__init__("BaseballJoker", price=6, sell_value=3)

    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):
        effect = JokerEffect()
        effect.mult_mult = 1.5 * Inventory.uncommon_joker_count
        return effect
    

class BlackBoardJoker(Joker):
    def __init__(self):
        super().__init__("Black Board", price=5, sell_value=2)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):

        effect = JokerEffect()
        dark_suits = [Suit.SPADES, Suit.CLUBS]
        all_dark = all(card.suit in dark_suits for card in hand if card.in_hand)
        
        if all_dark and len(hand) > 0:
            effect.mult_mult = 3
        return effect

class PhotographJoker(Joker):
    def __init__(self):
        super().__init__("Photograph", price=4, sell_value=2)
        self.first_face_card_found = False
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):

        effect = JokerEffect()
        
        for card in hand:
            if card.face and card.scored and not self.first_face_card_found:
                effect.mult_mult = 2
                self.first_face_card_found = True
                break
                
        return effect
    
    def reset(self):
        """Reset the joker for a new round"""
        self.first_face_card_found = False

class EightBallJoker(Joker):
    def __init__(self):
        super().__init__("8 Ball", price=5, sell_value=2)
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):

        effect = JokerEffect()
        inventory = round_info.get('inventory')
        
        eight_count = sum(1 for card in hand if card.rank == Rank.EIGHT and card.scored)
        
        created_tarots = 0
        for _ in range(eight_count):
            if random.random() < 0.25 and inventory and inventory.get_available_space() > 0:
                tarot = create_random_tarot()
                inventory.add_consumable(tarot)
                created_tarots += 1
        
        if created_tarots > 0:
            effect.message = f"Created {created_tarots} Tarot card(s) from 8s"
            
        return effect

class CartomancerJoker(Joker):
    def __init__(self):
        super().__init__("Cartomancer", price=6, sell_value=3)
        self.blind_selected_this_round = False
    
    def calculate_effect(self, hand, discards, deck, round_info, Inventory=None):

        effect = JokerEffect()
        inventory = round_info.get('inventory')
        blind_selected = round_info.get('blind_selected', False)
        
        if blind_selected and not self.blind_selected_this_round and inventory and inventory.get_available_space() > 0:
            tarot = create_random_tarot()
            inventory.add_consumable(tarot)
            self.blind_selected_this_round = True
            effect.message = "Created a Tarot card from blind selection"
            
        return effect
    
    def reset(self):
        """Reset the joker for a new round"""
        self.blind_selected_this_round = False

class FacelessJoker(Joker):
    def __init__(self):
        super().__init__("Faceless", price=5, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        face_cards_discarded = round_info.get('face_cards_discarded_count', 0)
        if face_cards_discarded >= 3:
            effect.money = 6
        return effect

class SocksAndBuskinJoker(Joker):
    def __init__(self):
        super().__init__("Socks and Buskin", price=5, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        for card in hand:
            if card.face and card.played:
                card.retrigger = True
        return effect

class BrainstormJoker(Joker):
    def __init__(self):
        super().__init__("Brainstorm", price=7, sell_value=3)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        inventory = round_info.get('inventory')
        if inventory and inventory.jokers and len(inventory.jokers) > 0:
            leftmost_joker = inventory.jokers[0]
            if leftmost_joker.name != "Brainstorm":
                copied_effect = leftmost_joker.calculate_effect(hand, discards, deck, round_info)
                effect.mult_add = copied_effect.mult_add
                effect.mult_mult = copied_effect.mult_mult
                effect.chips = copied_effect.chips
                effect.money = copied_effect.money
        return effect

class BootstrapsJoker(Joker):
    def __init__(self):
        super().__init__("Bootstraps", price=6, sell_value=3)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        inventory = round_info.get('inventory')
        if inventory:
            money = inventory.money
            effect.mult_add = 2 * (money // 5)
        return effect

class SplashJoker(Joker):
    def __init__(self):
        super().__init__("Splash", price=4, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        effect.count_all_played = True
        return effect

class Cloud9Joker(Joker):
    def __init__(self):
        super().__init__("Cloud 9", price=6, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        
        nine_count = sum(1 for card in deck if card.rank == Rank.NINE)
        effect.money = nine_count
        
        return effect