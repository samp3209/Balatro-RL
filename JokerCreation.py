from Joker import Joker, JokerEffect
from enum import Enum, auto
from typing import List, Optional
from Enums import *

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

class GreenJoker(Joker):
    def __init__(self):
        super().__init__("Green Joker", price=4, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        # +1 mult per hand played, -1 mult per discard
        effect.mult = round_info.get('hands_played', 0) - discards
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

class CleverJoker(Joker):
    def __init__(self):
        super().__init__("Clever", price=4, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        # +80 Chips if played hand contains a two pair
        if round_info.get('hand_type') == 'two_pair':
            effect.chips = 80
        return effect

class WalkieTalkieJoker(Joker):
    def __init__(self):
        super().__init__("Walkie Talkie", price=4, sell_value=2)
        
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        effect = JokerEffect()
        # Each played 10 or 4 gives +10 chips and +4 mult when scored
        ten_four_count = sum(1 for card in hand if card.rank in [Rank.TEN, Rank.FOUR])
        effect.chips = 10 * ten_four_count
        effect.mult = 4 * ten_four_count
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