from enum import Enum, auto
from typing import List, Optional


class JokerEffect:
    def __init__(self):
        self.mult_add = 0
        self.mult_mult = 1
        self.chips = 0
        self.money = 0
        self.triggered_effects = []


class Joker(object):
    def __init__(self, name: str, price: int, sell_value: int):
        self.name = name
        self.rarity = 'base'
        self.price = price
        self.sell_value = sell_value
        self.mult_effect = 0 
        self.chips_effect = 0
        self.played_hand_effect = 0
        self.left_most = False
        self.boss_blind_defeated = 0
        self.retrigger = False
    
    def calculate_effect(self, hand: List, discards: int, deck: List, round_info: dict) -> JokerEffect:
        """
        Base method to be overridden by specific jokers
        Calculate the joker's effect based on the current game state
        """
        return JokerEffect()

