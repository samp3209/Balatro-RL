from enum import Enum

class joker_chip_modifiers(Enum):
    base = 0
    foil = 50
    holo = 0
    poly = 0

class joker_multadd_modifiers(Enum):
    base = 0 
    foil = 0 
    holo = 10
    poly = 0

class joker_multmult_modifiers(Enum):
    base = 0 
    foil = 0 
    holo = 0
    poly = 1.5

class card_chipsadd_modifier(Enum):
    bonus = 30
    mult = 0
    wild = 0
    glass = 0 
    steel = 0
    stone = 50
    gold =0 
    lucky = 0
    gold_seal = 0
    red_seal = 0
    blue_seal = 0
    purple_seal = 0

class card_multadd_modifier(Enum):
    bonus = 0
    mult = 4
    wild = 0
    glass = 0 
    steel = 0
    stone = 0
    gold =0 
    lucky = 20
    gold_seal = 0
    red_seal = 0
    blue_seal = 0
    purple_seal = 0

class card_multmult_modifier(Enum):
    bonus = 0
    mult = 0
    wild = 0
    glass = 0 
    steel = 2
    stone = 0
    gold =0 
    lucky = 0
    gold_seal = 0
    red_seal = 0
    blue_seal = 0
    purple_seal = 0

class joker_shop_chance(Enum): 
    negative = 0.3 #percent
    poly = 0.3
    holo = 1.4
    foil = 2.0

class card_shop_chance(Enum):
    poly = 1.2 #percent
    holo = 2.8
    foil = 4.0







