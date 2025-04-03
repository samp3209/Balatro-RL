from enum import Enum, auto
from typing import List, Optional


class HandType(Enum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9

class CardEnhancement(Enum):
    NONE = 0
    FOIL = 1  # +50 chips
    HOLO = 2  # +10 mult
    POLY = 3  # x1.5 mult
    WILD = 4  # All suits at once
    STEEL = 5  # x1.5 mult if held in hand
    GLASS = 6  # x2 mult, 1/4 chance to be destroyed
    GOLD = 7   # +3 money if held in hand
    STONE = 8  # +50 chips, no rank or suit
    LUCKY = 9  # 1/5 chance of +20 mult, 1/15 to get +20 money
    MULT = 10  # +4 mult
    BONUS = 11 # +30 chips

class PlanetType(Enum):
    MERCURY = auto()
    VENUS = auto()
    EARTH = auto()
    MARS = auto()
    JUPITER = auto()
    SATURN = auto()
    URANUS = auto()
    NEPTUNE = auto()
    PLUTO = auto()

class ShopItemType(Enum):
    JOKER = auto()
    TAROT = auto()
    PLANET = auto()
    BOOSTER = auto()
    VOUCHER = auto()

class joker_shop_chance(Enum): 
    negative = 0.3 #percent
    poly = 0.3
    holo = 1.4
    foil = 2.0

class card_shop_chance(Enum):
    poly = 1.2 #percent
    holo = 2.8
    foil = 4.0

class Suit(Enum):
    HEARTS = auto()
    DIAMONDS = auto()
    CLUBS = auto()
    SPADES = auto()
    WILD = auto()

class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 10
    QUEEN = 10
    KING = 10
    ACE = 11

class ConsumableType(Enum):
    TAROT = 1
    PLANET = 2


class TarotType(Enum):
    THE_FOOL = auto()
    THE_MAGICIAN = auto()
    THE_HIGH_PRIESTESS = auto()
    THE_EMPRESS = auto()
    THE_EMPEROR = auto()
    THE_HIEROPHANT = auto()
    THE_LOVERS = auto()
    THE_CHARIOT = auto()
    JUSTICE = auto()
    THE_HERMIT = auto()
    WHEEL_OF_FORTUNE = auto()
    STRENGTH = auto()
    THE_HANGED_MAN = auto()
    DEATH = auto()
    TEMPERANCE = auto()
    THE_DEVIL = auto()
    THE_TOWER = auto()
    THE_STAR = auto()
    THE_MOON = auto()
    THE_SUN = auto()
    JUDGEMENT = auto()
    THE_WORLD = auto()


class PackType(Enum):
    STANDARD = "Standard Pack"
    BUFFOON = "Buffoon Pack"
    CELESTIAL = "Celestial Pack"
    ARCANA = "Arcana Pack"
    JUMBO_STANDARD = "Jumbo Standard Pack"
    JUMBO_BUFFOON = "Jumbo Buffoon Pack"
    JUMBO_CELESTIAL = "Jumbo Celestial Pack"
    JUMBO_ARCANA = "Jumbo Arcana Pack"
    MEGA_CELESTIAL = "Mega Celestial Pack"
    MEGA_ARCANA = "Mega Arcana Pack"