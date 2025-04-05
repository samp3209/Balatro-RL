from enum import Enum, auto
from typing import List, Optional, Callable, Dict
import random
from Card import *
from Enums import *


class PlanetEffect:
    def __init__(self):
        self.hand_type_upgraded = None
        self.mult_bonus = 0
        self.chip_bonus = 0
        self.message = ""


class Planet:
    def __init__(self, planet_type: PlanetType):
        self.planet_type = planet_type
        self.name = self._get_name()
        self.description = self._get_description()
        self.price = 3
        self.sell_value = 1
        
    def _get_name(self) -> str:
        """Convert enum to readable name"""
        return self.planet_type.name.title()
    
    def _get_description(self) -> str:
        """Return description of the planet card's effect on hand types"""
        descriptions = {
            PlanetType.PLUTO: "Upgrades High Card: +1 Mult, +10 chips",
            PlanetType.MERCURY: "Upgrades Pair: +1 Mult, +15 chips",
            PlanetType.URANUS: "Upgrades Two Pair: +1 Mult, +20 chips",
            PlanetType.VENUS: "Upgrades Three of a Kind: +2 Mult, +20 chips",
            PlanetType.SATURN: "Upgrades Straight: +3 Mult, +30 chips",
            PlanetType.EARTH: "Upgrades Full House: +2 Mult, +15 chips",
            PlanetType.MARS: "Upgrades Four of a Kind: +3 Mult, +30 chips",
            PlanetType.NEPTUNE: "Upgrades Straight Flush: +4 Mult, +40 chips"
        }
        return descriptions.get(self.planet_type, "Unknown effect")
    
    def get_hand_type(self) -> HandType:
        """Return the hand type this planet upgrades"""
        hand_mappings = {
            PlanetType.PLUTO: HandType.HIGH_CARD,
            PlanetType.MERCURY: HandType.PAIR,
            PlanetType.URANUS: HandType.TWO_PAIR,
            PlanetType.VENUS: HandType.THREE_OF_A_KIND,
            PlanetType.SATURN: HandType.STRAIGHT,
            PlanetType.EARTH: HandType.FULL_HOUSE,
            PlanetType.MARS: HandType.FOUR_OF_A_KIND,
            PlanetType.NEPTUNE: HandType.STRAIGHT_FLUSH
        }
        return hand_mappings.get(self.planet_type)
    
    def get_base_values(self) -> tuple:
        """Return the base multiplier and chips for the hand type"""
        base_values = {
            HandType.HIGH_CARD: (1, 5),
            HandType.PAIR: (2, 10),
            HandType.TWO_PAIR: (2, 20),
            HandType.THREE_OF_A_KIND: (3, 30),
            HandType.STRAIGHT: (4, 30),
            HandType.FULL_HOUSE: (4, 40),
            HandType.FOUR_OF_A_KIND: (7, 60),
            HandType.STRAIGHT_FLUSH: (8, 100),

        }
        
        hand_type = self.get_hand_type()
        return base_values.get(hand_type, (0, 0))
    
    def get_bonus_values(self) -> tuple:
        """Return the bonus multiplier and chips this planet adds"""
        bonus_values = {
            PlanetType.PLUTO: (1, 10),     #High Card
            PlanetType.MERCURY: (1, 15),   #Pair
            PlanetType.URANUS: (1, 20),    #Two Pair
            PlanetType.VENUS: (2, 20),     #Three of a Kind
            PlanetType.SATURN: (3, 30),    #Straight
            PlanetType.EARTH: (2, 15),     #Full House
            PlanetType.MARS: (3, 30),      #Four of a Kind
            PlanetType.NEPTUNE: (4, 40)    #Straight Flush
        }
        
        return bonus_values.get(self.planet_type, (0, 0))
    
    def apply_effect(self, hand_type, game_state: Dict) -> Dict:
        """
        Apply the planet's effect based on the hand type
        
        Args:
            hand_type: The hand type being played
            game_state: Dictionary containing game state information
            
        Returns:
            Dictionary containing effect details
        """
        result = {
            'message': f"Used {self.name} planet card",
            'mult_bonus': 0,
            'chip_bonus': 0
        }
        
        mult_bonus, chip_bonus = self.get_bonus_values()
        
        result['mult_bonus'] = mult_bonus
        result['chip_bonus'] = chip_bonus
        result['message'] = f"Used {self.name} planet card: +{mult_bonus} mult, +{chip_bonus} chips"
        
        return result


def create_planet(planet_type: PlanetType) -> Planet:
    """Create a planet card of the specified type"""
    return Planet(planet_type)

def create_random_planet() -> Planet:
    """Create a random planet card"""
    planet_types = list(PlanetType)
    random_type = random.choice(planet_types)
    return create_planet(random_type)


def create_planet_by_name(name: str) -> Optional[Planet]:
    """
    Create a Planet card object by name
    
    Args:
        name: The name of the planet (e.g., "Mars", "Venus", "Jupiter")
        
    Returns:
        A Planet card object with appropriate properties
    """
    planet_name = name.lower().strip()
    
    planet_map = {
        "pluto": PlanetType.PLUTO,
        "mercury": PlanetType.MERCURY,
        "uranus": PlanetType.URANUS,
        "venus": PlanetType.VENUS,
        "saturn": PlanetType.SATURN,
        "earth": PlanetType.EARTH,
        "mars": PlanetType.MARS,
        "neptune": PlanetType.NEPTUNE,
        "jupiter": PlanetType.JUPITER
    }
    
    if planet_name in planet_map:
        return create_planet(planet_map[planet_name])
    
    print(f"Warning: Unknown planet name '{name}'")
    return None