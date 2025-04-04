from enum import Enum, auto
from typing import List, Optional, Callable, Dict, Any
import random
from Card import *
from Enums import *

class TarotEffect:
    def __init__(self):
        self.selected_cards_required = 0
        self.money_gained = 0
        self.cards_created = []
        self.cards_enhanced = []
        self.cards_deleted = []
        self.jokers_created = []
        self.tarots_created = []
        self.message = ""

class Tarot:
    def __init__(self, tarot_type: TarotType):
        self.tarot_type = tarot_type
        self.name = self._get_name()
        self.description = self._get_description()
        self.selected_cards_required = self._get_selected_cards_required()
        self.price = 3 
        self.sell_value = 1
        
    def _get_name(self) -> str:
        """Convert enum to readable name"""
        return self.tarot_type.name.replace('_', ' ').title()
    
    def _get_selected_cards_required(self) -> int:
        """Return how many cards need to be selected for this tarot"""
        selection_requirements = {
            TarotType.THE_MAGICIAN: 2,
            TarotType.THE_EMPRESS: 2,
            TarotType.THE_HIEROPHANT: 2,
            TarotType.THE_LOVERS: 1,
            TarotType.THE_CHARIOT: 1,
            TarotType.JUSTICE: 1,
            TarotType.WHEEL_OF_FORTUNE: 1,
            TarotType.STRENGTH: 2,
            TarotType.THE_HANGED_MAN: 2,
            TarotType.DEATH: 2,
            TarotType.THE_DEVIL: 1,
            TarotType.THE_TOWER: 1,
            TarotType.THE_STAR: 3,
            TarotType.THE_MOON: 3,
            TarotType.THE_SUN: 3,
            TarotType.THE_WORLD: 3
        }
        return selection_requirements.get(self.tarot_type, 0)
    
    def _get_description(self) -> str:
        """Return description of the tarot card"""
        descriptions = {
            TarotType.THE_FOOL: "Creates the last Tarot and Planet card used during this run",
            TarotType.THE_MAGICIAN: "Turns two selected cards into lucky cards",
            TarotType.THE_HIGH_PRIESTESS: "Generates two random planet cards to the inventory",
            TarotType.THE_EMPRESS: "Selects two cards into mult cards (+4 mult)",
            TarotType.THE_EMPEROR: "Creates up to 2 random tarot cards",
            TarotType.THE_HIEROPHANT: "Select two cards into bonus cards (+30 chips)",
            TarotType.THE_LOVERS: "Enhances 1 card into a wild card (all suits at the same time)",
            TarotType.THE_CHARIOT: "Enhances 1 card into a steel card (x1.5 mult if held in hand)",
            TarotType.JUSTICE: "Enhances 1 card into a glass card (x2 mult, 1/4 chance to be destroyed after hand)",
            TarotType.THE_HERMIT: "Doubles money (max 20)",
            TarotType.WHEEL_OF_FORTUNE: "20% chance to get foil, 10% to get holo, 5% to get polychrome",
            TarotType.STRENGTH: "Increases the rank of two selected cards by 1. (ace wraps around to 2)",
            TarotType.THE_HANGED_MAN: "Deletes 2 selected cards",
            TarotType.DEATH: "Converts the left selected card into the right",
            TarotType.TEMPERANCE: "Adds the total sell amount of jokers in inventory",
            TarotType.THE_DEVIL: "Enhances 1 selected card into a gold card (+3 money if held in hand at end of round)",
            TarotType.THE_TOWER: "Enhances 1 selected card into a stone card (+50 chips no rank or suit)",
            TarotType.THE_STAR: "Converts 3 selected cards to diamond",
            TarotType.THE_MOON: "Converts 3 selected cards to clubs",
            TarotType.THE_SUN: "Converts 3 selected cards to hearts",
            TarotType.JUDGEMENT: "Creates a random joker (must have room)",
            TarotType.THE_WORLD: "Converts 3 selected cards to spades"
        }
        return descriptions.get(self.tarot_type, "Unknown effect")
    
    def apply_effect(self, selected_cards: List[Card], inventory: Any, game_state: Dict) -> TarotEffect:
        """
        Apply the tarot effect based on type and return the result
        """
        effect = TarotEffect()
        
        if len(selected_cards) < self.selected_cards_required:
            effect.message = f"This tarot requires {self.selected_cards_required} selected cards."
            return effect
        
        if self.tarot_type == TarotType.THE_FOOL:
            last_tarot = game_state.get('last_tarot_used')
            last_planet = game_state.get('last_planet_used')
            
            if last_tarot:
                effect.cards_created.append(last_tarot)
            
            if last_planet:
                effect.cards_created.append(last_planet)
                
            effect.message = "Created copies of the last Tarot and Planet cards."
            
        elif self.tarot_type == TarotType.THE_MAGICIAN:
            for card in selected_cards[:2]:
                card.enhancement = CardEnhancement.LUCKY
                effect.cards_enhanced.append(card)
            
            effect.message = "Turned selected cards into lucky cards."
            
        elif self.tarot_type == TarotType.THE_HERMIT:
            current_money = game_state.get('money', 0)
            money_gain = min(current_money, 20)
            effect.money_gained = money_gain
            effect.message = f"Doubled money by +${money_gain}."
        
        elif self.tarot_type == TarotType.THE_EMPRESS:
            for card in selected_cards[:2]:
                card.enhancement = CardEnhancement.MULT
                effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected cards with +4 mult."

        elif self.tarot_type == TarotType.THE_HIEROPHANT:
            for card in selected_cards[:2]:
                card.enhancement = CardEnhancement.BONUS
                effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected cards with +30 chips."

        elif self.tarot_type == TarotType.THE_LOVERS:
            card = selected_cards[0]
            card.enhancement = CardEnhancement.WILD
            effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected card to be wild (all suits)."

        elif self.tarot_type == TarotType.THE_CHARIOT:
            card = selected_cards[0]
            card.enhancement = CardEnhancement.STEEL
            effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected card to steel (x1.5 mult if held)."

        elif self.tarot_type == TarotType.JUSTICE:
            card = selected_cards[0]
            card.enhancement = CardEnhancement.GLASS
            effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected card to glass (x2 mult, may break)."

        elif self.tarot_type == TarotType.WHEEL_OF_FORTUNE:
            if selected_cards:
                card = selected_cards[0]
                roll = random.random()
                
                if roll < 0.05:
                    card.enhancement = CardEnhancement.POLY
                    effect.message = "Card enhanced to polychrome (x1.5 mult)."
                elif 0.05 < roll < 0.2:
                    card.enhancement = CardEnhancement.HOLO
                    effect.message = "Card enhanced to holographic (+10 mult)."
                elif roll < 0.35:
                    card.enhancement = CardEnhancement.FOIL
                    effect.message = "Card enhanced to foil (+50 chips)."
                else:
                    effect.message = "No enhancement applied."
                    
                effect.cards_enhanced.append(card)

        elif self.tarot_type == TarotType.STRENGTH:
            for card in selected_cards[:2]:
                if hasattr(card, 'rank'):
                    if card.rank.value == Rank.ACE.value:
                        card.rank = Rank.TWO
                    elif card.rank.value < Rank.KING.value:
                        next_rank = None
                        for rank in Rank:
                            if rank.value == card.rank.value + 1:
                                next_rank = rank
                                break
                        if next_rank:
                            card.rank = next_rank
                    
                effect.cards_enhanced.append(card)
            
            effect.message = "Increased rank of selected cards by 1."

        elif self.tarot_type == TarotType.THE_HANGED_MAN:
            for card in selected_cards[:2]:
                effect.cards_deleted.append(card)
            
            effect.message = "Deleted selected cards."

        elif self.tarot_type == TarotType.DEATH:
            if len(selected_cards) >= 2:
                left_card = selected_cards[0]
                right_card = selected_cards[1]
                
                left_card.suit = right_card.suit
                left_card.rank = right_card.rank
                left_card.enhancement = right_card.enhancement
                
                effect.cards_enhanced.append(left_card)
                effect.message = "Converted left card into a copy of the right card."
            else:
                effect.message = "Need two cards to perform conversion."

        elif self.tarot_type == TarotType.TEMPERANCE:
            if inventory and hasattr(inventory, 'jokers'):
                total_sell_value = sum(getattr(joker, 'sell_value', 0) for joker in inventory.jokers)
                effect.money_gained = total_sell_value
                effect.message = f"Added ${total_sell_value} from joker sell values."
            else:
                effect.message = "No jokers in inventory."

        elif self.tarot_type == TarotType.THE_DEVIL:
            card = selected_cards[0]
            card.enhancement = CardEnhancement.GOLD
            effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected card to gold (+3 money if held)."

        elif self.tarot_type == TarotType.THE_TOWER:
            card = selected_cards[0]
            card.enhancement = CardEnhancement.STONE
            effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected card to stone (+50 chips, no rank/suit)."

        elif self.tarot_type == TarotType.THE_STAR:
            for card in selected_cards[:3]:
                card.suit = Suit.DIAMONDS
                effect.cards_enhanced.append(card)
            
            effect.message = "Converted selected cards to diamonds."

        elif self.tarot_type == TarotType.THE_MOON:
            for card in selected_cards[:3]:
                card.suit = Suit.CLUBS
                effect.cards_enhanced.append(card)
            
            effect.message = "Converted selected cards to clubs."

        elif self.tarot_type == TarotType.THE_SUN:
            for card in selected_cards[:3]:
                card.suit = Suit.HEARTS
                effect.cards_enhanced.append(card)
            
            effect.message = "Converted selected cards to hearts."

        elif self.tarot_type == TarotType.THE_WORLD:
            for card in selected_cards[:3]:
                card.suit = Suit.SPADES
                effect.cards_enhanced.append(card)
            
            effect.message = "Converted selected cards to spades."
        
        
        return effect
    

def create_tarot(tarot_type: TarotType) -> Tarot:
    """Create a tarot card of the specified type"""
    return Tarot(tarot_type)

def create_random_tarot() -> Tarot:
    """Create a random tarot card"""
    tarot_types = list(TarotType)
    random_type = random.choice(tarot_types)
    return create_tarot(random_type)

def create_tarot_by_name(name: str) -> Optional[Tarot]:
    """
    Create a Tarot card object by name
    
    Args:
        name: The name of the tarot card (e.g., "Hierophant", "Moon", "Sun")
        
    Returns:
        A Tarot card object with appropriate properties
    """
    # Convert name to lowercase and remove "the" if present
    tarot_name = name.lower().strip()
    if tarot_name.startswith("the "):
        tarot_name = tarot_name[4:]
    
    # Map string names to TarotType enum values
    tarot_map = {
        "fool": TarotType.THE_FOOL,
        "magician": TarotType.THE_MAGICIAN,
        "high priestess": TarotType.THE_HIGH_PRIESTESS,
        "empress": TarotType.THE_EMPRESS,
        "emperor": TarotType.THE_EMPEROR,
        "hierophant": TarotType.THE_HIEROPHANT,
        "lovers": TarotType.THE_LOVERS,
        "chariot": TarotType.THE_CHARIOT,
        "justice": TarotType.JUSTICE,
        "hermit": TarotType.THE_HERMIT,
        "wheel of fortune": TarotType.WHEEL_OF_FORTUNE,
        "strength": TarotType.STRENGTH,
        "hanged man": TarotType.THE_HANGED_MAN,
        "death": TarotType.DEATH,
        "temperance": TarotType.TEMPERANCE,
        "devil": TarotType.THE_DEVIL,
        "tower": TarotType.THE_TOWER,
        "star": TarotType.THE_STAR,
        "moon": TarotType.THE_MOON,
        "sun": TarotType.THE_SUN,
        "judgement": TarotType.JUDGEMENT,
        "world": TarotType.THE_WORLD
    }
    
    if tarot_name in tarot_map:
        return create_tarot(tarot_map[tarot_name])
    
    print(f"Warning: Unknown tarot card name '{name}'")
    return None