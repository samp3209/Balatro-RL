from enum import Enum, auto
from typing import List, Optional, Callable
import random
from Card import *
from Inventory import *
from JokerCreation import *
from Enums import *
from Planet import *

class TarotEffect:
    def __init__(self):
        self.selected_cards_required = 0
        self.money_gained = 0
        self.cards_created = []
        self.cards_enhanced = []
        self.cards_deleted = []
        self.jokers_created = []
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
    
    def apply_effect(self, selected_cards: List[Card], Inventory: Inventory, game_state: dict) -> TarotEffect:
        """
        Apply the tarot effect based on type and return the result
        """
        effect = TarotEffect()
        
        # Validate selection count
        if len(selected_cards) < self.selected_cards_required:
            effect.message = f"This tarot requires {self.selected_cards_required} selected cards."
            return effect
        
        # Apply effects based on tarot type
        if self.tarot_type == TarotType.THE_FOOL:
            # Implementation for The Fool
            last_tarot = game_state.get('last_tarot_used')
            last_planet = game_state.get('last_planet_used')
            
            if last_tarot:
                # Logic to create a copy of the last tarot
                effect.cards_created.append(last_tarot)
            
            if last_planet:
                # Logic to create a copy of the last planet
                effect.cards_created.append(last_planet)
                
            effect.message = "Created copies of the last Tarot and Planet cards."
            
        elif self.tarot_type == TarotType.THE_MAGICIAN:
            # Turn two selected cards into lucky cards
            for card in selected_cards[:2]:
                # Implementation for making a card "lucky"
                # This would set some property on the card
                effect.cards_enhanced.append(card)
            
            effect.message = "Turned selected cards into lucky cards."
            
        elif self.tarot_type == TarotType.THE_HIGH_PRIESTESS:
            # Generate random planet cards
            available_space = Inventory.get_available_space()
            num_planets = min(2, available_space)
            
            # Logic to create planet cards would go here
            for _ in range(num_planets):
                # Create planet card
                pass
                
            effect.message = f"Generated {num_planets} random planet cards."
            
        elif self.tarot_type == TarotType.THE_HERMIT:
            # Double money up to a max of 20
            current_money = game_state.get('money', 0)
            money_gain = min(current_money, 20)
            effect.money_gained = money_gain
            effect.message = f"Doubled money by +${money_gain}."
        
        elif self.tarot_type == TarotType.JUDGEMENT:
            # Create a random joker if there's room
            if Inventory.has_joker_space():
                # Logic to create random joker
                # effect.jokers_created.append(random_joker)
                effect.message = "Created a random joker."
            else:
                effect.message = "No room for a joker in inventory."


        elif self.tarot_type == TarotType.THE_EMPRESS:
            # Selects two cards into mult cards (+4 mult)
            for card in selected_cards[:2]:
                card.enhancement = CardEnhancement.MULT
                effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected cards with +4 mult."

        elif self.tarot_type == TarotType.THE_EMPEROR:
            # Creates up to 2 random tarot cards
            available_space = Inventory.get_available_space()
            num_tarots = min(2, available_space)
            
            for _ in range(num_tarots):
                random_tarot = create_random_tarot()
                effect.tarots_created.append(random_tarot)
                
            effect.message = f"Created {num_tarots} random tarot cards."

        elif self.tarot_type == TarotType.THE_HIEROPHANT:
            # Select two cards into bonus cards (+30 chips)
            for card in selected_cards[:2]:
                card.enhancement = CardEnhancement.BONUS
                effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected cards with +30 chips."

        elif self.tarot_type == TarotType.THE_LOVERS:
            # Enhances 1 card into a wild card (all suits at the same time)
            card = selected_cards[0]
            card.enhancement = CardEnhancement.WILD
            effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected card to be wild (all suits)."


        elif self.tarot_type == TarotType.THE_CHARIOT:
            # Enhances 1 card into a steel card (x1.5 mult if held in hand)
            card = selected_cards[0]
            card.enhancement = CardEnhancement.STEEL
            effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected card to steel (x1.5 mult if held)."


        elif self.tarot_type == TarotType.JUSTICE:
            # Enhances 1 card into a glass card (x2 mult, 1/4 chance to be destroyed after hand)
            card = selected_cards[0]
            card.enhancement = CardEnhancement.GLASS
            effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected card to glass (x2 mult, may break)."


        elif self.tarot_type == TarotType.THE_HERMIT:
            # Doubles money (max 20)
            current_money = game_state.get('money', 0)
            money_gain = min(current_money, 20)
            effect.money_gained = money_gain
            
            effect.message = f"Doubled money by +${money_gain}."


        elif self.tarot_type == TarotType.WHEEL_OF_FORTUNE:
            # 20% chance to get foil, 10% to get holo, 5% to get polychrome
            card = selected_cards[0]
            roll = random.random()
            
            if roll < 0.05:  # 5% for polychrome
                card.enhancement = CardEnhancement.POLY
                effect.message = "Card enhanced to polychrome (x1.5 mult)."
            elif roll < 0.15:  # 10% for holo
                card.enhancement = CardEnhancement.HOLO
                effect.message = "Card enhanced to holographic (+10 mult)."
            elif roll < 0.35:  # 20% for foil
                card.enhancement = CardEnhancement.FOIL
                effect.message = "Card enhanced to foil (+50 chips)."
            else:
                effect.message = "No enhancement applied."
                
            effect.cards_enhanced.append(card)

        elif self.tarot_type == TarotType.STRENGTH:
            # Increases the rank of two selected cards by 1. (ace wraps around to 2)
            for card in selected_cards[:2]:
                if hasattr(card, 'rank'):
                    if card.rank.value == 1:  # Ace
                        card.rank = 2  # Set to 2
                    else:
                        card.rank = min(card.rank.value + 1, 13)  # Max is King (13)
                    
                effect.cards_enhanced.append(card)
            
            effect.message = "Increased rank of selected cards by 1."

        elif self.tarot_type == TarotType.THE_HANGED_MAN:
            # Deletes 2 selected cards
            for card in selected_cards[:2]:
                effect.cards_deleted.append(card)
            
            effect.message = "Deleted selected cards."


        elif self.tarot_type == TarotType.DEATH:
            # Converts the left selected card into the right
            if len(selected_cards) >= 2:
                left_card = selected_cards[0]
                right_card = selected_cards[1]
                
                # Copy properties from right to left
                left_card.suit = right_card.suit
                left_card.rank = right_card.rank
                left_card.enhancement = right_card.enhancement
                
                effect.cards_enhanced.append(left_card)
                effect.message = "Converted left card into a copy of the right card."
            else:
                effect.message = "Need two cards to perform conversion."

        elif self.tarot_type == TarotType.TEMPERANCE:
            # Adds the total sell amount of jokers in inventory
            total_sell_value = sum(joker.sell_value for joker in Inventory.jokers)
            effect.money_gained = total_sell_value
            
            effect.message = f"Added ${total_sell_value} from joker sell values."

        elif self.tarot_type == TarotType.THE_DEVIL:
            # Enhances 1 selected card into a gold card (+3 money if held in hand at end of round)
            card = selected_cards[0]
            card.enhancement = CardEnhancement.GOLD
            effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected card to gold (+3 money if held)."

        elif self.tarot_type == TarotType.THE_TOWER:
            # Enhances 1 selected card into a stone card (+50 chips no rank or suit)
            card = selected_cards[0]
            card.enhancement = CardEnhancement.STONE
            effect.cards_enhanced.append(card)
            
            effect.message = "Enhanced selected card to stone (+50 chips, no rank/suit)."

        elif self.tarot_type == TarotType.THE_STAR:
            # Converts 3 selected cards to diamond
            for card in selected_cards[:3]:
                card.suit = Suit.DIAMOND
                effect.cards_enhanced.append(card)
            
            effect.message = "Converted selected cards to diamonds."

        elif self.tarot_type == TarotType.THE_STAR:
            # Converts 3 selected cards to diamond
            for card in selected_cards[:3]:
                card.suit = Suit.DIAMOND
                effect.cards_enhanced.append(card)
            
            effect.message = "Converted selected cards to diamonds."

        elif self.tarot_type == TarotType.THE_WORLD:
            # Converts 3 selected cards to spades
            for card in selected_cards[:3]:
                card.suit = Suit.SPADE
                effect.cards_enhanced.append(card)
            
            effect.message = "Converted selected cards to spades."
        
        return effect
    


def create_tarot(tarot_type: TarotType) -> Tarot:
    return Tarot(tarot_type)

def create_random_tarot() -> Tarot:
    tarot_types = list(TarotType)
    random_type = random.choice(tarot_types)
    return create_tarot(random_type)

def create_random_planet():
    # Your implementation to create a random planet
    planet_types = list(PlanetType)
    random_type = random.choice(planet_types)
    return create_planet(random_type)

