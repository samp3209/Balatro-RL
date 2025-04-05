from collections import defaultdict
from typing import List, Optional, Union, Any
from Card import *
from Hand import *
from Enums import *  # Make sure ConsumableType is defined here
import random


class Consumable:
    """Wrapper class to store either a Tarot or Planet in the consumables list"""
    def __init__(self, item: Any):
        self.item = item
        if hasattr(item, 'tarot_type'):  # Check if it's a Tarot
            self.type = ConsumableType.TAROT
        elif hasattr(item, 'planet_type'):  # Check if it's a Planet
            self.type = ConsumableType.PLANET
        else:
            raise ValueError("Consumable must be either a Tarot or Planet")
        

class Inventory:
    def __init__(self):
        # Core storage
        self.jokers = []  # list of jokers
        self.deck = []  # list of cards in the current deck
        self.master_deck = []  # master list of all cards that should be in the deck
        self.consumables = []  # max is 2, stores Consumable objects (wrappers for Tarots/Planets)
        self.max_consumables = 2
        
        # Track last used cards
        self.last_tarot = None  # last played tarot card
        self.last_planet = None  # last played planet card
        
        # Stats and counters
        self.joker_sell_values = []  # list of joker sell values
        self.uncommon_joker_count = 0
        self.booster_skip = 0  # counts how many boosters we skipped
        self.tarot_used = 0  # counts tarot cards used
        self.planet_used = 0  # counts planet cards used
        self.money = 0  # add money attribute to track player's funds
        
        # Planet level trackers
        self.planet_levels = {
            PlanetType.PLUTO: 1,    # default 1 mult x 5 chips
            PlanetType.MERCURY: 1,  # default 2 mult x 10 chips
            PlanetType.URANUS: 1,   # default 2 mult x 20 chips
            PlanetType.VENUS: 1,    # default 3 mult x 30 chips
            PlanetType.SATURN: 1,   # default 4 mult x 30 chips
            PlanetType.EARTH: 1,    # default 4 mult x 40 chips
            PlanetType.MARS: 1,     # default 7 mult x 60 chips
            PlanetType.NEPTUNE: 1   # default 8 mult x 100 chips
        }

    def add_joker(self, joker) -> bool:
        """Add a joker to inventory. Returns True if successful."""
        self.jokers.append(joker)
        self.joker_sell_values.append(joker.sell_value)
        if joker.rarity == "Uncommon":
            self.uncommon_joker_count += 1
        return True
    
    def remove_joker(self, joker_index: int):
        """Remove joker at given index and return it."""
        if 0 <= joker_index < len(self.jokers):
            joker = self.jokers.pop(joker_index)
            # Update stats when removing
            self.joker_sell_values.remove(joker.sell_value)
            if joker.rarity == "Uncommon":
                self.uncommon_joker_count -= 1
            return joker
        return None

    def has_joker_space(self) -> bool:
        """Check if there's space for a new joker."""
        if len(self.jokers) == 5:
            return False
        return True 
    
    def add_consumable(self, item) -> bool:
        """Add a Tarot or Planet card to consumables. Returns True if successful."""
        if len(self.consumables) >= self.max_consumables:
            return False
        
        consumable = Consumable(item)
        self.consumables.append(consumable)
        return True
    
    def remove_consumable(self, index: int):
        """Remove consumable at given index and return the item."""
        if 0 <= index < len(self.consumables):
            consumable = self.consumables.pop(index)
            return consumable.item
        return None
    
    def use_tarot(self, index: int, selected_cards, game_state: dict):
        """Use a tarot card from consumables and return its effect."""
        if 0 <= index < len(self.consumables):
            consumable = self.consumables[index]
            if consumable.type == ConsumableType.TAROT:
                tarot = consumable.item
                # Apply tarot effect
                effect = tarot.apply_effect(selected_cards, self, game_state)
                # Update last tarot and stats
                self.last_tarot = tarot
                self.tarot_used += 1
                # Remove the used tarot
                self.consumables.pop(index)
                return effect
        return None
    
    def use_planet(self, index: int, hand_type, game_state: dict):
        """Use a planet card from consumables and return its effect."""
        if 0 <= index < len(self.consumables):
            consumable = self.consumables[index]
            if consumable.type == ConsumableType.PLANET:
                planet = consumable.item
                # Apply planet effect
                effect = planet.apply_effect(hand_type, game_state)
                # Update last planet and stats
                self.last_planet = planet
                self.planet_used += 1
                # Remove the used planet
                self.consumables.pop(index)
                
                # Level up the planet type
                planet_type = planet.planet_type
                self.planet_levels[planet_type] += 1
                
                return effect
        return None
    
    def get_consumable_tarot_indices(self) -> List[int]:
        """Get indices of all tarot cards in consumables."""
        return [i for i, consumable in enumerate(self.consumables) 
                if consumable.type == ConsumableType.TAROT]
    
    def get_consumable_planet_indices(self) -> List[int]:
        """Get indices of all planet cards in consumables."""
        return [i for i, consumable in enumerate(self.consumables) 
                if consumable.type == ConsumableType.PLANET]
    
    def get_available_space(self) -> int:
        """Get number of available spaces in consumables."""
        return max(0, self.max_consumables - len(self.consumables))
    
    def get_planet_level(self, planet_type: PlanetType) -> int:
        """Get the current level of a specific planet type."""
        return self.planet_levels.get(planet_type, 1)
    
    def get_planet_bonus(self, planet_type: PlanetType) -> tuple:
        """Get the current bonus multiplier and chips for a planet type based on its level."""
        level = self.planet_levels.get(planet_type, 1) - 1  # Subtract 1 to get bonus levels
        
        # Base bonuses per level for each planet
        bonuses = {
            PlanetType.PLUTO: (1, 10),     # +1 Mult, +10 chips per level
            PlanetType.MERCURY: (1, 15),   # +1 Mult, +15 chips per level
            PlanetType.URANUS: (1, 20),    # +1 Mult, +20 chips per level
            PlanetType.VENUS: (2, 20),     # +2 Mult, +20 chips per level
            PlanetType.SATURN: (3, 30),    # +3 Mult, +30 chips per level
            PlanetType.EARTH: (2, 15),     # +2 Mult, +15 chips per level
            PlanetType.MARS: (3, 30),      # +3 Mult, +30 chips per level
            PlanetType.NEPTUNE: (4, 40)    # +4 Mult, +40 chips per level
        }
        
        base_mult, base_chips = bonuses.get(planet_type, (0, 0))
        return (base_mult * level, base_chips * level)
    
    def calculate_hand_value(self, hand_type, game_state: dict) -> tuple:
        """
        Calculate the total value of a hand including all planet level bonuses
        Returns (total_mult, total_chips)
        """
        # Base values for each hand type
        base_values = {
            "HIGH_CARD": (1, 5),
            "PAIR": (2, 10),
            "TWO_PAIR": (2, 20),
            "THREE_OF_A_KIND": (3, 30),
            "STRAIGHT": (4, 30),
            "FULL_HOUSE": (4, 40),
            "FLUSH": (4, 35),
            "FOUR_OF_A_KIND": (7, 60),
            "STRAIGHT_FLUSH": (8, 100)
        }
        
        # Get base values for the hand type
        base_mult, base_chips = base_values.get(hand_type.name, (0, 0))
        
        # Initialize with base values
        total_mult = base_mult
        total_chips = base_chips
        
        # Map hand types to corresponding planet types
        hand_to_planet = {
            "HIGH_CARD": PlanetType.PLUTO,
            "PAIR": PlanetType.MERCURY,
            "TWO_PAIR": PlanetType.URANUS,
            "THREE_OF_A_KIND": PlanetType.VENUS,
            "STRAIGHT": PlanetType.SATURN,
            "FULL_HOUSE": PlanetType.EARTH,
            "FOUR_OF_A_KIND": PlanetType.MARS,
            "STRAIGHT_FLUSH": PlanetType.NEPTUNE
        }
        
        # Get the corresponding planet type for this hand
        planet_type = hand_to_planet.get(hand_type.name)
        
        if planet_type:
            # Add bonuses based on planet level
            bonus_mult, bonus_chips = self.get_planet_bonus(planet_type)
            total_mult += bonus_mult
            total_chips += bonus_chips
        
        # Apply any global multipliers from game state
        stake_multiplier = game_state.get('stake_multiplier', 1)
        total_chips = total_chips * stake_multiplier
        
        return (total_mult, total_chips)
    
    def add_card_to_deck(self, card: Card) -> bool:
        """Add a card to the deck and master deck. Returns True if successful."""
        self.deck.append(card)
        # Also add to master_deck to keep track of all cards
        self.master_deck.append(card)
        return True
    
    def remove_card_from_deck(self, card_index: int) -> Optional[Card]:
        """Remove a card from the deck at the given index and return it."""
        if 0 <= card_index < len(self.deck):
            card = self.deck.pop(card_index)
            # Also remove from master_deck if it's there
            for i, master_card in enumerate(self.master_deck):
                if master_card is card:
                    self.master_deck.pop(i)
                    break
            return card
        return None
    
    def shuffle_deck(self):
        """Shuffle the current deck."""
        random.shuffle(self.deck)
        print(f"Shuffled deck with {len(self.deck)} cards")

    def get_deck_size(self) -> int:
        """Return the number of cards in the deck."""
        return len(self.deck)
        
    def is_deck_empty(self) -> bool:
        """Check if the deck is empty."""
        return len(self.deck) == 0
    
    def initialize_standard_deck(self):
        """Initialize a standard 52-card deck."""
        self.deck = []
        self.master_deck = []  # Clear master deck too
        
        # Add cards with proper rank and suit enums
        for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
            for rank in [Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, 
                         Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN, 
                         Rank.JACK, Rank.QUEEN, Rank.KING]:
                card = Card(suit, rank)
                self.deck.append(card)
                self.master_deck.append(card)  # Add to master deck too
        
        # Validate that we have exactly 52 cards
        if len(self.deck) != 52:
            print(f"WARNING: Deck initialized with {len(self.deck)} cards, expected 52")
        
        # Print deck distribution for debugging
        self._print_deck_distribution()

    def _print_deck_distribution(self):
        """Print the distribution of cards in the deck by rank and suit"""
        rank_counts = defaultdict(int)
        suit_counts = defaultdict(int)
        
        # Check for invalid cards
        invalid_cards = []
        for i, card in enumerate(self.deck):
            try:
                # Verify card has proper Rank and Suit objects
                if not isinstance(card.rank, Rank):
                    invalid_cards.append((i, f"Invalid rank type: {type(card.rank)}"))
                elif not isinstance(card.suit, Suit):
                    invalid_cards.append((i, f"Invalid suit type: {type(card.suit)}"))
                else:
                    # Count valid cards
                    rank_counts[card.rank] += 1
                    suit_counts[card.suit] += 1
            except Exception as e:
                invalid_cards.append((i, f"Exception: {str(e)}"))
        
        # Print any invalid cards found
        if invalid_cards:
            print(f"WARNING: Found {len(invalid_cards)} invalid cards in deck:")
            for idx, error in invalid_cards:
                try:
                    card = self.deck[idx]
                    card_repr = str(card.__dict__)  # Safe way to get card contents
                    print(f"  - Index {idx}: {card_repr} - Error: {error}")
                except Exception as e:
                    print(f"  - Index {idx}: <Error displaying card: {str(e)}> - Error: {error}")
        
        try:
            # Print rank distribution
            valid_ranks = {}
            for rank, count in rank_counts.items():
                try:
                    valid_ranks[rank.name] = count
                except Exception:
                    valid_ranks[str(rank)] = count
            print("Ranks:", valid_ranks)
            
            # Print suit distribution
            valid_suits = {}
            for suit, count in suit_counts.items():
                try:
                    valid_suits[suit.name] = count
                except Exception:
                    valid_suits[str(suit)] = count
            print("Suits:", valid_suits)
        except Exception as e:
            print(f"Error printing distribution: {e}")
        
        print(f"Total cards: {len(self.deck)}")
        print(f"Master deck size: {len(self.master_deck)}")

    def reset_deck(self, played_cards: List[Card], discarded_cards: List[Card], hand_cards: List[Card]):
        """
        Reset the deck by returning all played and discarded cards to the deck
        
        Args:
            played_cards: Cards that were played
            discarded_cards: Cards that were discarded
            hand_cards: Cards still in hand
        """
        # Start with an empty deck
        self.deck = []
        
        # Create a set to track unique cards we've added back to the deck
        added_cards = set()
        
        # First add all master deck cards to the deck, resetting their state
        for card in self.master_deck:
            card_id = (card.rank, card.suit, id(card))
            if card_id not in added_cards:
                added_cards.add(card_id)
                card.reset_state()
                self.deck.append(card)
        
        # Now add any cards from played/discarded/hand that weren't in the master deck
        for card in played_cards + discarded_cards + hand_cards:
            card_id = (card.rank, card.suit, id(card))
            if card_id not in added_cards:
                # This is a new card that wasn't in our master deck, so add it
                added_cards.add(card_id)
                card.reset_state()
                self.deck.append(card)
                self.master_deck.append(card)  # Also add to master deck for future rounds
        
        # Handle special cases like broken glass cards
        cards_to_remove = []
        for card in list(self.deck):
            if card.enhancement == CardEnhancement.GLASS and not card.in_deck:
                cards_to_remove.append(card)
                # Also remove from master_deck
                for master_card in list(self.master_deck):
                    if master_card is card:
                        self.master_deck.remove(master_card)
        
        for card in cards_to_remove:
            if card in self.deck:
                self.deck.remove(card)
        
        if cards_to_remove:
            print(f"Removed {len(cards_to_remove)} broken glass cards from the deck")
        
        # Validate deck size
        print(f"Reset deck: {len(self.deck)} cards in deck, {len(self.master_deck)} cards in master deck")
        self._print_deck_distribution()
        # Shuffle the deck
        self.shuffle_deck()