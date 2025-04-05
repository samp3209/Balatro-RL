from enum import Enum, auto
from typing import List, Optional, Union, Tuple, Dict, Any
import random
from Card import *
from JokerCreation import *
from Joker import *
from Tarot import *
from Enums import *
from Planet import *


class ShopItem:
    def __init__(self, item_type: ShopItemType, item: Union[Joker, Tarot, Planet, str], price: Optional[int] = None):
        self.item_type = item_type
        self.item = item
        
        if price is None and hasattr(item, 'price'):
            self.price = item.price
        else:
            self.price = price or 0
            
    def get_name(self) -> str:
        if self.item_type in [ShopItemType.JOKER, ShopItemType.TAROT, ShopItemType.PLANET]:
            return self.item.name
        else:
            return str(self.item)
            
    def get_description(self) -> str:
        if hasattr(self.item, 'description'):
            return self.item.description
        elif self.item_type == ShopItemType.BOOSTER:
            return "Adds cards to your deck."
        elif self.item_type == ShopItemType.VOUCHER:
            return "Discount on your next purchase."
        else:
            return ""


class Shop:
    def __init__(self):
        self.items = [None, None, None, None]  # Two main items and two pack/booster items
        self.discount = 0
        self.has_voucher = False
        self.restock()
        
    def restock(self):
        """Restock all empty shop slots"""
        for i in range(2):
            if self.items[i] is None:
                self.items[i] = self._generate_random_item()
                
        for i in range(2, 4):
            if self.items[i] is None:
                self.items[i] = self._generate_random_booster()

    def buy_item(self, slot: int, inventory) -> bool:
        """
        Buy an item from the shop
        Returns True if purchase was successful
        """
        if not (0 <= slot < len(self.items)) or self.items[slot] is None:
            return False
            
        shop_item = self.items[slot]
        final_price = max(0, shop_item.price - self.discount)
        
        if inventory.money < final_price:
            return False
            
        inventory.money -= final_price 
        item_added = False
        
        if shop_item.item_type == ShopItemType.JOKER:
            if inventory.has_joker_space():
                inventory.add_joker(shop_item.item)
                item_added = True
                
        elif shop_item.item_type == ShopItemType.TAROT:
            if inventory.get_available_space() > 0:
                inventory.add_consumable(shop_item.item)
                item_added = True
                
        elif shop_item.item_type == ShopItemType.PLANET:
            if inventory.get_available_space() > 0:
                inventory.add_consumable(shop_item.item)
                item_added = True
                
        elif shop_item.item_type == ShopItemType.BOOSTER:
            self._process_booster(shop_item.item, inventory)
            item_added = True
            
        elif shop_item.item_type == ShopItemType.VOUCHER:
            self.discount = shop_item.item 
            self.has_voucher = True
            item_added = True
        
        if item_added:
            self.items[slot] = None
            return True
            
        return False
    
    def sell_item(self, item_type: str, item_index: int, inventory) -> int:
        """
        Sell an item from the inventory
        Returns the amount of money received
        """
        sell_value = 0
        
        if item_type == "joker" and 0 <= item_index < len(inventory.jokers):
            joker = inventory.remove_joker(item_index)
            if joker:
                sell_value = joker.sell_value
                
        elif item_type == "consumable" and 0 <= item_index < len(inventory.consumables):
            consumable = inventory.remove_consumable(item_index)
            if consumable:
                sell_value = consumable.sell_value
                
        
        if sell_value > 0:
            inventory.money += sell_value
            
        return sell_value
    
    def skip_booster(self, slot: int, inventory) -> bool:
        """
        Skip a booster pack and increment the booster_skip counter
        Returns True if successful
        """
        if 2 <= slot < 4 and self.items[slot] is not None:
            if self.items[slot].item_type == ShopItemType.BOOSTER:
                inventory.booster_skip += 1
                self.items[slot] = None
                return True
        return False
    
    def get_item_price(self, slot: int) -> int:
        """Get the price of an item with any discount applied"""
        if 0 <= slot < len(self.items) and self.items[slot] is not None:
            return max(0, self.items[slot].price - self.discount)
        return 0
    
    def _generate_random_item(self) -> ShopItem:
        """Generate a random shop item (Joker, Tarot, Planet, or Voucher)"""
        item_type = random.choices(
            [ShopItemType.JOKER, ShopItemType.TAROT, ShopItemType.PLANET, ShopItemType.VOUCHER],
            weights=[0.5, 0.25, 0.20, 0.05],
            k=1
        )[0]
        
        if item_type == ShopItemType.JOKER:
            joker_names = [
                "Green Joker", "Mr. Bones", "Delayed Gratification", 
                "Clever", "Mad", "Wily", "Crafty", "Misprint",
                "Wrathful", "Smiley", "Even Steven", "Blue",
                "Walkie Talkie", "Rocket", "Scary Face", "Banner",
                "The Duo", "Gluttonous", "Fortune Teller", "Business Card",
                "Baseball"
            ]
            joker_name = random.choice(joker_names)
            joker = create_joker(joker_name)
            return ShopItem(ShopItemType.JOKER, joker)
            
        elif item_type == ShopItemType.TAROT:
            tarot = create_random_tarot()
            return ShopItem(ShopItemType.TAROT, tarot)
            
        elif item_type == ShopItemType.PLANET:
            planet = create_random_planet()
            return ShopItem(ShopItemType.PLANET, planet)
            
        elif item_type == ShopItemType.VOUCHER:
            voucher_value = random.choice([1, 2, 3])
            return ShopItem(ShopItemType.VOUCHER, voucher_value, price=voucher_value+1)
    
    def _generate_random_booster(self) -> ShopItem:
        """Generate a random booster pack"""
        booster_types = [
            "Standard Pack",
            "Enhanced Pack",
            "Rare Pack",
            "Suited Pack"
        ]
        
        booster_type = random.choice(booster_types)
        price = {
            "Standard Pack": 2,
            "Enhanced Pack": 4,
            "Rare Pack": 5,
            "Suited Pack": 3
        }.get(booster_type, 2)
        
        return ShopItem(ShopItemType.BOOSTER, booster_type, price=price)
    
    def _process_booster(self, booster_type: str, inventory):
        """Process the booster pack and add cards to inventory"""
        if booster_type == "Standard Pack":
            for _ in range(5):
                suit = random.choice(list(Suit))
                rank = random.randint(1, 13)
                card = Card(rank, suit)
                inventory.add_card_to_deck(card)
 



class AnteShops:
    def __init__(self):
        self.ante_shops = self._initialize_ante_shops()
        
    def _initialize_ante_shops(self) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
        """Initialize all ante shops with their items and prices"""
        shops = {}
        
        shops[1] = {
            "small_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Green", "price": 4},
                {"item_type": ShopItemType.PLANET, "name": "Mars", "price": 3},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.BUFFOON, "price": 4, 
                 "contents": ["Mr. Bones", "Cartomancer"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.CELESTIAL, "price": 4, 
                 "contents": ["Saturn", "Uranus", "Pluto"]}
            ],
            "medium_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Delayed Gratification", "price": 4},
                {"item_type": ShopItemType.JOKER, "name": "Square", "price": 4},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.CELESTIAL, "price": 4, 
                 "contents": ["Saturn", "Uranus", "Pluto"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.ARCANA, "price": 4, 
                 "contents": ["Justice", "Moon", "Magician"]}
            ],
            "boss_blind": [
                {"item_type": ShopItemType.PLANET, "name": "Uranus", "price": 3},
                {"item_type": ShopItemType.TAROT, "name": "Hierophant", "price": 3},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_ARCANA, "price": 6, 
                 "contents": ["Devil", "World", "Magician", "Emperor", "Sun"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.CELESTIAL, "price": 6, 
                 "contents": ["Earth", "Venus", "Mars", "Mercury", "Neptune"]}
            ]
        }
        
        shops[2] = {
            "small_blind": [
                {"item_type": ShopItemType.TAROT, "name": "Hierophant", "price": 3},
                {"item_type": ShopItemType.JOKER, "name": "Bootstraps", "price": 5},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_ARCANA, "price": 6, 
                 "contents": ["Tower", "Moon", "Chariot", "World", "Wheel of Fortune"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.STANDARD, "price": 4, 
                 "contents": ["6 spade", "3 glass hearts", "6 blue stamp spade"]}
            ],
            "medium_blind": [
                {"item_type": ShopItemType.TAROT, "name": "Sun", "price": 3},
                {"item_type": ShopItemType.JOKER, "name": "Clever", "price": 4},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.CELESTIAL, "price": 4, 
                 "contents": ["Mercury", "Saturn", "Uranus"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.MEGA_CELESTIAL, "price": 8, 
                 "contents": ["Mercury", "Saturn", "Uranus", "Pluto", "Neptune"]}
            ],
            "boss_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Mad", "price": 4},
                {"item_type": ShopItemType.TAROT, "name": "Hierophant", "price": 3},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.STANDARD, "price": 4, 
                 "contents": ["K mult heart", "A wild heart", "10 club"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_STANDARD, "price": 6, 
                 "contents": ["9 lucky club", "5 club", "J diamond", "10 spade", "7 glass diamond"]}
            ]
        }
        
        shops[3] = {
            "small_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Wily", "price": 4},
                {"item_type": ShopItemType.JOKER, "name": "Smiley", "price": 4},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_CELESTIAL, "price": 6, 
                 "contents": ["Mars", "Uranus", "Venus", "Earth", "Mercury"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.CELESTIAL, "price": 4, 
                 "contents": ["Mars", "Uranus", "Venus"]}
            ],
            "medium_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Crafty", "price": 4},
                {"item_type": ShopItemType.JOKER, "name": "Cloud 9", "price": 6},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_ARCANA, "price": 6, 
                 "contents": ["Strength", "Fool", "Moon", "Star", "Justice"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.STANDARD, "price": 4, 
                 "contents": ["10 gold diamond", "9 holo club", "J diamond"]}
            ],
            "boss_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Splash", "price": 3},
                {"item_type": ShopItemType.TAROT, "name": "Magician", "price": 3},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.ARCANA, "price": 4, 
                 "contents": ["Justice", "Hanged Man", "Moon"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_CELESTIAL, "price": 6, 
                 "contents": ["Venus", "Earth", "Mars", "Saturn", "Uranus"]}
            ]
        }
        
        shops[4] = {
            "small_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Misprint", "price": 4},
                {"item_type": ShopItemType.TAROT, "name": "Star", "price": 3},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_ARCANA, "price": 6, 
                 "contents": ["Hierophant", "Lovers", "Death", "Justice", "Sun"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.ARCANA, "price": 4, 
                 "contents": ["Moon", "Wheel of Fortune", "Empress"]}
            ],
            "medium_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Wrathful", "price": 5},
                {"item_type": ShopItemType.JOKER, "name": "Cartomancer", "price": 6},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_STANDARD, "price": 6, 
                 "contents": ["9 gold foil heart", "4 holo gold stamp diamond", "6 spade", "J bonus club", "7 glass heart"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.BUFFOON, "price": 4, 
                 "contents": ["Droll Joker"]}
            ],
            "boss_blind": [
                {"item_type": ShopItemType.PLANET, "name": "Pluto", "price": 3},
                {"item_type": ShopItemType.JOKER, "name": "Clever", "price": 4},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.CELESTIAL, "price": 4, 
                 "contents": ["Neptune", "Mercury", "Earth"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.ARCANA, "price": 4, 
                 "contents": ["Tower", "Moon", "Hanged Man"]}
            ]
        }
        
        shops[5] = {
            "small_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Scary Face", "price": 4},
                {"item_type": ShopItemType.JOKER, "name": "Blue", "price": 5},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_BUFFOON, "price": 6, 
                 "contents": ["Even Steven", "Banner", "Walkie Talkie", "Brainstorm"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.ARCANA, "price": 4, 
                 "contents": ["Tower", "Moon", "Hanged Man"]}
            ],
            "medium_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Baseball Card", "price": 8},
                {"item_type": ShopItemType.PLANET, "name": "Mars", "price": 3},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_STANDARD, "price": 6, 
                 "contents": ["10 gold spade", "K spade", "Q heart", "9 gold diamond", "4 gold stamp bonus spade"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.STANDARD, "price": 4, 
                 "contents": ["10 gold spade", "K spade", "Q heart"]}
            ],
            "boss_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Socks and Buskin", "price": 6},
                {"item_type": ShopItemType.JOKER, "name": "8 Ball", "price": 5},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.CELESTIAL, "price": 4, 
                 "contents": ["Neptune", "Pluto", "Black Hole"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_CELESTIAL, "price": 6, 
                 "contents": ["Jupiter", "Uranus", "Mars", "Mercury", "Pluto"]}
            ]
        }
        
        shops[6] = {
            "small_blind": [
                {"item_type": ShopItemType.TAROT, "name": "Chariot", "price": 3},
                {"item_type": ShopItemType.JOKER, "name": "Even Steven", "price": 4},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.ARCANA, "price": 4, 
                 "contents": ["Tower", "World", "Fool"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.CELESTIAL, "price": 4, 
                 "contents": ["Earth", "Jupiter", "Neptune"]}
            ],
            "medium_blind": [
                {"item_type": ShopItemType.TAROT, "name": "Fool", "price": 3},
                {"item_type": ShopItemType.TAROT, "name": "Temperance", "price": 3},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.MEGA_ARCANA, "price": 8, 
                 "contents": ["High Priestess", "Moon", "Death", "Emperor", "Strength"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.ARCANA, "price": 4, 
                 "contents": ["Moon", "Sun", "High Priestess"]}
            ],
            "boss_blind": [
                {"item_type": ShopItemType.JOKER, "name": "The Duo", "price": 8},
                {"item_type": ShopItemType.TAROT, "name": "Temperance", "price": 3},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.ARCANA, "price": 4, 
                 "contents": ["Hermit", "Star", "Death"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.ARCANA, "price": 4, 
                 "contents": ["Sun", "Devil", "World"]}
            ]
        }
        
        shops[7] = {
            "small_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Rocket", "price": 6},
                {"item_type": ShopItemType.JOKER, "name": "Gluttonous", "price": 5},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.STANDARD, "price": 4, 
                 "contents": ["2 heart", "3 steel club", "3 heart"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_STANDARD, "price": 6, 
                 "contents": ["K mult spade", "A blue stamp mult club", "4 wild gold stamp spade", 
                              "9 gold stamp steel spade", "2 club"]}
            ],
            "medium_blind": [
                {"item_type": ShopItemType.PLANET, "name": "Jupiter", "price": 3},
                {"item_type": ShopItemType.TAROT, "name": "Devil", "price": 3},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.STANDARD, "price": 4, 
                 "contents": ["Q steel spade", "3 glass heart", "K diamond"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.STANDARD, "price": 4, 
                 "contents": ["2 gold heart", "J bonus diamond", "A heart"]}
            ],
            "boss_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Droll", "price": 6},
                {"item_type": ShopItemType.JOKER, "name": "Walkie Talkie", "price": 4},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.STANDARD, "price": 4, 
                 "contents": ["10 club", "10 gold stamp heart", "A mult diamond"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_BUFFOON, "price": 6, 
                 "contents": ["Fortune Teller Joker", "Faceless Joker", "Cloud 9", "Business Card Joker"]}
            ]
        }
        
        shops[8] = {
            "small_blind": [
                {"item_type": ShopItemType.PLANET, "name": "Saturn", "price": 3},
                {"item_type": ShopItemType.JOKER, "name": "Blackboard", "price": 6},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.ARCANA, "price": 4, 
                 "contents": ["Judgement", "Strength", "Moon"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.STANDARD, "price": 4, 
                 "contents": ["9 mult glass club", "5 glass spade", "8 club"]}
            ],
            "medium_blind": [
                {"item_type": ShopItemType.JOKER, "name": "Photograph", "price": 5},
                {"item_type": ShopItemType.PLANET, "name": "Neptune", "price": 3},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.JUMBO_ARCANA, "price": 6, 
                 "contents": ["Devil", "High Priestess", "Fool", "Emperor", "Justice"]},
                {"item_type": ShopItemType.BOOSTER, "pack_type": PackType.CELESTIAL, "price": 6, 
                 "contents": ["Pluto", "Earth", "Jupiter", "Mars", "Venus"]}
            ]
        }
        
        return shops
    
    def get_shop_for_ante(self, ante_number: int, shop_type: str) -> List[Dict[str, Any]]:
        """
        Get shop items for a specific ante and shop type
        
        Args:
            ante_number: The ante number (1-8)
            shop_type: The shop type ("small_blind", "medium_blind", or "boss_blind")
            
        Returns:
            A list of shop items for the requested ante and shop type
        """
        if ante_number in self.ante_shops and shop_type in self.ante_shops[ante_number]:
            return self.ante_shops[ante_number][shop_type]
        return []
    
    def create_shop_items(self, ante_number: int, shop_type: str) -> List[ShopItem]:
        """
        Create ShopItem objects for a specific ante and shop type
        
        Args:
            ante_number: The ante number (1-8)
            shop_type: The shop type ("small_blind", "medium_blind", or "boss_blind")
            
        Returns:
            A list of ShopItem objects
        """
        shop_data = self.get_shop_for_ante(ante_number, shop_type)
        shop_items = []
        
        for item_data in shop_data:
            if item_data["item_type"] == ShopItemType.JOKER:
                joker = create_joker(item_data["name"])
                shop_items.append(ShopItem(ShopItemType.JOKER, joker, item_data["price"]))
                
            elif item_data["item_type"] == ShopItemType.TAROT:
                tarot = create_tarot_by_name(item_data["name"])
                shop_items.append(ShopItem(ShopItemType.TAROT, tarot, item_data["price"]))
                
            elif item_data["item_type"] == ShopItemType.PLANET:
                planet = create_planet_by_name(item_data["name"])
                shop_items.append(ShopItem(ShopItemType.PLANET, planet, item_data["price"]))
                
            elif item_data["item_type"] == ShopItemType.BOOSTER:
                booster_name = item_data["pack_type"].value
                shop_items.append(
                    ShopItem(
                        ShopItemType.BOOSTER, 
                        booster_name, 
                        item_data["price"]
                    )
                )
        
        return shop_items
    
class FixedShop(Shop):
    """Extension of the Shop class that uses predefined items from AnteShops"""
    
    def __init__(self, ante_number: int, shop_type: str):
        self.ante_shops = AnteShops()
        self.ante_number = ante_number
        self.shop_type = shop_type
        
        super().__init__()
    
    def _initialize_fixed_shop(self):
        """Initialize shop with predefined items from AnteShops"""
        shop_items = self.ante_shops.create_shop_items(self.ante_number, self.shop_type)
        
        for i, item in enumerate(shop_items):
            if i < len(self.items):
                self.items[i] = item
    
    def restock(self):
        """Override to ensure shop only provides predetermined items"""
        if any(item is None for item in self.items):
            self._initialize_fixed_shop()
    
    def reroll(self, inventory):
        """Override to disable rerolling in fixed shops"""
        return False
    
def initialize_shops_for_game():
    all_shops = {}
    
    for ante in range(1, 9):
        all_shops[ante] = {
            "small_blind": FixedShop(ante, "small_blind"),
            "medium_blind": FixedShop(ante, "medium_blind"),
            "boss_blind": FixedShop(ante, "boss_blind") if ante < 8 else None
        }
    
    return all_shops

def parse_card(card_string: str) -> Card:
    """Parse card string like '3 glass heart' into a Card object with enhancements"""
    parts = card_string.split()
    
    # Map string rank to Rank enum
    rank_map = {
        "A": Rank.ACE, 
        "2": Rank.TWO,
        "3": Rank.THREE,
        "4": Rank.FOUR,
        "5": Rank.FIVE,
        "6": Rank.SIX,
        "7": Rank.SEVEN,
        "8": Rank.EIGHT,
        "9": Rank.NINE,
        "10": Rank.TEN,
        "J": Rank.JACK, 
        "Q": Rank.QUEEN, 
        "K": Rank.KING
    }
    
    rank_str = parts[0]
    rank = rank_map.get(rank_str)
    
    if rank is None:
        try:
            rank_value = int(rank_str)
            for r in Rank:
                if r.value == rank_value:
                    rank = r
                    break
            if rank is None:
                print(f"WARNING: Invalid rank '{rank_str}', defaulting to ACE")
                rank = Rank.ACE
        except ValueError:
            print(f"WARNING: Could not parse rank '{rank_str}', defaulting to ACE")
            rank = Rank.ACE
    
    suit_part = parts[-1].lower()
    suit_map = {
        "heart": Suit.HEARTS,
        "hearts": Suit.HEARTS,
        "diamond": Suit.DIAMONDS,
        "diamonds": Suit.DIAMONDS,
        "club": Suit.CLUBS,
        "clubs": Suit.CLUBS,
        "spade": Suit.SPADES,
        "spades": Suit.SPADES
    }
    suit = suit_map.get(suit_part, Suit.HEARTS)
    if suit_part not in suit_map:
        print(f"WARNING: Invalid suit '{suit_part}', defaulting to HEARTS")
    
    card = Card(suit, rank)
    
    for part in parts[1:-1]:
        part = part.lower()
        if part == "foil":
            card.enhancement = CardEnhancement.FOIL
        elif part == "holo":
            card.enhancement = CardEnhancement.HOLO
        elif part == "poly":
            card.enhancement = CardEnhancement.POLY
        elif part == "gold":
            card.enhancement = CardEnhancement.GOLD
        elif part == "steel":
            card.enhancement = CardEnhancement.STEEL
        elif part == "glass":
            card.enhancement = CardEnhancement.GLASS
        elif part == "stone":
            card.enhancement = CardEnhancement.STONE
        elif part == "wild":
            card.enhancement = CardEnhancement.WILD
        elif part == "mult":
            card.enhancement = CardEnhancement.MULT
        elif part == "bonus":
            card.enhancement = CardEnhancement.BONUS
    
    return card






