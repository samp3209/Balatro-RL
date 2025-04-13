from collections import deque
import itertools
from GameManager import GameManager
import numpy as np
from Inventory import Inventory
from Card import Card
from Enums import *
from Shop import Shop, ShopItem, ShopItemType, initialize_shops_for_game, FixedShop
from HandEvaluator import *


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


class BalatroEnv:
    def __init__(self, config=None):
        self.game_manager = GameManager()
        self.config = {
            'simplified': False,
            'full_features': False
        }
        
        if config:
            self.config.update(config)
        
        self.game_manager = GameManager()
        self.last_action_was_discard = False
        
        self.pending_tarots = []
        
        self.episode_step = 0
        self.episode_max_blind = 0
        
        self.start_new_game()
        self.all_shops = initialize_shops_for_game()
        self.current_shop = None


    def start_new_game(self):
        """Initialize game and shop consistently, similar to GameTest.py"""
        self.game_manager.start_new_game()
        
        if self.config['simplified']:
            self.game_manager.game.is_boss_blind = False
            self.game_manager.max_hands_per_round = 4
            
        self.update_shop()
        
        print(f"\n===== STARTING NEW GAME =====")
        print(f"Current Ante: {self.game_manager.game.current_ante}, Blind: {self.game_manager.game.current_blind}")
        print(f"Money: ${self.game_manager.game.inventory.money}")

    def reset(self):
        """Reset the environment to initial state with a completely fresh game"""
        self.last_action_was_discard = False
        self.episode_step = 0
        self.episode_max_blind = 0
        self.pending_tarots = []
        
        self.game_manager = GameManager()
        
        self.game_manager.start_new_game()
        
        if self.config.get('add_bootstrap', False):
            from JokerCreation import create_joker
            joker = create_joker("Bootstraps")
            if joker:
                self.game_manager.game.inventory.add_joker(joker)
                print(f"Added Bootstraps to inventory")
        
        self.game_manager.game.inventory.money = 10
        
        self.all_shops = initialize_shops_for_game()
        self.current_shop = None
        self.update_shop()
        
        return self._get_play_state()
    
    def update_shop(self):
        """Update the shop for the current ante with improved reliability"""
        current_ante = self.game_manager.game.current_ante
        
        if current_ante > 24:
            print("Game completed! No more shops available.")
            self.current_shop = Shop()
            return
        
        ante_number = ((current_ante - 1) // 3) + 1
        
        blind_index = (current_ante - 1) % 3
        blind_type_map = {
            0: "small_blind",
            1: "medium_blind", 
            2: "boss_blind"
        }
        
        blind_type = blind_type_map[blind_index]
        
        print(f"Looking for shop for Ante {ante_number} ({blind_type})")
        
        if ante_number == 8 and blind_type == "boss_blind":
            print("No boss_blind exists for Ante 8, creating victory shop")
            self.current_shop = Shop()
            self.current_shop.items = [None, None, None, None]
            return
        
        if not hasattr(self, 'all_shops') or self.all_shops is None:
            self.all_shops = initialize_shops_for_game()
        
        shop = None
        if ante_number in self.all_shops:
            if blind_type in self.all_shops[ante_number]:
                shop = self.all_shops[ante_number][blind_type]
        
        if shop is None:
            print(f"Shop not found for Ante {ante_number}, {blind_type}, creating new shop")
            shop = create_shop_for_ante(ante_number, blind_type)
        
        self.current_shop = shop
        
        print(f"Updated shop for Round {current_ante} ({blind_type})")
        print("Shop contents:")
        for i, item in enumerate(self.current_shop.items):
            if item is not None:
                item_name = "Unknown"
                if hasattr(item, 'item_type'):
                    if item.item_type == ShopItemType.JOKER and hasattr(item.item, 'name'):
                        item_name = item.item.name
                    elif item.item_type in [ShopItemType.TAROT, ShopItemType.PLANET] and hasattr(item.item, 'name'):
                        item_name = item.item.name
                    else:
                        item_name = str(item.item)
                price = self.current_shop.get_item_price(i)
                print(f"  {i}: {item_name} - ${price}")
            else:
                print(f"  {i}: [Empty]")

    def step_strategy(self, action):
        """Process a strategy action with IMPROVED reward signals and better item handling"""
        if self.game_manager.game.current_ante > 24:
            print("Game already completed! Starting new game.")
            self.reset()
            return self._get_strategy_state(), 0, False, {"message": "Game completed, starting new game"}

        if self.current_shop is None:
            self.update_shop()
        
        print(f"\n=== SHOP CONTENTS ===")
        for i, item in enumerate(self.current_shop.items):
            if item is not None:
                item_name = "Unknown"
                price = self.current_shop.get_item_price(i)
                if hasattr(item, 'item_type'):
                    if item.item_type == ShopItemType.JOKER and hasattr(item.item, 'name'):
                        item_name = item.item.name
                    elif item.item_type in [ShopItemType.TAROT, ShopItemType.PLANET] and hasattr(item.item, 'name'):
                        item_name = item.item.name
                    else:
                        item_name = str(item.item)
                print(f"{i}: {item_name} - ${price}")
            else:
                print(f"{i}: [Empty]")

        reward = 0
        done = self.game_manager.game_over
        info = {"message": "Unknown action"}
        
        joker_count_before = len(self.game_manager.game.inventory.jokers)
        ante_num = self.game_manager.game.current_ante
        
        money_before = self.game_manager.game.inventory.money
        
        joker_names = set()
        for joker in self.game_manager.game.inventory.jokers:
            if hasattr(joker, 'name'):
                joker_names.add(joker.name)
        joker_diversity = len(joker_names)
        
        min_expected_jokers = min(3, ante_num)
        joker_deficit = max(0, min_expected_jokers - joker_count_before)
        
        planet_count = len([c for c in self.game_manager.game.inventory.consumables 
                            if hasattr(c, 'type') and c.type == ConsumableType.PLANET])
        
        tarot_count = len(self.game_manager.game.inventory.get_consumable_tarot_indices())
        
        if action < 4:
            slot = action
            if self.current_shop and slot < len(self.current_shop.items) and self.current_shop.items[slot] is not None:
                item = self.current_shop.items[slot]
                price = self.current_shop.get_item_price(slot)
                
                if self.game_manager.game.inventory.money >= price:
                    item_name = "Unknown Item"
                    
                    if hasattr(item, 'item_type'):
                        if item.item_type == ShopItemType.JOKER:
                            item_name = item.item.name if hasattr(item.item, 'name') else "Joker"
                            
                            base_reward = 20.0 * (1.0 / (0.3 * ante_num + 0.7))
                            
                            joker_count_scale = 1.0
                            if joker_count_before == 0:
                                joker_count_scale = 10.5  
                            elif joker_count_before < 2:
                                joker_count_scale = 10.5 
                            elif joker_count_before < 4:
                                joker_count_scale = 9.5 
                            elif joker_count_before >= 4:
                                joker_count_scale = 8.0  
                            
                            if joker_count_before == 4:
                                joker_count_scale = 8.0
                            
                            if joker_count_before < min_expected_jokers:
                                deficit_scale = 1.0 + 0.5 * (min_expected_jokers - joker_count_before)
                                joker_count_scale *= deficit_scale
                            
                            joker_exists = item_name in joker_names
                            if not joker_exists:
                                diversity_bonus = 12.0
                                if ante_num >= 3:
                                    diversity_bonus *= 1.3
                                base_reward += diversity_bonus
                            
                            powerful_jokers = [
                                "Bootstraps", "Socks and Buskin",
                                "The Duo", "Rocket", "Blackboard", "Smiley", "Green"
                            ]
                            if hasattr(item.item, 'name') and item.item.name in powerful_jokers:
                                tier_multiplier = 10.5 if ante_num <= 1 else 1.2
                                base_reward += 15.0 * tier_multiplier
                            
                            value_multiplier = 1.0
                            if price <= 3:
                                value_multiplier = 1.8
                            elif price <= 5:
                                value_multiplier = 1.4
                            
                            reward = base_reward * joker_count_scale * value_multiplier
                            
                        elif item.item_type == ShopItemType.PLANET:
                            item_name = item.item.name if hasattr(item.item, 'name') else "Planet"
                            
                            base_reward = 15.0 * (1.0 / (0.7 * ante_num))
                            
                            if planet_count == 0:
                                base_reward *= 1.8
                            
                            planet_type = None
                            if hasattr(item.item, 'planet_type'):
                                planet_type = item.item.planet_type
                                current_level = self.game_manager.game.inventory.planet_levels.get(planet_type, 1)
                                
                                if current_level <= 2:
                                    base_reward *= 1.2
                                elif current_level >= 5:
                                    base_reward *= 0.8
                            
                            if hasattr(item.item, 'name'):
                                if item.item.name in ["jupiter", "earth", "saturn"]:
                                    base_reward += 12.0 
                                elif item.item.name in ["uranus", "neptune", "venus"]:
                                    base_reward += 8.0
                                else:
                                    base_reward += 4.0
                            
                            if ante_num >= 4:
                                base_reward *= 1.2
                                    
                            reward = base_reward
                            
                        elif item.item_type == ShopItemType.TAROT:
                            item_name = item.item.name if hasattr(item.item, 'name') else "Tarot"
                            
                            base_reward = 15.0
                            
                            if tarot_count == 0:
                                base_reward *= 1.7
                            
                            if hasattr(item.item, 'name'):
                                if item.item.name in ["magician", "fool", "high priestess", "justice", "hermit", "judgement"]:
                                    base_reward += 10.0 
                                elif item.item.name in ["hierophant", "chariot", "emperor"]:
                                    base_reward += 6.0 
                            
                            if ante_num >= 3:
                                base_reward *= 1.2
                                
                            self.pending_tarots.append(item_name)
                                    
                            reward = base_reward
                            
                        elif item.item_type == ShopItemType.BOOSTER:
                            item_name = str(item.item) if hasattr(item, 'item') else "Booster"
                            
                            money_ratio = self.game_manager.game.inventory.money / (price * 2)
                            
                            base_reward = 6.0 * min(money_ratio, 1.5)
                            
                            pack_type = str(item.item).upper()
                            
                            if "BUFFOON" in pack_type:
                                base_reward *= 3.8
                                
                                if joker_count_before < min_expected_jokers:
                                    base_reward *= 4.4
                            elif "ARCANA" in pack_type:
                                base_reward *= 3.4
                            elif "CELESTIAL" in pack_type:
                                base_reward *= 3.6
                            
                            if "JUMBO" in pack_type or "MEGA" in pack_type:
                                base_reward *= 5.5  
                                
                            if joker_count_before >= 3:
                                base_reward *= 1.2
                            elif joker_count_before < 1 and ante_num <= 3:
                                base_reward *= 0.5
                            elif joker_count_before < 2 and ante_num <= 3:
                                base_reward *= 0.7 
                                
                            reward = base_reward
                    
                    success = self.current_shop.buy_item(slot, self.game_manager.game.inventory)
                    if success:
                        info['message'] = f"Bought {item_name} for ${price}"
                        print(f"Bought {item_name} for ${price}, reward: {reward}")
                        
                        if item.item_type == ShopItemType.BOOSTER:
                            pack_contents = self.get_shop_item_contents(item)
                            if pack_contents:
                                pack_type = str(item.item)
                                result_message = self.handle_pack_opening(pack_type, pack_contents, 
                                                                        self.game_manager.game.inventory, 
                                                                        self.game_manager)
                                info['message'] += f" {result_message}"
                        
                        elif item.item_type == ShopItemType.PLANET:
                            if hasattr(item.item, 'planet_type'):
                                planet_type = item.item.planet_type
                                current_level = self.game_manager.game.inventory.planet_levels.get(planet_type, 1)
                                
                                planet_indices = self.game_manager.game.inventory.get_consumable_planet_indices()
                                for idx in planet_indices:
                                    consumable = self.game_manager.game.inventory.consumables[idx]
                                    if (hasattr(consumable.item, 'planet_type') and 
                                        consumable.item.planet_type == planet_type):
                                        self.game_manager.game.inventory.remove_consumable(idx)
                                        break
                                
                                self.game_manager.game.inventory.planet_levels[planet_type] = current_level + 1
                                info['message'] += f" Upgraded {item_name} to level {current_level + 1}"
                    else:
                        reward = 0
                else:
                    reward = -0.2
                    info['message'] = f"Not enough money to buy item (costs ${price})"
        
        elif action < 9:  
            joker_idx = action - 4
            if joker_idx < len(self.game_manager.game.inventory.jokers):
                joker = self.game_manager.game.inventory.jokers[joker_idx]
                joker_name = joker.name if hasattr(joker, 'name') else "Unknown Joker"
                
                reward = 0.5
                
                if hasattr(joker, 'sell_value'):
                    sell_value = joker.sell_value
                    
                    if self.game_manager.game.inventory.money < 2:
                        reward = 3.0 
                    elif self.game_manager.game.inventory.money < 4 and ante_num >= 3:
                        reward = 1.5
                    else:
                        tier_penalty = 0.0
                        if sell_value >= 4:
                            tier_penalty = -4.0
                        elif sell_value >= 3:
                            tier_penalty = -2.0 
                            
                        reward += tier_penalty
                    
                    if len(self.game_manager.game.inventory.jokers) > 5:
                        reward += 2.0 
                        
                        all_values = [j.sell_value for j in self.game_manager.game.inventory.jokers if hasattr(j, 'sell_value')]
                        if all_values:
                            avg_value = sum(all_values) / len(all_values)
                            if sell_value < avg_value * 0.8:
                                reward += 2.0
                
                critical_jokers = ["Mr. Bones", "Green Joker", "Bootstraps", "Socks and Buskin", "The Duo", "8 Ball", "Rocket"]
                if hasattr(joker, 'name') and joker.name in critical_jokers:
                    if self.game_manager.game.inventory.money >= 2:
                        reward -= 4.0 
                
                if joker_count_before <= min_expected_jokers:
                    reward -= 3.0 
                
                sell_value = self.current_shop.sell_item("joker", joker_idx, self.game_manager.game.inventory)
                if sell_value > 0:
                    info['message'] = f"Sold {joker_name} for ${sell_value}"
                    print(f"Sold {joker_name} for ${sell_value}, reward: {reward}")
                else:
                    reward = 0
        
        elif action < 15:
            tarot_idx = (action - 9) // 3
            selection_strategy = (action - 9) % 3
            
            tarot_indices = self.game_manager.game.inventory.get_consumable_tarot_indices()
            if tarot_indices and tarot_idx < len(tarot_indices):
                actual_idx = tarot_indices[tarot_idx]
                tarot = self.game_manager.game.inventory.consumables[actual_idx].item
                tarot_name = tarot.name if hasattr(tarot, 'name') else "Unknown Tarot"
                
                selected_indices = []
                if selection_strategy > 0 and self.game_manager.current_hand:
                    if selection_strategy == 1:
                        cards_by_value = [(i, card.rank.value) for i, card in enumerate(self.game_manager.current_hand)]
                        cards_by_value.sort(key=lambda x: x[1])
                        selected_indices = [idx for idx, _ in cards_by_value[:3]]
                    else:
                        cards_by_value = [(i, card.rank.value) for i, card in enumerate(self.game_manager.current_hand)]
                        cards_by_value.sort(key=lambda x: x[1], reverse=True)
                        selected_indices = [idx for idx, _ in cards_by_value[:3]]
                
                success, message = self.game_manager.use_tarot(actual_idx, selected_indices)
                if success:
                    reward = 6.0 
                    
                    if hasattr(tarot, 'name'):
                        if tarot.name in ["Magician", "Devil", "Sun", "Star"]:
                            if selection_strategy == 2:
                                reward += 3.0
                            elif selection_strategy != 2:
                                reward -= 1.5
                                
                        elif tarot.name in ["Tower", "Death", "Moon"]:
                            if selection_strategy == 1:
                                reward += 3.0
                            elif selection_strategy != 1:
                                reward -= 1.5
                    
                    info['message'] = message
                    print(f"Used {tarot_name}: {message}, reward: {reward}")
        
        elif action == 15 and self.game_manager.current_ante_beaten:
            money_efficiency = min(1.0, money_before / 8.0)
            money_penalty = 0.0
            
            if money_before > 12:
                money_penalty = -3.0 
            elif money_before > 8:
                money_penalty = -1.5 
            
            base_reward = 18.0
            
            ante_bonus = self.game_manager.game.current_ante * 3.0
            
            joker_count = len(self.game_manager.game.inventory.jokers)
            
            joker_deficit_penalty = 0.0
            if ante_num >= 3 and joker_count < min_expected_jokers:
                joker_deficit_penalty = -8.0 * (min_expected_jokers - joker_count)
            
            joker_bonus = joker_count * 3.0
            if joker_diversity <= 1 and ante_num >= 2:
                joker_bonus *= 0.6 
            elif joker_diversity <= 2 and ante_num >= 3:
                joker_bonus *= 0.8
            
            money = self.game_manager.game.inventory.money
            money_bonus = min(money / 8.0, 5.0)
            
            planets_tarots_bonus = (planet_count + tarot_count) * 1.5
            
            reward = (base_reward + ante_bonus + joker_bonus + money_bonus + planets_tarots_bonus) * (0.8 + 0.2*money_efficiency) + money_penalty + joker_deficit_penalty
            
            if self.pending_tarots and self.game_manager.current_hand:
                self.use_pending_tarots()
            
            success = self.game_manager.next_ante()
            
            if success:
                info['message'] = f"Advanced to Ante {self.game_manager.game.current_ante}"
                print(f"Successfully advanced to Ante {self.game_manager.game.current_ante}, reward: {reward}")
                
                if not self.game_manager.current_hand:
                    self.game_manager.deal_new_hand()
            else:
                reward = 0
                info['message'] = "Failed to advance to next ante"
                print("Failed to advance to next ante")
        
        next_state = self._get_strategy_state()
        return next_state, reward, done, info

    def get_shop_item_contents(self, shop_item):
        """
        Extract contents directly from a shop item if available
        
        Args:
            shop_item: The shop item object
            
        Returns:
            list: Contents of the item if it's a booster pack, otherwise None
        """
        if shop_item.item_type == ShopItemType.BOOSTER:
            pack_type = str(shop_item.item)
            
            if hasattr(shop_item, 'contents'):
                return shop_item.contents
                
            return self.get_pack_contents(pack_type)
            
        return None


    def get_pack_contents(self, pack_type):
        """
        Get the contents of a pack based on its type
        
        Args:
            pack_type: The type of pack (e.g., "Standard Pack")
            
        Returns:
            list: The contents of the pack
        """
        try:
            pack_enum = None
            for pt in PackType:
                if pt.value == pack_type:
                    pack_enum = pt
                    break
            
            if pack_enum is None:
                print(f"WARNING: Unknown pack type: {pack_type}")
                return []
            
            from Shop import AnteShops
            ante_shops = AnteShops()
            
            for ante_num in range(1, 9):
                if ante_num not in ante_shops.ante_shops:
                    continue
                    
                for blind_type in ["small_blind", "medium_blind", "boss_blind"]:
                    if blind_type not in ante_shops.ante_shops[ante_num]:
                        continue
                        
                    shop_items = ante_shops.ante_shops[ante_num][blind_type]
                    
                    for item in shop_items:
                        if (item.get("item_type") == ShopItemType.BOOSTER and 
                            item.get("pack_type") == pack_enum and
                            "contents" in item):
                            print(f"Found contents for {pack_type}: {item['contents']}")
                            return item["contents"]
            
            print(f"Warning: No contents found for pack type: {pack_type}")
            return []
        except Exception as e:
            print(f"Error in get_pack_contents: {e}")
            return []


    def handle_pack_opening(self, pack_type, pack_contents, inventory, game_manager=None):
        """
        Handle opening a booster pack and selecting items from it
        
        Args:
            pack_type (str): The type of pack (Standard, Celestial, Arcana, etc.)
            pack_contents (list): List of items in the pack
            inventory: The game inventory to add items to
            game_manager: Optional GameManager for tarot card usage
            
        Returns:
            str: Message about what happened with the pack
        """
        from JokerCreation import create_joker
        from Tarot import create_tarot_by_name
        from Planet import create_planet_by_name
        from Card import Card
        from Enums import Suit, Rank, CardEnhancement
        
        print(f"\n=== Opening {pack_type} ===")
        print("Pack contents:")
        
        for i, item in enumerate(pack_contents):
            print(f"{i}: {item}")
        
        if "MEGA" in pack_type.upper():
            num_to_select = 2
        else:
            num_to_select = 1
        
        message = ""
        
        if "STANDARD" in pack_type.upper():
            for i in range(min(num_to_select, len(pack_contents))):
                selected_idx = random.randint(0, len(pack_contents) - 1)
                
                try:
                    card_string = pack_contents[selected_idx]
                    print(f"Processing card string: '{card_string}'")
                    
                    parts = card_string.split()
                    if not parts:
                        print(f"WARNING: Empty card string")
                        continue
                    
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
                    
                    suit_map = {
                        "heart": Suit.HEARTS, 
                        "hearts": Suit.HEARTS, 
                        "♥": Suit.HEARTS,
                        "diamond": Suit.DIAMONDS, 
                        "diamonds": Suit.DIAMONDS, 
                        "♦": Suit.DIAMONDS,
                        "club": Suit.CLUBS, 
                        "clubs": Suit.CLUBS, 
                        "♣": Suit.CLUBS,
                        "spade": Suit.SPADES, 
                        "spades": Suit.SPADES, 
                        "♠": Suit.SPADES
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
                            print(f"WARNING: Invalid rank '{rank_str}', defaulting to ACE")
                            rank = Rank.ACE

                    suit_str = parts[-1].lower() if len(parts) > 1 else "hearts"
                    suit = suit_map.get(suit_str, Suit.HEARTS)
                    if suit_str not in suit_map:
                        print(f"WARNING: Invalid suit '{suit_str}', defaulting to HEARTS")
                    
                    if not isinstance(rank, Rank) or not isinstance(suit, Suit):
                        print(f"ERROR: Invalid rank or suit type - rank: {type(rank)}, suit: {type(suit)}")
                        continue
                        
                    card = Card(suit, rank)
                    print(f"Created card: {card}, rank type: {type(card.rank)}, suit type: {type(card.suit)}")
                    
                    enhancement_map = {
                        "foil": CardEnhancement.FOIL,
                        "holo": CardEnhancement.HOLO,
                        "poly": CardEnhancement.POLY,
                        "wild": CardEnhancement.WILD,
                        "steel": CardEnhancement.STEEL,
                        "glass": CardEnhancement.GLASS,
                        "gold": CardEnhancement.GOLD,
                        "stone": CardEnhancement.STONE,
                        "lucky": CardEnhancement.LUCKY,
                        "mult": CardEnhancement.MULT,
                        "bonus": CardEnhancement.BONUS,
                        "blue": None,
                        "stamp": None
                    }
                    
                    for part in parts[1:-1]:
                        part_lower = part.lower()
                        if part_lower in enhancement_map and enhancement_map[part_lower] is not None:
                            card.enhancement = enhancement_map[part_lower]
                        elif part_lower == "blue" and "stamp" in [p.lower() for p in parts[1:-1]]:
                            card.enhancement = CardEnhancement.FOIL
                    
                    if hasattr(card, 'rank') and hasattr(card, 'suit') and isinstance(card.rank, Rank) and isinstance(card.suit, Suit):
                        inventory.add_card_to_deck(card)
                        message += f"Added {card_string} to deck. "
                        print(f"Successfully added {card_string} to deck")
                    else:
                        error_msg = f"Card validation failed: {str(card.__dict__)}"
                        print(error_msg)
                        message += f"Failed to add card: {error_msg}. "
                    
                except Exception as e:
                    print(f"Error processing card: {e}")
                    message += f"Failed to add card: {e}. "
        
        elif "CELESTIAL" in pack_type.upper():
            for i in range(min(num_to_select, len(pack_contents))):
                selected_idx = random.randint(0, len(pack_contents) - 1)
                planet_name = pack_contents[selected_idx]
                
                try:
                    from Planet import create_planet_by_name
                    planet = create_planet_by_name(planet_name)
                    if planet and hasattr(planet, 'planet_type'):
                        planet_type = planet.planet_type
                        current_level = inventory.planet_levels.get(planet_type, 1)
                        inventory.planet_levels[planet_type] = current_level + 1
                        
                        message += f"Used {planet_name} planet to upgrade to level {current_level + 1}. "
                        print(f"Used {planet_name} planet to upgrade to level {current_level + 1}")
                    else:
                        message += f"Failed to process planet {planet_name}. "
                        print(f"Failed to process planet {planet_name}")
                except Exception as e:
                    print(f"Error processing planet: {e}")
                    message += f"Failed to upgrade planet: {e}. "
        
        elif "ARCANA" in pack_type.upper():
            selected_tarots = []
            
            for i in range(min(num_to_select, len(pack_contents))):
                selected_idx = random.randint(0, len(pack_contents) - 1)
                tarot_name = pack_contents[selected_idx]
                
                try:
                    from Tarot import create_tarot_by_name
                    tarot = create_tarot_by_name(tarot_name)
                    if tarot:
                        inventory.add_consumable(tarot)
                        message += f"Added {tarot_name} tarot to inventory. "
                        print(f"Added {tarot_name} tarot to inventory")
                        
                        selected_tarots.append(tarot_name)
                        
                        self.pending_tarots.append(tarot_name)
                    else:
                        message += f"Failed to create tarot {tarot_name}. "
                        print(f"Failed to create tarot {tarot_name}")
                except Exception as e:
                    print(f"Error processing tarot: {e}")
                    message += f"Failed to add tarot: {e}. "
            
            if selected_tarots and self.game_manager.current_hand:
                if "MEGA" not in pack_type.upper() and len(selected_tarots) > 0:
                    result = self.use_specific_tarot(selected_tarots[0])
                    if result:
                        message += f"Immediately used {selected_tarots[0]} tarot. "
                        if selected_tarots[0] in self.pending_tarots:
                            self.pending_tarots.remove(selected_tarots[0])
                
                elif "MEGA" in pack_type.upper() and len(selected_tarots) >= 2:
                    for i in range(min(2, len(selected_tarots))):
                        result = self.use_specific_tarot(selected_tarots[i])
                        if result:
                            message += f"Immediately used {selected_tarots[i]} tarot. "
                            if selected_tarots[i] in self.pending_tarots:
                                self.pending_tarots.remove(selected_tarots[i])
        
        elif "BUFFOON" in pack_type.upper():
            for i in range(min(num_to_select, len(pack_contents))):
                selected_idx = random.randint(0, len(pack_contents) - 1)
                joker_name = pack_contents[selected_idx]
                
                try:
                    from JokerCreation import create_joker
                    joker = create_joker(joker_name)
                    if joker:
                        if len(inventory.jokers) < 5:
                            inventory.add_joker(joker)
                            message += f"Added {joker_name} joker to inventory. "
                            print(f"Added {joker_name} joker to inventory")
                        else:
                            message += f"No space for joker {joker_name}. "
                            print(f"No space for joker {joker_name}")
                    else:
                        message += f"Failed to create joker {joker_name}. "
                        print(f"Failed to create joker {joker_name}")
                except Exception as e:
                    print(f"Error processing joker: {e}")
                    message += f"Failed to add joker: {e}. "
        
        return message
    

    def use_specific_tarot(self, tarot_name):
        """
        Use a specific tarot card by name
        
        Args:
            tarot_name (str): Name of the tarot to use
            
        Returns:
            bool: True if tarot was used successfully, False otherwise
        """
        tarot_indices = self.game_manager.game.inventory.get_consumable_tarot_indices()
        tarot_index = None
        
        for idx in tarot_indices:
            consumable = self.game_manager.game.inventory.consumables[idx]
            if hasattr(consumable.item, 'name') and consumable.item.name.lower() == tarot_name.lower():
                tarot_index = idx
                break
        
        if tarot_index is None:
            print(f"Could not find tarot {tarot_name} in inventory")
            return False
            
        tarot = self.game_manager.game.inventory.consumables[tarot_index].item
        cards_required = tarot.selected_cards_required if hasattr(tarot, 'selected_cards_required') else 0
        
        if cards_required > len(self.game_manager.current_hand):
            print(f"Not enough cards to use {tarot_name}, needs {cards_required}")
            return False
            
        selected_indices = []
        
        if cards_required > 0:
            if hasattr(tarot, 'name'):
                tarot_lower = tarot.name.lower()
                
                if any(keyword in tarot_lower for keyword in ["magician", "devil", "sun", "star"]):
                    card_values = [(i, card.rank.value) for i, card in enumerate(self.game_manager.current_hand)]
                    card_values.sort(key=lambda x: x[1], reverse=True)
                    selected_indices = [idx for idx, _ in card_values[:cards_required]]
                
                elif any(keyword in tarot_lower for keyword in ["tower", "death", "moon"]):
                    card_values = [(i, card.rank.value) for i, card in enumerate(self.game_manager.current_hand)]
                    card_values.sort(key=lambda x: x[1])
                    selected_indices = [idx for idx, _ in card_values[:cards_required]]
                
                else:
                    all_indices = list(range(len(self.game_manager.current_hand)))
                    random.shuffle(all_indices)
                    selected_indices = all_indices[:cards_required]
            else:
                card_values = [(i, card.rank.value) for i, card in enumerate(self.game_manager.current_hand)]
                card_values.sort(key=lambda x: x[1])
                selected_indices = [idx for idx, _ in card_values[:cards_required]]
        
        success, message = self.game_manager.use_tarot(tarot_index, selected_indices)
        if success:
            print(f"Used {tarot_name}: {message}")
            return True
        else:
            print(f"Failed to use {tarot_name}: {message}")
            return False


    def step_play(self, action):
        """Process a playing action with stricter enforcement of rules"""
        self.episode_step += 1
        
        if (self.game_manager.hands_played >= self.game_manager.max_hands_per_round and 
            self.game_manager.current_score < self.game_manager.game.current_blind and 
            not self.game_manager.current_ante_beaten):
            
            print(f"\n***** GAME OVER: Failed to beat the ante *****")
            print(f"Score: {self.game_manager.current_score}/{self.game_manager.game.current_blind}")
            print(f"Max hands played: {self.game_manager.hands_played}/{self.game_manager.max_hands_per_round}")
            
            mr_bones_index = None
            for i, joker in enumerate(self.game_manager.game.inventory.jokers):
                if joker.name == "Mr. Bones":
                    mr_bones_index = i
                    break
                    
            if mr_bones_index is not None and self.game_manager.current_score >= (self.game_manager.game.current_blind * 0.25):
                self.game_manager.current_ante_beaten = True
                removed_joker = self.game_manager.game.inventory.remove_joker(mr_bones_index)
                print(f"Mr. Bones saved you and vanished!")
                
                next_state = self._get_play_state()
                return next_state, 0.5, False, {"message": "Mr. Bones saved you!", "shop_phase": True}
            else:
                self.game_manager.game_over = True
                next_state = self._get_play_state()
                return next_state, -65.0, True, {"message": "GAME OVER: Failed to beat the ante"}
        
        is_discard = self.is_discard_action(action)
        
        if is_discard and self.game_manager.discards_used >= self.game_manager.max_discards_per_round:
            print(f"WARNING: Attempted to discard but already used max discards: {self.game_manager.discards_used}/{self.game_manager.max_discards_per_round}")
            is_discard = False
            action = action % 256 
        
        card_indices = self._convert_action_to_card_indices(action)
        
        self.last_action_was_discard = is_discard
        
        if self.episode_step % 10 == 0:
            action_type = "Discard" if is_discard else "Play"
            print(f"Step {self.episode_step}: {action_type} action with indices {card_indices}")
            print(f"  Hand size: {len(self.game_manager.current_hand)}, Score: {self.game_manager.current_score}/{self.game_manager.game.current_blind}")
        
        if is_discard:
            success, message = self.game_manager.discard_cards(card_indices)
        else:
            success, message = self.game_manager.play_cards(card_indices)
        
        reward = self._calculate_play_reward()
        
        done = self.game_manager.game_over
        shop_phase = self.game_manager.current_ante_beaten and not done
        
        if shop_phase:
            print(f"\n***** Round {self.game_manager.game.current_ante} BEATEN! *****")
            print(f"Score: {self.game_manager.current_score}/{self.game_manager.game.current_blind}")
            print(f"Moving to shop phase")
        
        next_state = self._get_play_state()
        
        info = {
            "message": message,
            "shop_phase": shop_phase, 
            "current_ante": self.game_manager.game.current_ante,
            "current_blind": self.game_manager.game.current_blind,
            "current_score": self.game_manager.current_score
        }
        
        return next_state, reward, done, info
    

    def _get_play_state(self):
        """Get the current state as a flat numpy array of floats with FIXED SIZE"""
        state_features = []
        
        state_features.append(float(self.game_manager.game.current_ante))
        state_features.append(float(self.game_manager.game.current_blind))
        state_features.append(float(self.game_manager.current_score))
        state_features.append(float(self.game_manager.hands_played))
        state_features.append(float(self.game_manager.max_hands_per_round))
        state_features.append(float(self.game_manager.discards_used))
        state_features.append(float(self.game_manager.max_discards_per_round))
        state_features.append(1.0 if self.game_manager.game.is_boss_blind else 0.0)
        
        has_pair = 0.0
        has_two_pair = 0.0
        has_three_kind = 0.0
        has_straight_potential = 0.0
        has_flush_potential = 0.0
        
        if best_hand_info := self.game_manager.get_best_hand_from_current():
            hand_type, _ = best_hand_info
            if hand_type.value >= HandType.PAIR.value:
                has_pair = 1.0
            if hand_type.value >= HandType.TWO_PAIR.value:
                has_two_pair = 1.0
            if hand_type.value >= HandType.THREE_OF_A_KIND.value:
                has_three_kind = 1.0
            if hand_type.value >= HandType.STRAIGHT.value:
                has_straight_potential = 1.0
            if hand_type.value >= HandType.FLUSH.value:
                has_flush_potential = 1.0
        
        state_features.extend([has_pair, has_two_pair, has_three_kind, 
                            has_straight_potential, has_flush_potential])



        for i in range(8):
            if i < len(self.game_manager.current_hand):
                card = self.game_manager.current_hand[i]
                
                rank_value = float(card.rank.value) / 14.0
                state_features.append(rank_value)
                
                suit_features = [0.0, 0.0, 0.0, 0.0]
                if card.suit == Suit.HEARTS:
                    suit_features[0] = 1.0
                elif card.suit == Suit.DIAMONDS:
                    suit_features[1] = 1.0
                elif card.suit == Suit.CLUBS:
                    suit_features[2] = 1.0
                elif card.suit == Suit.SPADES:
                    suit_features[3] = 1.0
                state_features.extend(suit_features)
                
                enhancement_value = float(card.enhancement.value) / 12.0
                state_features.append(enhancement_value)
                
                state_features.append(1.0 if card.face else 0.0)
                state_features.append(1.0 if getattr(card, 'debuffed', False) else 0.0)
            else:
                state_features.extend([0.0] * 8)
        
        assert len(state_features) == 77, f"Expected 77 features, got {len(state_features)}"
        
        return np.array(state_features, dtype=np.float32)
        
    def _get_strategy_state(self):
        """Get the current strategy state as a flat numpy array"""
        state_features = []
        
        state_features.append(float(self.game_manager.game.current_ante))
        state_features.append(float(self.game_manager.game.current_blind))
        state_features.append(float(self.game_manager.game.inventory.money))
        state_features.append(float(len(self.game_manager.game.inventory.jokers)))
        state_features.append(float(len(self.game_manager.game.inventory.consumables)))
        state_features.append(1.0 if self.game_manager.game.is_boss_blind else 0.0)
        
        boss_effect = [0.0] * len(BossBlindEffect)
        if self.game_manager.game.is_boss_blind and self.game_manager.game.active_boss_blind_effect:
            boss_effect[self.game_manager.game.active_boss_blind_effect.value - 1] = 1.0
        state_features.extend(boss_effect)
        
        joker_features = [0.0] * 10
        for i, joker in enumerate(self.game_manager.game.inventory.jokers[:5]):
            if joker:
                joker_features[i*2] = joker.sell_value / 5.0
                joker_features[i*2+1] = 1.0 if joker.rarity == "uncommon" else 0.5
        state_features.extend(joker_features)
        
        if hasattr(self, 'shop'):
            for i in range(4):
                if i < len(self.shop.items) and self.shop.items[i] is not None:
                    state_features.append(1.0)
                    state_features.append(float(self.shop.get_item_price(i)) / 10.0) 
                    
                    item_type_features = [0.0] * len(ShopItemType)
                    item_type_features[self.shop.items[i].item_type.value - 1] = 1.0
                    state_features.extend(item_type_features)
                else:
                    state_features.append(0.0)
                    state_features.append(0.0)
                    state_features.extend([0.0] * len(ShopItemType))
        else:
            for _ in range(4 * (2 + len(ShopItemType))):
                state_features.append(0.0)
        print(f"Strategy state size: {len(state_features)}")

        return np.array(state_features, dtype=np.float32)
    
    def _define_play_action_space(self):
        return 512 

    def _define_strategy_action_space(self):
        return 26
    
    def _calculate_play_reward(self):
        """Drastically improved reward calculation with ante-specific adjustments"""
        current_ante = self.game_manager.game.current_ante
        
        score_progress = min(1.0, self.game_manager.current_score / self.game_manager.game.current_blind)
        
        ante_factor = max(1.0, 4.0 / current_ante)
        progress_reward = score_progress * 30.0 * ante_factor
        
        if self.game_manager.current_ante_beaten:
            progress_reward += 125.0
        
        hand_quality_reward = 0
        if self.game_manager.hand_result:
            if current_ante <= 3:
                early_hand_map = {
                    HandType.HIGH_CARD: -5.0,
                    HandType.PAIR: -5.0,
                    HandType.TWO_PAIR: 25.0,
                    HandType.THREE_OF_A_KIND: 30.0,
                    HandType.STRAIGHT: 75.0,
                    HandType.FLUSH: 75.0,
                    HandType.FULL_HOUSE: 75.0,
                    HandType.FOUR_OF_A_KIND: 100.0,
                    HandType.STRAIGHT_FLUSH: 150.0
                }
                hand_quality_reward = early_hand_map.get(self.game_manager.hand_result, 0.5)
            else:
                late_hand_map = {
                    HandType.HIGH_CARD: -3.0,
                    HandType.PAIR: -2.0,
                    HandType.TWO_PAIR: 20.0,
                    HandType.THREE_OF_A_KIND: 20.0,
                    HandType.STRAIGHT: 45.0,
                    HandType.FLUSH: 45.0,
                    HandType.FULL_HOUSE: 45.0,
                    HandType.FOUR_OF_A_KIND: 50.0,
                    HandType.STRAIGHT_FLUSH: 90.0
                }
                hand_quality_reward = late_hand_map.get(self.game_manager.hand_result, 0.5)
        
        cards_played = len(self.game_manager.played_cards)
        if cards_played == 5:
            cards_bonus = 4.0 + (cards_played - 5)
        else:
            cards_bonus = 0.5 * cards_played
        
        discard_bonus = 0
        if hasattr(self, 'last_action_was_discard') and self.last_action_was_discard:
            if current_ante <= 3:
                discard_bonus = 1.0  
            else:
                if not self.game_manager.hand_result or self.game_manager.hand_result.value <= HandType.PAIR.value:
                    discard_bonus = 2.0
        
        game_over_penalty = 0
        if self.game_manager.game_over:
            base_penalty = -300.0
            ante_multiplier = max(1.0, 5.0 / current_ante)
            game_over_penalty = base_penalty * ante_multiplier
        
        total_reward = progress_reward + hand_quality_reward + cards_bonus + discard_bonus + game_over_penalty
        
        if self.game_manager.hand_result and self.game_manager.hand_result.value >= HandType.FULL_HOUSE.value:
            multiplier = 1.5 if current_ante <= 3 else 2.0
            total_reward *= multiplier
        
        if current_ante == 1:
            total_reward *= 1.3  
        
        return total_reward
    
    def _calculate_strategy_reward(self):
        ante_progress = 0.2 * self.game_manager.game.current_ante
        
        joker_quality = sum(j.sell_value for j in self.game_manager.game.inventory.jokers) * 0.02
        
        money_reward = 0.01 * self.game_manager.game.inventory.money
        
        planet_level_reward = 0.05 * sum(self.game_manager.game.inventory.planet_levels.values())
        
        game_over_penalty = -5.0 if self.game_manager.game_over else 0
        
        return ante_progress + joker_quality + money_reward + planet_level_reward + game_over_penalty
    
    def get_normalized_state(self, is_shop_phase=False):
        """
        Get a state representation with consistent dimensionality regardless of phase
        """
        if is_shop_phase:
            raw_state = self._get_strategy_state()
        else:
            raw_state = self._get_play_state()
            
        if not isinstance(raw_state, np.ndarray):
            raw_state = np.array(raw_state, dtype=np.float32)
        
        fixed_size = 77
        normalized_state = np.zeros(fixed_size, dtype=np.float32)
        
        normalized_state[:len(raw_state)] = raw_state
        
        return normalized_state


    def _encode_cards(self, cards):
        encoded = []
        for card in cards:
            rank_feature = (card.rank.value - 2) / 12
            
            suit_features = [0, 0, 0, 0]
            suit_features[card.suit.value - 1] = 1
            
            enhancement_feature = card.enhancement.value / len(CardEnhancement)
            
            is_scored = 1.0 if card.scored else 0.0
            is_face = 1.0 if card.face else 0.0
            is_debuffed = 1.0 if hasattr(card, 'debuffed') and card.debuffed else 0.0
            
            card_features = [rank_feature] + suit_features + [enhancement_feature, is_scored, is_face, is_debuffed]
            encoded.extend(card_features)
        
        return np.array(encoded)


    def _encode_boss_blind_effect(self):
        """Encode the boss blind effect as a feature vector."""
        encoded = [0] * (len(BossBlindEffect) + 1)
        
        if not self.game_manager.game.is_boss_blind:
            encoded[0] = 1
        else:
            effect_value = self.game_manager.game.active_boss_blind_effect.value
            encoded[effect_value] = 1
            
        return encoded

    def _encode_joker_effects(self):
        """Encode joker effects as a feature vector."""
        joker_count = len(self.game_manager.game.inventory.jokers)
        
        has_mr_bones = 0
        has_green_joker = 0
        has_socks_buskin = 0
        boss_blind_counters = 0
        
        for joker in self.game_manager.game.inventory.jokers:
            if joker.name == "Mr. Bones":
                has_mr_bones = 1
            elif joker.name == "Green Joker":
                has_green_joker = 1
            elif joker.name == "Socks and Buskin":
                has_socks_buskin = 1
            elif joker.name == "Rocket" and hasattr(joker, "boss_blind_defeated"):
                boss_blind_counters = joker.boss_blind_defeated
        
        return [joker_count, has_mr_bones, has_green_joker, has_socks_buskin, boss_blind_counters]


    def _encode_jokers(self, jokers):
        """
        Encode jokers into a feature vector for the neural network.
        
        Args:
            jokers: List of joker objects
            
        Returns:
            numpy array of encoded joker features
        """
        max_jokers = 5 
        features_per_joker = 6  
        
        encoded = []
        
        for joker in jokers:
            rarity_feature = 0.0
            if joker.rarity == 'base':
                rarity_feature = 0.33
            elif joker.rarity == 'uncommon':
                rarity_feature = 0.66
            elif joker.rarity == 'rare':
                rarity_feature = 1.0
            
            sell_value = joker.sell_value / 5.0
            
            has_mult_effect = 1.0 if joker.mult_effect > 0 else 0.0
            has_chips_effect = 1.0 if joker.chips_effect > 0 else 0.0
            
            has_boss_counter = 1.0 if hasattr(joker, 'boss_blind_defeated') and joker.boss_blind_defeated > 0 else 0.0
            is_retrigger = 1.0 if joker.retrigger else 0.0
            
            joker_features = [rarity_feature, sell_value, has_mult_effect, 
                            has_chips_effect, has_boss_counter, is_retrigger]
            
            encoded.extend(joker_features)
        
        padding_needed = max_jokers - len(jokers)
        if padding_needed > 0:
            encoded.extend([0.0] * (padding_needed * features_per_joker))
        
        return np.array(encoded)

    def _encode_consumables(self):
        """
        Encode consumables (tarot and planet cards) into a feature vector.
        
        Returns:
            numpy array of encoded consumable features
        """
        max_consumables = 2 
        features_per_consumable = 5
        
        encoded = []
        
        tarot_indices = self.game_manager.game.inventory.get_consumable_tarot_indices()
        planet_indices = self.game_manager.game.inventory.get_consumable_planet_indices()
        
        for idx in range(len(self.game_manager.game.inventory.consumables)):
            if idx >= max_consumables:
                break
                
            consumable = self.game_manager.game.inventory.consumables[idx]
            
            is_planet = 1.0 if consumable.type == ConsumableType.PLANET else 0.0
            
            cards_required = 0.0
            if consumable.type == ConsumableType.TAROT and hasattr(consumable.item, 'selected_cards_required'):
                cards_required = consumable.item.selected_cards_required / 3.0
            
            affects_cards = 0.0
            affects_money = 0.0
            affects_game = 0.0
            
            if consumable.type == ConsumableType.TAROT:
                tarot_type = consumable.item.tarot_type
                
                if tarot_type in [TarotType.THE_MAGICIAN, TarotType.THE_EMPRESS, 
                                TarotType.THE_HIEROPHANT, TarotType.THE_LOVERS,
                                TarotType.THE_CHARIOT, TarotType.JUSTICE, 
                                TarotType.STRENGTH, TarotType.THE_DEVIL, 
                                TarotType.THE_TOWER, TarotType.THE_STAR,
                                TarotType.THE_MOON, TarotType.THE_SUN,
                                TarotType.THE_WORLD]:
                    affects_cards = 1.0
                
                if tarot_type in [TarotType.THE_HERMIT, TarotType.TEMPERANCE]:
                    affects_money = 1.0
                    
                if tarot_type in [TarotType.THE_FOOL, TarotType.THE_HIGH_PRIESTESS,
                                TarotType.THE_EMPEROR, TarotType.WHEEL_OF_FORTUNE,
                                TarotType.JUDGEMENT]:
                    affects_game = 1.0
            
            elif consumable.type == ConsumableType.PLANET:
                affects_game = 1.0
            
            consumable_features = [is_planet, cards_required, affects_cards, 
                                affects_money, affects_game]
            
            encoded.extend(consumable_features)
        
        padding_needed = max_consumables - len(self.game_manager.game.inventory.consumables)
        if padding_needed > 0:
            encoded.extend([0.0] * (padding_needed * features_per_consumable))
        
        return np.array(encoded)

    def _encode_shop_items(self):
        """
        Encode shop items into a feature vector.
        
        Returns:
            numpy array of encoded shop item features
        """

        max_shop_items = 4 
        features_per_item = 6 
        
        encoded = []
        
        shop = None
        if hasattr(self, 'shop'):
            shop = self.shop
        elif hasattr(self.game_manager, 'shop'):
            shop = self.game_manager.shop
        
        if shop is None:
            return np.zeros(max_shop_items * features_per_item)
        
        for i in range(max_shop_items):
            has_item = 1.0 if i < len(shop.items) and shop.items[i] is not None else 0.0
            
            item_type = 0.0
            price = 0.0
            can_afford = 0.0
            is_joker = 0.0
            is_consumable = 0.0
            
            if has_item:
                shop_item = shop.items[i]
                
                item_type = shop_item.item_type.value / len(ShopItemType)
                
                price = shop.get_item_price(i) / 10.0
                can_afford = 1.0 if self.game_manager.game.inventory.money >= price else 0.0
                
                is_joker = 1.0 if shop_item.item_type == ShopItemType.JOKER else 0.0
                is_consumable = 1.0 if shop_item.item_type in [ShopItemType.TAROT, ShopItemType.PLANET] else 0.0
            
            item_features = [has_item, item_type, price, can_afford, is_joker, is_consumable]
            encoded.extend(item_features)
        
        return np.array(encoded)

    def _encode_planet_levels(self):
        """
        Encode planet levels into a feature vector.
        
        Returns:
            numpy array of encoded planet level features
        """
        encoded = []
        
        for planet_type in PlanetType:
            level = self.game_manager.game.inventory.planet_levels.get(planet_type, 1)
            
            normalized_level = (level - 1) / 9.0
            
            encoded.append(normalized_level)
        
        return np.array(encoded)
    

    def use_pending_tarots(self, limit=None):
        """
        Use tarot cards that were purchased from the shop
        
        Args:
            limit (int, optional): Maximum number of tarots to use
        """
        if not self.pending_tarots:
            return False
        
        print("\n=== Using Tarot Cards From Shop ===")
        used_any = False
        used_count = 0
        
        for tarot_name in self.pending_tarots.copy():
            if limit is not None and used_count >= limit:
                break
            tarot_indices = self.game_manager.game.inventory.get_consumable_tarot_indices()
            tarot_index = None
            
            for idx in tarot_indices:
                consumable = self.game_manager.game.inventory.consumables[idx]
                if hasattr(consumable.item, 'name') and consumable.item.name.lower() == tarot_name.lower():
                    tarot_index = idx
                    break
            
            if tarot_index is None:
                print(f"Could not find tarot {tarot_name} in inventory")
                continue
                
            tarot = self.game_manager.game.inventory.consumables[tarot_index].item
            cards_required = tarot.selected_cards_required if hasattr(tarot, 'selected_cards_required') else 0
            
            if cards_required > len(self.game_manager.current_hand):
                print(f"Not enough cards to use {tarot_name}, needs {cards_required}")
                continue
                
            selected_indices = []
            
            if cards_required > 0:
                card_values = [(i, card.rank.value) for i, card in enumerate(self.game_manager.current_hand)]
                card_values.sort(key=lambda x: x[1])
                selected_indices = [idx for idx, _ in card_values[:cards_required]]
            
            success, message = self.game_manager.use_tarot(tarot_index, selected_indices)
            if success:
                print(f"Used {tarot_name}: {message}")
                self.pending_tarots.remove(tarot_name)
                used_any = True
            else:
                print(f"Failed to use {tarot_name}: {message}")
        
        return used_any



    def get_valid_play_actions(self):
        """Return valid play actions with proper handling for end-of-round cases"""
        valid_actions = []
        
        hands_limit_reached = self.game_manager.hands_played >= self.game_manager.max_hands_per_round
        ante_beaten = self.game_manager.current_ante_beaten
        

        if hands_limit_reached and not ante_beaten:
            print("No valid actions: max hands played and ante not beaten")

            all_cards_action = (1 << len(self.game_manager.current_hand)) - 1
            return [all_cards_action]
        
        if not hands_limit_reached:
            best_hand_info = self.game_manager.get_best_hand_from_current()
            
            if best_hand_info:
                best_hand, best_cards = best_hand_info
                
                if best_hand.value >= HandType.PAIR.value:
                    recommended_indices = []
                    for card in best_cards:
                        for i, hand_card in enumerate(self.game_manager.current_hand):
                            if hasattr(hand_card, 'rank') and hasattr(card, 'rank') and \
                            hasattr(hand_card, 'suit') and hasattr(card, 'suit') and \
                            hand_card.rank == card.rank and hand_card.suit == card.suit:
                                recommended_indices.append(i)
                                break
                    
                    if recommended_indices:
                        action = self._indices_to_action(recommended_indices, is_discard=False)
                        valid_actions.append(action)
                        
                        for _ in range(3):
                            valid_actions.append(action)
            
            for i in range(min(256, 2**len(self.game_manager.current_hand))):
                if self._is_valid_play_action(i):
                    valid_actions.append(i)
        
        if self.game_manager.discards_used < self.game_manager.max_discards_per_round:
            best_hand_info = self.game_manager.get_best_hand_from_current()
            if not best_hand_info or best_hand_info[0].value <= HandType.PAIR.value:
                for i in range(min(256, 2**len(self.game_manager.current_hand))):
                    if self._is_valid_discard_action(i):
                        valid_actions.append(i + 256)
        
        if not valid_actions and len(self.game_manager.current_hand) > 0:
            print("Warning: No valid actions found with cards in hand. Defaulting to play all.")
            all_cards_action = (1 << len(self.game_manager.current_hand)) - 1
            valid_actions.append(all_cards_action)
        
        return valid_actions

    def _is_valid_play_action(self, action):
        """Check if a play action is valid (has at least one card selected)"""
        card_indices = self._convert_action_to_card_indices(action)
        
        if len(card_indices) == 0:
            return False
        
        if self.game_manager.hands_played >= self.game_manager.max_hands_per_round:
            return False
        
        return True

    def _is_valid_discard_action(self, action):
        """Check if a discard action is valid (has at least one card selected)"""
        card_indices = self._convert_action_to_card_indices(action)
        
        if len(card_indices) == 0:
            return False
        
        if self.game_manager.discards_used >= self.game_manager.max_discards_per_round:
            return False
        
        return True

    def _indices_to_action(self, indices, is_discard=False):
        """Convert a list of card indices to an action number"""
        action = 0
        for idx in indices:
            action |= (1 << idx)
        
        if is_discard:
            action += 256
            
        return action

    def get_valid_strategy_actions(self):
        """Return valid strategy actions based on current game state with better prioritization"""
        valid_actions = []
        
        if not hasattr(self, 'current_shop') or self.current_shop is None:
            self.update_shop()
        
        joker_slots = []
        for i in range(4):
            if i < len(self.current_shop.items) and self.current_shop.items[i] is not None:
                item = self.current_shop.items[i]
                if (hasattr(item, 'item_type') and 
                    item.item_type == ShopItemType.JOKER and
                    self.game_manager.game.inventory.money >= self.current_shop.get_item_price(i)):
                    joker_slots.append(i)
        
        planet_slots = []
        for i in range(4):
            if i < len(self.current_shop.items) and self.current_shop.items[i] is not None:
                item = self.current_shop.items[i]
                if (hasattr(item, 'item_type') and 
                    item.item_type == ShopItemType.PLANET and
                    self.game_manager.game.inventory.money >= self.current_shop.get_item_price(i)):
                    planet_slots.append(i)
        
        tarot_slots = []
        for i in range(4):
            if i < len(self.current_shop.items) and self.current_shop.items[i] is not None:
                item = self.current_shop.items[i]
                if (hasattr(item, 'item_type') and 
                    item.item_type == ShopItemType.TAROT and
                    self.game_manager.game.inventory.money >= self.current_shop.get_item_price(i)):
                    tarot_slots.append(i)
        
        booster_slots = []
        for i in range(4):
            if i < len(self.current_shop.items) and self.current_shop.items[i] is not None:
                item = self.current_shop.items[i]
                if (hasattr(item, 'item_type') and 
                    item.item_type == ShopItemType.BOOSTER and
                    self.game_manager.game.inventory.money >= self.current_shop.get_item_price(i)):
                    booster_slots.append(i)
        
        valid_actions.extend(joker_slots)
        valid_actions.extend(planet_slots)
        valid_actions.extend(tarot_slots)
        valid_actions.extend(booster_slots)
        
        joker_count = len(self.game_manager.game.inventory.jokers)
        if joker_count > 0:
            if joker_count >= 4 or self.game_manager.game.inventory.money <= 1:
                for i in range(min(joker_count, 5)):
                    valid_actions.append(i + 4)
        
        tarot_indices = self.game_manager.game.inventory.get_consumable_tarot_indices()
        for i, tarot_idx in enumerate(tarot_indices):
            if i < 2: 
                valid_actions.append(9 + i*3) 
                valid_actions.append(10 + i*3)  
                valid_actions.append(11 + i*3)  
        

        if self.game_manager.current_ante_beaten:
            valid_actions.append(15)
        
        if not valid_actions:
            valid_actions.append(15)
        
        return valid_actions

    def _convert_action_to_card_indices(self, action):
        """
        Convert an action integer into a list of card indices to play or discard
        
        Args:
            action: Integer representing the action
            
        Returns:
            List of card indices to play/discard
        """
        action_type = action // 256
        
        card_mask = action % 256
        
        binary = format(card_mask, '08b')
        
        card_indices = [i for i, bit in enumerate(reversed(binary)) if bit == '1']
        
        return card_indices

    def is_discard_action(self, action):
        """
        Check if an action is a discard (vs play) action
        
        Args:
            action: Integer representing the action
            
        Returns:
            Boolean indicating if this is a discard action
        """
        return action >= 256




class PlayingAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95                  
        self.epsilon = 1.0                 
        self.epsilon_min = 0.01            
        self.epsilon_decay = 0.995         
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.recent_rewards = deque(maxlen=100)
        self.recent_hands = deque(maxlen=100) 
        self.recent_discards = deque(maxlen=100) 
        
        self.PLAY_ACTION = 0
        self.DISCARD_ACTION = 1
        
    def _build_model(self):
        """Build a more sophisticated neural network for predicting Q-values"""
        model = Sequential()
        
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.0005))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        self.recent_rewards.append(reward)
        
    def act(self, state, valid_actions=None):
        """Choose an action with robust input validation"""
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
            
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        if state.shape[1] != self.state_size:
            print(f"WARNING: Play agent received incorrect state size. Expected {self.state_size}, got {state.shape[1]}")
            if state.shape[1] < self.state_size:
                padded_state = np.zeros((1, self.state_size), dtype=np.float32)
                padded_state[0, :state.shape[1]] = state[0, :]
                state = padded_state
            else:
                state = state[:, :self.state_size]
        
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        
        if valid_actions is not None and len(valid_actions) > 0:
            mask = np.full(self.action_size, -1e6)
            for action in valid_actions:
                mask[action] = 0
            act_values = act_values + mask
        
        return np.argmax(act_values[0])
    
    def decode_action(self, action):
        """Decode an action into play/discard and card indices"""
        action_type = action // 256
        
        card_mask = action % 256
        
        binary = format(card_mask, '08b') 
        card_indices = [i for i, bit in enumerate(reversed(binary)) if bit == '1']
        
        action_name = "Discard" if action_type == self.DISCARD_ACTION else "Play"
        return action_type, card_indices, f"{action_name} cards {card_indices}"
        
    def replay(self, batch_size):
        """Improved learning strategy with better experience prioritization"""
        if len(self.memory) < batch_size:
            return
        
        priorities = []
        for state, action, reward, next_state, done in self.memory:
            priority = abs(reward) + 0.1 
            
            if reward > 20: 
                priority *= 4.0 
            elif reward > 10:  
                priority *= 2.0
                
            priorities.append(priority)
        
        priorities = np.array(priorities, dtype=np.float64) 
        total_priority = np.sum(priorities)
        
        if total_priority <= 0 or np.isnan(total_priority) or np.isinf(total_priority):
            print("Warning: Problem with priorities, using uniform distribution")
            probabilities = np.ones(len(priorities)) / len(priorities)
        else:
            probabilities = priorities / total_priority
            
            if not np.isclose(np.sum(probabilities), 1.0):
                probabilities = probabilities / np.sum(probabilities)
        
        try:
            indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        except ValueError:
            print("Warning: Sampling error, using random sampling")
            indices = np.random.choice(len(self.memory), batch_size)
        
        minibatch = [self.memory[idx] for idx in indices]
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for experience in minibatch:
            state, action, reward, next_state, done = experience
            
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)
            
            if len(state.shape) > 1:
                state = state.flatten()
                
            if len(next_state.shape) > 1:
                next_state = next_state.flatten()
                
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        

        max_state_length = max(len(state) for state in states)
        max_next_state_length = max(len(next_state) for next_state in next_states)
        
        for i in range(len(states)):
            if len(states[i]) < max_state_length:
                states[i] = np.pad(states[i], (0, max_state_length - len(states[i])), 'constant')
            if len(next_states[i]) < max_next_state_length:
                next_states[i] = np.pad(next_states[i], (0, max_next_state_length - len(next_states[i])), 'constant')
        
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        targets = self.model.predict(states, verbose=0)
        
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        return history.history['loss'][0]
    
    def decay_epsilon(self):
        """Reduce exploration rate over time"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, file_path):
        """Save model to file"""
        self.model.save(file_path)
    
    def load_model(self, file_path):
        """Load model from file"""
        self.model = tf.keras.models.load_model(file_path)
        self.update_target_model()
    
    def get_stats(self):
        """Return performance statistics"""
        if not self.recent_rewards:
            return {"avg_reward": 0}
            
        return {
            "avg_reward": sum(self.recent_rewards) / len(self.recent_rewards),
            "epsilon": self.epsilon,
            "memory_size": len(self.memory)
        }
    
    def track_hand_played(self, hand_type, cards_played):
        """Track what hands the agent is playing"""
        self.recent_hands.append((hand_type, len(cards_played)))
    
    def track_discard(self, cards_discarded):
        """Track discards for analysis"""
        self.recent_discards.append(len(cards_discarded))
    
        
    def evaluate_all_possible_plays(self, hand, evaluator):
        """
        Evaluate all possible card combinations to find the best hand
        This can be used to supplement RL training with expert demonstrations
        """
        best_score = 0
        best_cards = []
        best_hand_type = None
        
        n = len(hand)
        for i in range(1, 2**n):
            binary = bin(i)[2:].zfill(n)
            selected_cards = [hand[j] for j in range(n) if binary[j] == '1']
            
            if len(selected_cards) < 5:
                continue
                
            hand_type, _, scoring_cards = evaluator.evaluate_hand(selected_cards)
            
            score = hand_type.value * len(scoring_cards)
            
            if score > best_score:
                best_score = score
                best_cards = selected_cards
                best_hand_type = hand_type
        
        return best_cards, best_hand_type, best_score
    
    def adaptive_exploration(self, state, valid_actions, episode):
        """
        More sophisticated exploration strategy that adapts based on 
        training progress and state characteristics
        """
        current_epsilon = self.epsilon
        
        episode_factor = max(0.1, min(1.0, 5000 / (episode + 5000)))
        
        hand_quality = self._estimate_hand_quality(state)
        quality_factor = max(0.5, 1.5 - hand_quality)
        
        adaptive_epsilon = current_epsilon * episode_factor * quality_factor
        
        if np.random.rand() <= adaptive_epsilon:
            return random.choice(valid_actions)
        
        state_array = np.array(state).reshape(1, -1)
        act_values = self.model.predict(state_array, verbose=0)
        
        mask = np.ones(self.action_size) * -1000000
        for action in valid_actions:
            mask[action] = 0
        act_values = act_values + mask
        
        return np.argmax(act_values[0])
    
    def _estimate_hand_quality(self, state):
        """Estimate the quality of the current hand (helper for adaptive_exploration)"""
        if hasattr(self, 'game_manager') and self.game_manager.current_hand:
            best_hand_info = self.game_manager.get_best_hand_from_current()
            if best_hand_info:
                hand_type, _ = best_hand_info
                
                hand_quality_map = {
                    HandType.HIGH_CARD: 0.1,
                    HandType.PAIR: 0.3,
                    HandType.TWO_PAIR: 0.5,
                    HandType.THREE_OF_A_KIND: 0.6,
                    HandType.STRAIGHT: 0.7,
                    HandType.FLUSH: 0.8,
                    HandType.FULL_HOUSE: 0.85,
                    HandType.FOUR_OF_A_KIND: 0.9,
                    HandType.STRAIGHT_FLUSH: 1.0
                }
                
                return hand_quality_map.get(hand_type, 0.1)
            
        return 0.5
        
    def prioritized_replay(self, batch_size):
        """Prioritized Experience Replay implementation"""
        if len(self.memory) < batch_size:
            return
        
        td_errors = []
        for state, action, reward, next_state, done in self.memory:
            state_array = np.array(state).reshape(1, -1)
            next_state_array = np.array(next_state).reshape(1, -1)
            
            current_q = self.model.predict(state_array, verbose=0)[0][action]
            
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * np.max(self.target_model.predict(next_state_array, verbose=0)[0])
            
            td_error = abs(target_q - current_q)
            td_errors.append(td_error)
        
        probabilities = np.array(td_errors) ** 0.6
        probabilities = probabilities / np.sum(probabilities)
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            states.append(self.memory[idx][0])
            actions.append(self.memory[idx][1])
            rewards.append(self.memory[idx][2])
            next_states.append(self.memory[idx][3])
            dones.append(self.memory[idx][4])
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)

class StrategyAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000) 
        self.gamma = 0.99                 
        self.epsilon = 1.0                 
        self.epsilon_min = 0.05            
        self.epsilon_decay = 0.998         
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.recent_rewards = deque(maxlen=100)
        self.recent_actions = deque(maxlen=100)
        
        self.action_map = {
            0: "Buy Shop Item 0",
            1: "Buy Shop Item 1",
            2: "Buy Shop Item 2",
            3: "Buy Shop Item 3",
            4: "Sell Joker 0",
            5: "Sell Joker 1",
            6: "Sell Joker 2",
            7: "Sell Joker 3",
            8: "Sell Joker 4",
            9: "Use Tarot 0 with no cards",
            10: "Use Tarot 0 with lowest cards",
            11: "Use Tarot 0 with highest cards",
            12: "Use Tarot 1 with no cards",
            13: "Use Tarot 1 with lowest cards",
            14: "Use Tarot 1 with highest cards",
            15: "Skip (Do Nothing)"
        }
        
    def _build_model(self):
        """Build a deeper neural network for predicting Q-values"""
        model = Sequential()
        
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory with robust state size handling"""
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        
        if len(state.shape) > 1:
            state = state.flatten()
        if len(next_state.shape) > 1:
            next_state = next_state.flatten()
        
        if len(state) != self.state_size:
            padded_state = np.zeros(self.state_size, dtype=np.float32)
            min_size = min(len(state), self.state_size)
            padded_state[:min_size] = state[:min_size]
            state = padded_state
            
        if len(next_state) != self.state_size:
            padded_next_state = np.zeros(self.state_size, dtype=np.float32)
            min_size = min(len(next_state), self.state_size)
            padded_next_state[:min_size] = next_state[:min_size]
            next_state = padded_next_state
        
        self.memory.append((state, action, reward, next_state, done))
        
        self.recent_rewards.append(reward)
    
    def act(self, state, valid_actions=None):
        """Choose an action with robust state size handling"""
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
            
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        if state.shape[1] != self.state_size:
            print(f"WARNING: Strategy agent handling state size mismatch: got {state.shape[1]}, expected {self.state_size}")
            
            padded_state = np.zeros((1, self.state_size), dtype=np.float32)
            
            min_size = min(state.shape[1], self.state_size)
            padded_state[0, :min_size] = state[0, :min_size]
            
            state = padded_state
        
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None:
                return random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        
        if valid_actions is not None:
            mask = np.ones(self.action_size) * -1000000
            for action in valid_actions:
                mask[action] = 0
            act_values = act_values + mask
        
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train the agent with experiences from memory with size adaptation"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        expected_size = 77  
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in minibatch:
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)
            
            if len(state.shape) > 1:
                state = state.flatten()
            if len(next_state.shape) > 1:
                next_state = next_state.flatten()
            
            if len(state) != expected_size:
                padded_state = np.zeros(expected_size, dtype=np.float32)
                padded_state[:len(state)] = state
                state = padded_state
                
            if len(next_state) != expected_size:
                padded_next_state = np.zeros(expected_size, dtype=np.float32)
                padded_next_state[:len(next_state)] = next_state
                next_state = padded_next_state
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
    
    def act_with_explanation(self, state, valid_actions=None):
        """Choose an action and provide explanation (for debugging)"""
        action = self.act(state, valid_actions)
        explanation = self.action_map.get(action, f"Unknown action {action}")
        
        state_array = np.array(state).reshape(1, -1)
        q_values = self.model.predict(state_array, verbose=0)[0]
        
        return action, explanation, q_values
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, file_path):
        """Save the model to disk"""
        self.model.save(file_path)
    
    def load_model(self, file_path):
        """Load model from file with compatibility check"""
        try:
            loaded_model = tf.keras.models.load_model(file_path)
            
            expected_input_shape = (None, self.state_size)
            actual_input_shape = loaded_model.layers[0].input_shape
            
            if actual_input_shape[1] != self.state_size:
                print(f"Warning: Loaded model expects input dimension {actual_input_shape[1]}, " +
                    f"but current state size is {self.state_size}. Creating new model.")
                return
                
            self.model = loaded_model
            self.update_target_model()
            print(f"Successfully loaded model from {file_path}")
        except Exception as e:
            print(f"Error loading model: {e}. Creating new model.")
    
    def get_stats(self):
        """Return current performance stats"""
        if not self.recent_rewards:
            return {"avg_reward": 0, "action_distribution": {}}
        
        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
        
        action_counts = {}
        for action in self.recent_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        action_distribution = {
            self.action_map.get(action, f"Action {action}"): count / len(self.recent_actions) * 100
            for action, count in action_counts.items()
        }
        
        return {
            "avg_reward": avg_reward,
            "action_distribution": action_distribution,
            "epsilon": self.epsilon
        }

    def prioritized_strategy_replay(self, batch_size):
        """Train the agent with prioritized replay focusing on successful purchases"""
        if len(self.memory) < batch_size:
            return
        
        priorities = []
        for state, action, reward, next_state, done in self.memory:
            priority = abs(reward) + 0.1  
            
            joker_count = 0
            if isinstance(state, np.ndarray) and len(state) > 3:
                joker_count = state[3]
            
            if action < 4 and reward > 0:
                priority *= 5.0  
                
                if joker_count < 3:
                    priority *= 1.5
                elif joker_count >= 3 and joker_count < 5:
                    priority *= 2.0
                    
            if action == 15 and reward > 0:
                priority *= 1.8
                
            priorities.append(priority)
        
        total_priority = sum(priorities)
        if total_priority > 0:
            probabilities = [p / total_priority for p in priorities]
        else:
            probabilities = [1.0 / len(priorities)] * len(priorities)
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        states = np.array([self.memory[i][0] for i in indices])
        actions = np.array([self.memory[i][1] for i in indices])
        rewards = np.array([self.memory[i][2] for i in indices])
        next_states = np.array([self.memory[i][3] for i in indices])
        dones = np.array([self.memory[i][4] for i in indices])
        
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        return


    def prioritized_replay(self, batch_size, alpha=0.6, beta=0.4):
        if len(self.memory) < batch_size:
            return
        
        td_errors = []
        for state, action, reward, next_state, done in self.memory:
            state_array = np.array(state).reshape(1, -1)
            next_state_array = np.array(next_state).reshape(1, -1)
            
            current_q = self.model.predict(state_array, verbose=0)[0][action]
            if done:
                target_q = reward
            else:
                next_q = np.max(self.target_model.predict(next_state_array, verbose=0)[0])
                target_q = reward + self.gamma * next_q
            
            td_error = abs(target_q - current_q)
            td_errors.append(td_error)
        
        priorities = np.power(np.array(td_errors) + 1e-6, alpha)
        probs = priorities / np.sum(priorities)
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        states = np.array([self.memory[i][0] for i in indices])
        actions = np.array([self.memory[i][1] for i in indices])
        rewards = np.array([self.memory[i][2] for i in indices])
        next_states = np.array([self.memory[i][3] for i in indices])
        dones = np.array([self.memory[i][4] for i in indices])
        
        weights = np.power(len(self.memory) * probs[indices], -beta)
        weights /= np.max(weights)
        
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        history = self.model.fit(states, targets, epochs=1, verbose=0, 
                                sample_weight=weights)
        
        return history.history['loss'][0]


def use_pending_tarots(self):
    """
    Use tarot cards that were purchased from the shop, similar to GameTest.py
    """
    if not self.pending_tarots:
        return False
    
    print("\n=== Using Tarot Cards From Shop ===")
    used_any = False
    
    for tarot_name in self.pending_tarots.copy():
        tarot_indices = self.game_manager.game.inventory.get_consumable_tarot_indices()
        tarot_index = None
        
        for idx in tarot_indices:
            consumable = self.game_manager.game.inventory.consumables[idx]
            if hasattr(consumable.item, 'name') and consumable.item.name.lower() == tarot_name.lower():
                tarot_index = idx
                break
        
        if tarot_index is None:
            print(f"Could not find tarot {tarot_name} in inventory")
            continue
            
        tarot = self.game_manager.game.inventory.consumables[tarot_index].item
        cards_required = tarot.selected_cards_required if hasattr(tarot, 'selected_cards_required') else 0
        
        if cards_required > len(self.game_manager.current_hand):
            print(f"Not enough cards to use {tarot_name}, needs {cards_required}")
            continue
            
        selected_indices = []
        
        if cards_required > 0:
            card_values = [(i, card.rank.value) for i, card in enumerate(self.game_manager.current_hand)]
            card_values.sort(key=lambda x: x[1])
            selected_indices = [idx for idx, _ in card_values[:cards_required]]
        
        success, message = self.game_manager.use_tarot(tarot_index, selected_indices)
        if success:
            print(f"Used {tarot_name}: {message}")
            self.pending_tarots.remove(tarot_name)
            used_any = True
        else:
            print(f"Failed to use {tarot_name}: {message}")
    
    return used_any




def create_shop_for_ante(ante_number, blind_type):
    """Create a new shop with appropriate items for the specified ante and blind type"""
    print(f"Creating new shop for Ante {ante_number}, {blind_type}")
    
    try:
        from Shop import FixedShop
        shop = FixedShop(ante_number, blind_type)
        
        has_items = False
        for item in shop.items:
            if item is not None:
                has_items = True
                break
        
        if has_items:
            return shop
    except Exception as e:
        print(f"Failed to create FixedShop: {e}")
    

def add_demonstration_examples(play_agent, num_examples=300):
    """Add expert demonstration examples to the agent's memory with better poker hand recognition"""
    env = BalatroEnv(config={'simplified': True})
    hand_evaluator = HandEvaluator()
    
    print(f"Adding {num_examples} demonstration examples...")
    examples_added = 0
    
    for _ in range(num_examples):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 50:
            steps += 1
            
            potential_hands = []
            current_hand = env.game_manager.current_hand
            
            for r in range(5, len(current_hand) + 1):
                for combo in itertools.combinations(range(len(current_hand)), r):
                    cards = [current_hand[i] for i in combo]
                    hand_type, _, scoring_cards = hand_evaluator.evaluate_hand(cards)
                    hand_value = hand_type.value
                    potential_hands.append((list(combo), hand_value, hand_type))
            
            potential_hands.sort(key=lambda x: x[1], reverse=True)
            
            if potential_hands and potential_hands[0][1] >= HandType.TWO_PAIR.value:
                indices = potential_hands[0][0]
                action = 0
                for idx in indices:
                    action |= (1 << idx)
                
                next_state, reward, done, info = env.step_play(action)
                
                enhanced_reward = reward * 4.0
                
                play_agent.remember(state, action, enhanced_reward, next_state, done)
                examples_added += 1
                
                state = next_state
            
            elif env.game_manager.discards_used < env.game_manager.max_discards_per_round:
                cards_with_values = [(i, card.rank.value) for i, card in enumerate(current_hand)]
                cards_with_values.sort(key=lambda x: x[1]) 
                
                indices = [idx for idx, _ in cards_with_values[:min(3, len(cards_with_values))]]
                
                action = 0
                for idx in indices:
                    action |= (1 << idx)
                
                action += 256
                
                next_state, reward, done, info = env.step_play(action)
                play_agent.remember(state, action, reward, next_state, done)
                examples_added += 1
                
                state = next_state
            
            else:
                if potential_hands:
                    indices = potential_hands[0][0]
                    action = 0
                    for idx in indices:
                        action |= (1 << idx)
                else:
                    action = (1 << len(current_hand)) - 1
                
                next_state, reward, done, info = env.step_play(action)
                play_agent.remember(state, action, reward, next_state, done)
                examples_added += 1
                
                state = next_state
            
            if info.get('shop_phase', False) and not done:
                next_state, _, done, _ = env.step_strategy(15) 
                state = next_state
    
    print(f"Successfully added {examples_added} demonstration examples to memory")
    return examples_added



def train_with_separate_agents():
    """Training function with improved shop behavior for Balatro RL agent and win tracking"""
    
    env = BalatroEnv(config={})
    
    play_state_size = len(env._get_play_state())
    play_action_size = env._define_play_action_space()
    
    strategy_state_size = len(env._get_strategy_state())
    strategy_action_size = env._define_strategy_action_space()
    
    print(f"Play agent: state_size={play_state_size}, action_size={play_action_size}")
    print(f"Strategy agent: state_size={strategy_state_size}, action_size={strategy_action_size}")
    
    play_agent = PlayingAgent(state_size=play_state_size, action_size=play_action_size)
    strategy_agent = StrategyAgent(state_size=strategy_state_size, action_size=strategy_action_size)
    
    add_demonstration_examples(play_agent, num_examples=200)
    
    episodes = 5000
    batch_size = 64
    log_interval = 500
    #save_interval = 1000
    
    play_rewards = []
    strategy_rewards = []
    
    win_history = []
    max_ante_history = []
    win_rate_over_time = [] 
    avg_max_ante_over_time = []
    
    jokers_purchased_history = []
    planets_purchased_history = []
    tarots_purchased_history = []
    packs_opened_history = []
    joker_acquisition_counts = {}

    win_window_size = 100
    rolling_wins = []
    
    for e in range(episodes):
        env.reset()
        total_reward = 0
        max_ante = 1
        game_steps = 0
        
        play_episode_reward = 0
        strategy_episode_reward = 0
        items_purchased = 0
        episode_jokers_purchased = 0
        episode_planets_purchased = 0
        episode_tarots_purchased = 0
        episode_packs_opened = 0
        
        unique_jokers = set()
        
        done = False
        show_shop_next = False
        
        while not done and game_steps < 500:
            game_steps += 1
            
            if show_shop_next:
                if env.game_manager.game.current_ante > 24:
                    print("\n===== GAME COMPLETED! =====")
                    print(f"Final score: {env.game_manager.current_score}")
                    
                    total_reward += 500.0
                    done = True
                    break

                print(f"\n===== SHOP PHASE (Episode {e+1}) =====")
                
                env.update_shop()
                
                shop_done = False
                shop_steps = 0
                max_shop_steps = 50
                
                while not shop_done and not done and shop_steps < max_shop_steps:
                    shop_steps += 1
                    
                    strategy_state = env._get_strategy_state()
                    valid_actions = env.get_valid_strategy_actions()
                    strategy_action = strategy_agent.act(strategy_state, valid_actions)
                    
                    next_strategy_state, strategy_reward, strategy_done, strategy_info = env.step_strategy(strategy_action)
                    
                    strategy_agent.remember(strategy_state, strategy_action, strategy_reward, next_strategy_state, strategy_done)
                    
                    strategy_episode_reward += strategy_reward
                    total_reward += strategy_reward
                    done = strategy_done
                    
                    message = strategy_info.get('message', '')
                    if strategy_action < 4 and "Bought" in message:
                        items_purchased += 1
                        
                        if "joker" in message.lower():
                            episode_jokers_purchased += 1
                            
                            parts = message.split("Bought ")
                            if len(parts) > 1:
                                name_parts = parts[1].split(" for $")
                                if len(name_parts) > 0:
                                    joker_name = name_parts[0].strip()
                                    unique_jokers.add(joker_name)
                                    joker_acquisition_counts[joker_name] = joker_acquisition_counts.get(joker_name, 0) + 1

                        elif "planet" in message.lower():
                            episode_planets_purchased += 1
                            
                        elif "tarot" in message.lower():
                            episode_tarots_purchased += 1
                            
                        elif any(pack_type in message.lower() for pack_type in ["pack", "buffoon", "celestial", "arcana"]):
                            episode_packs_opened += 1
                    
                    if strategy_action == 15 or done:
                        if hasattr(env, 'pending_tarots') and env.pending_tarots and env.game_manager.current_hand:
                            env.use_pending_tarots()
                        
                        shop_done = True
                        show_shop_next = False
                        print(f"Advanced to Ante {env.game_manager.game.current_ante}")
                        
                        if not env.game_manager.current_hand and not done:
                            env.game_manager.deal_new_hand()
                
                if shop_steps >= max_shop_steps and not shop_done:
                    print(f"Forcing shop exit after {shop_steps} steps")
                    show_shop_next = False
                    
                    if not env.game_manager.current_hand and not done:
                        env.game_manager.deal_new_hand()
                
                continue
            
            play_state = env._get_play_state()
            valid_actions = env.get_valid_play_actions()
            play_action = play_agent.act(play_state, valid_actions)
            
            next_play_state, play_reward, done, play_info = env.step_play(play_action)
            
            play_agent.remember(play_state, play_action, play_reward, next_play_state, done)
            
            play_episode_reward += play_reward
            total_reward += play_reward
            
            if play_info.get('shop_phase', False) and not done:
                print(f"\n***** ANTE {env.game_manager.game.current_ante} BEATEN! *****")
                show_shop_next = True
            
            max_ante = max(max_ante, env.game_manager.game.current_ante)
        
        is_win = max_ante > 24
        win_history.append(1 if is_win else 0)
        max_ante_history.append(max_ante)
        
        jokers_purchased_history.append(episode_jokers_purchased)
        planets_purchased_history.append(episode_planets_purchased)
        tarots_purchased_history.append(episode_tarots_purchased)
        packs_opened_history.append(episode_packs_opened)
        if e % 100 == 0:
            hand_counts, percentages = analyze_hand_selection(play_agent, env)
            print("\nHand Type Distribution:")
            for hand_type, percentage in sorted(percentages.items(), key=lambda x: HandType[x[0]].value):
                print(f"  {hand_type}: {percentage:.1f}%")
        rolling_wins.append(1 if is_win else 0)
        if len(rolling_wins) > win_window_size:
            rolling_wins.pop(0)
        
        current_win_rate = 100 * sum(rolling_wins) / len(rolling_wins)
        win_rate_over_time.append(current_win_rate)
        
        recent_antes = max_ante_history[-win_window_size:] if len(max_ante_history) >= win_window_size else max_ante_history
        avg_max_ante = sum(recent_antes) / len(recent_antes)
        avg_max_ante_over_time.append(avg_max_ante)
        
        if len(play_agent.memory) >= batch_size:
            play_agent.replay(batch_size)
        
        if len(strategy_agent.memory) >= batch_size:
            strategy_agent.prioritized_strategy_replay(batch_size)
        
        play_agent.decay_epsilon()
        strategy_agent.decay_epsilon()
        
        play_rewards.append(play_episode_reward)
        strategy_rewards.append(strategy_episode_reward)
        
        if (e + 1) % log_interval == 0:
            avg_play_reward = sum(play_rewards[-log_interval:]) / log_interval
            avg_strategy_reward = sum(strategy_rewards[-log_interval:]) / log_interval
            avg_ante = sum(max_ante_history[-log_interval:]) / log_interval
            win_rate = 100 * sum(win_history[-log_interval:]) / log_interval
            
            print(f"\n===== Episode {e+1}/{episodes} =====")
            print(f"Play Agent Reward: {avg_play_reward:.2f}")
            print(f"Strategy Agent Reward: {avg_strategy_reward:.2f}")
            print(f"Average Max Ante: {avg_ante:.2f}")
            print(f"Win Rate (last {log_interval} episodes): {win_rate:.2f}%")
            print(f"Items Purchased in Episode: {items_purchased}")
            print(f"Jokers Purchased in Episode: {episode_jokers_purchased}")
            print(f"Planets Purchased in Episode: {episode_planets_purchased}")
            print(f"Tarots Purchased in Episode: {episode_tarots_purchased}")
            print(f"Packs Opened in Episode: {episode_packs_opened}")
            print(f"Unique Joker Types: {len(unique_jokers)}")
            print(f"Play Epsilon: {play_agent.epsilon:.3f}")
            print(f"Strategy Epsilon: {strategy_agent.epsilon:.3f}")
        
    
    play_agent.save_model("play_agent_final.h5")
    strategy_agent.save_model("strategy_agent_final.h5")
    
    print("\n===== Generating Training Plots =====")
    plot_training_metrics(episodes, win_history, max_ante_history, win_rate_over_time, 
                          avg_max_ante_over_time, jokers_purchased_history,
                          planets_purchased_history, tarots_purchased_history,
                          packs_opened_history)
    plot_joker_usage(joker_acquisition_counts)

    return play_agent, strategy_agent

def track_joker_usage(env, episodes=50):
    """Track which jokers are acquired and used most frequently"""
    env.reset()
    
    joker_counts = {}
    
    for e in range(episodes):
        env.reset()
        done = False
        game_steps = 0
        
        episode_jokers = set()
        
        while not done and game_steps < 500:
            game_steps += 1
            
            for joker in env.game_manager.game.inventory.jokers:
                if hasattr(joker, 'name'):
                    joker_name = joker.name
                    episode_jokers.add(joker_name)
                    joker_counts[joker_name] = joker_counts.get(joker_name, 0) + 1
            
            if hasattr(env, 'current_shop') and env.current_shop is not None:
                valid_actions = env.get_valid_strategy_actions()
                if valid_actions:
                    _, _, done, _ = env.step_strategy(random.choice(valid_actions))
            else:
                valid_actions = env.get_valid_play_actions()
                if valid_actions:
                    _, _, done, info = env.step_play(random.choice(valid_actions))
                    if info.get('shop_phase', False) and not done:
                        env.update_shop()
        
        if e % 10 == 0:
            print(f"Tracked jokers in {e+1}/{episodes} episodes")
            
    return joker_counts

def plot_joker_usage(joker_counts, top_n=15):
    """Create a bar chart showing the most frequently used jokers"""
    sorted_jokers = sorted(joker_counts.items(), key=lambda x: x[1], reverse=True)
    
    top_jokers = sorted_jokers[:top_n]
    
    names = [j[0] for j in top_jokers]
    counts = [j[1] for j in top_jokers]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(names, counts, color='skyblue')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}', ha='center', va='bottom')
    
    plt.title(f'Top {top_n} Most Frequently Used Jokers')
    plt.xlabel('Joker Name')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('joker_usage_frequency.png', dpi=300)
    plt.show()

def plot_training_metrics(episodes, win_history, max_ante_history, win_rate_over_time, 
                          avg_max_ante_over_time, jokers_history, planets_history, 
                          tarots_history, packs_history):
    """Plot the training metrics over time."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(1, len(win_rate_over_time) + 1), win_rate_over_time)
    plt.title('Win Rate Over Time (100-episode rolling window)')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(1, len(avg_max_ante_over_time) + 1), avg_max_ante_over_time)
    plt.title('Average Max Round Over Time (100-episode rolling window)')
    plt.xlabel('Episode')
    plt.ylabel('Average Max Round')
    plt.grid(True)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.subplot(2, 2, 3)
    plt.scatter(np.arange(1, len(max_ante_history) + 1), max_ante_history, 
               alpha=0.3, s=3, c=max_ante_history, cmap='viridis')
    
    plt.axhline(y=8, color='r', linestyle='--', alpha=0.7, label='Win Threshold (Round 24)')
    
    plt.title('Max Round Reached per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Max Round')
    plt.grid(True)
    plt.colorbar(label='Round Level')
    plt.legend()
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.subplot(2, 2, 4)
    
    window_size = 100
    jokers_smooth = moving_average(jokers_history, window_size)
    planets_smooth = moving_average(planets_history, window_size)
    tarots_smooth = moving_average(tarots_history, window_size)
    packs_smooth = moving_average(packs_history, window_size)
    
    min_length = min(len(jokers_smooth), len(planets_smooth), len(tarots_smooth), len(packs_smooth))
    jokers_smooth = jokers_smooth[:min_length]
    planets_smooth = planets_smooth[:min_length]
    tarots_smooth = tarots_smooth[:min_length]
    packs_smooth = packs_smooth[:min_length]
    
    x = np.arange(window_size, window_size + min_length)
    
    plt.stackplot(x, 
                 jokers_smooth, planets_smooth, tarots_smooth, packs_smooth,
                 labels=['Jokers', 'Planets', 'Tarots', 'Packs'],
                 colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'],
                 alpha=0.7)
    
    plt.title('Item Purchases Over Time (100-episode rolling average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Items Purchased')
    plt.grid(True)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('balatro_training_metrics.png', dpi=300)
    print("Plots saved as 'balatro_training_metrics.png'")
    plt.show()


def moving_average(data, window_size):
    """Calculate the moving average of a list."""
    if len(data) < window_size:
        return [sum(data) / len(data)] * len(data)
    
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size



def evaluate_with_purchase_tracking(play_agent, strategy_agent, episodes=20):
    """Evaluate agents with tracking of shop purchase behavior"""
    env = BalatroEnv()
    
    play_epsilon = play_agent.epsilon
    strategy_epsilon = strategy_agent.epsilon
    
    play_agent.epsilon = 0
    strategy_agent.epsilon = 0
    
    results = {
        'max_antes': [],
        'win_rate': 0,
        'average_score': 0,
        'items_purchased': [],
        'item_types': {
            'joker': 0,
            'planet': 0,
            'tarot': 0,
            'booster': 0
        }
    }
    
    for e in range(episodes):
        env.reset()
        max_ante = 1
        
        items_purchased = 0
        jokers_bought = 0
        planets_bought = 0
        tarots_bought = 0
        boosters_bought = 0
        
        done = False
        shop_phase = False
        game_steps = 0
        
        while not done and game_steps < 500:
            game_steps += 1
            
            if shop_phase:
                strategy_state = env._get_strategy_state()
                valid_actions = env.get_valid_strategy_actions()
                strategy_action = strategy_agent.act(strategy_state, valid_actions)
                
                next_state, reward, done, info = env.step_strategy(strategy_action)
                
                message = info.get('message', '')
                if strategy_action < 4 and "Bought" in message:
                    items_purchased += 1
                    
                    if "joker" in message.lower():
                        jokers_bought += 1
                    elif "planet" in message.lower():
                        planets_bought += 1
                    elif "tarot" in message.lower():
                        tarots_bought += 1
                    else:
                        boosters_bought += 1
                
                if strategy_action == 15 or done:
                    shop_phase = False
                    
                    if not env.game_manager.current_hand and not done:
                        env.game_manager.deal_new_hand()
                        
            else:
                play_state = env._get_play_state()
                valid_actions = env.get_valid_play_actions()
                play_action = play_agent.act(play_state, valid_actions)

                next_play_state, play_reward, done, info = env.step_play(play_action)
                                
                if info.get('shop_phase', False) and not done:
                    shop_phase = True
                    env.update_shop()
                
                max_ante = max(max_ante, env.game_manager.game.current_ante)
        
        results['max_antes'].append(max_ante)
        results['items_purchased'].append(items_purchased)
        results['item_types']['joker'] += jokers_bought
        results['item_types']['planet'] += planets_bought
        results['item_types']['tarot'] += tarots_bought
        results['item_types']['booster'] += boosters_bought
        
        if max_ante > 24:
            results['win_rate'] += 1
    
    results['win_rate'] = (results['win_rate'] / episodes) * 100
    results['average_score'] = sum(results['max_antes']) / episodes
    results['avg_items_purchased'] = sum(results['items_purchased']) / episodes
    
    total_items = sum(results['items_purchased'])
    if total_items > 0:
        results['item_types']['joker_percent'] = (results['item_types']['joker'] / total_items) * 100
        results['item_types']['planet_percent'] = (results['item_types']['planet'] / total_items) * 100
        results['item_types']['tarot_percent'] = (results['item_types']['tarot'] / total_items) * 100
        results['item_types']['booster_percent'] = (results['item_types']['booster'] / total_items) * 100
    else:
        results['item_types']['joker_percent'] = 0
        results['item_types']['planet_percent'] = 0
        results['item_types']['tarot_percent'] = 0
        results['item_types']['booster_percent'] = 0
    
    play_agent.epsilon = play_epsilon
    strategy_agent.epsilon = strategy_epsilon
    
    return results


def analyze_hand_selection(play_agent, env, episodes=20):
    """Analyze what kinds of hands the agent is selecting"""
    hand_counts = {ht.name: 0 for ht in HandType}
    hand_evaluator = HandEvaluator()
    
    original_epsilon = play_agent.epsilon
    play_agent.epsilon = 0.0 
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            if not env.game_manager.current_hand:
                break
                
            valid_actions = env.get_valid_play_actions()
            action = play_agent.act(state, valid_actions)
            
            if action < 256: 
                indices = env._convert_action_to_card_indices(action)
                if indices:
                    cards = [env.game_manager.current_hand[i] for i in indices 
                           if i < len(env.game_manager.current_hand)]
                    if cards:
                        hand_type, _, _ = hand_evaluator.evaluate_hand(cards)
                        hand_counts[hand_type.name] += 1
            
            next_state, _, done, info = env.step_play(action)
            state = next_state
            
            if info.get('shop_phase', False) and not done:
                next_state, _, done, _ = env.step_strategy(15) 
                state = next_state
    
    total_hands = sum(hand_counts.values())
    if total_hands > 0:
        percentages = {ht: (count / total_hands) * 100 for ht, count in hand_counts.items()}
    else:
        percentages = {ht: 0 for ht in hand_counts}
    
    play_agent.epsilon = original_epsilon
    
    return hand_counts, percentages


def evaluate_agents(play_agent, strategy_agent, episodes=100):
    """Evaluate agent performance without exploration"""
    env = BalatroEnv()
    
    play_epsilon = play_agent.epsilon
    strategy_epsilon = strategy_agent.epsilon
    
    play_agent.epsilon = 0
    strategy_agent.epsilon = 0
    
    results = {
        'play_rewards': [],
        'strategy_rewards': [],
        'max_antes': [],
        'hands_played': [],
        'win_rate': 0,
        'average_score': 0
    }
    
    for e in range(episodes):
        env.reset()
        play_state = env._get_play_state()
        play_total_reward = 0
        strategy_total_reward = 0
        
        max_ante = 1
        hands_played = 0
        game_won = False
        game_steps = 0
        
        done = False
        
        while not done and game_steps < 500:
            game_steps += 1
            
            valid_play_actions = env.get_valid_play_actions()
            play_action = play_agent.act(play_state, valid_actions=valid_play_actions)
            next_play_state, play_reward, done, info = env.step_play(play_action)
            
            play_state = next_play_state
            play_total_reward += play_reward
            
            hands_played += 1
            max_ante = max(max_ante, env.game_manager.game.current_ante)
            
            if info.get('shop_phase', False) and not done:
                strategy_state = env._get_strategy_state()
                shop_done = False
                
                while not shop_done and not done and game_steps < 500:
                    game_steps += 1
                    valid_strategy_actions = env.get_valid_strategy_actions()
                    strategy_action = strategy_agent.act(strategy_state, valid_strategy_actions)
                    
                    next_strategy_state, strategy_reward, strategy_done, _ = env.step_strategy(strategy_action)
                    strategy_state = next_strategy_state
                    strategy_total_reward += strategy_reward
                    done = strategy_done
                    
                    if strategy_action == 15 or not env.game_manager.current_ante_beaten or done:
                        shop_done = True
                
                if not done:
                    play_state = env._get_play_state()
        
        results['play_rewards'].append(play_total_reward)
        results['strategy_rewards'].append(strategy_total_reward)
        results['max_antes'].append(max_ante)
        results['hands_played'].append(hands_played)
        
        if max_ante > 24:
            game_won = True
        results['win_rate'] += 1 if game_won else 0
        
        if (e + 1) % 10 == 0:
            print(f"Evaluated {e + 1}/{episodes} episodes. Current max ante: {max_ante}")
    
    results['win_rate'] = results['win_rate'] / episodes * 100
    results['average_score'] = sum(results['max_antes']) / episodes
    
    play_agent.epsilon = play_epsilon
    strategy_agent.epsilon = strategy_epsilon
    
    return results


if __name__ == "__main__":
    play_agent, strategy_agent = train_with_separate_agents()
    
    results = evaluate_with_purchase_tracking(play_agent, strategy_agent, episodes=50)
    
    print("\n===== FINAL EVALUATION =====")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Average Max Ante: {results['average_score']:.2f}")
    print(f"Average Items Purchased: {results['avg_items_purchased']:.2f}")
    print("\nPurchase Breakdown:")
    print(f"  Jokers: {results['item_types']['joker']} ({results['item_types']['joker_percent']:.1f}%)")
    print(f"  Planets: {results['item_types']['planet']} ({results['item_types']['planet_percent']:.1f}%)")
    print(f"  Tarots: {results['item_types']['tarot']} ({results['item_types']['tarot_percent']:.1f}%)")
    print(f"  Boosters: {results['item_types']['booster']} ({results['item_types']['booster_percent']:.1f}%)")
