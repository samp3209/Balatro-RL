import random
from typing import List, Dict, Tuple
from Enums import *
from Card import Card
from Inventory import Inventory
from Game import Game
from HandEvaluator import *
from GameManager import GameManager
from JokerCreation import create_joker
from Shop import Shop, ShopItem, ShopItemType, initialize_shops_for_game, FixedShop

def print_card(card: Card) -> str:
    """Format a card as a string for display"""
    rank_symbols = {
        Rank.ACE: "A",
        Rank.TWO: "2", 
        Rank.THREE: "3",
        Rank.FOUR: "4",
        Rank.FIVE: "5",
        Rank.SIX: "6",
        Rank.SEVEN: "7",
        Rank.EIGHT: "8",
        Rank.NINE: "9",
        Rank.TEN: "10",
        Rank.JACK: "J",
        Rank.QUEEN: "Q",
        Rank.KING: "K"
    }
    
    suit_symbols = {
        Suit.HEARTS: "♥",
        Suit.DIAMONDS: "♦",
        Suit.CLUBS: "♣",
        Suit.SPADES: "♠",
        Suit.WILD: "★"
    }
    
    rank_str = rank_symbols.get(card.rank, str(card.rank))
    suit_str = suit_symbols.get(card.suit, "?")
    
    enhancement = ""
    if card.enhancement != CardEnhancement.NONE:
        enhancement = f"[{card.enhancement.name}]"
        
    status = ""
    if hasattr(card, 'debuffed') and card.debuffed:
        status = " (DEBUFFED)"
        
    return f"{rank_str}{suit_str}{enhancement}{status}"

def print_hand(cards: List[Card], title: str = "Current Hand"):
    """Print a formatted display of cards"""
    print(f"\n=== {title} ===")
    for i, card in enumerate(cards):
        status = ""
        if card.scored:
            status = " (scoring)"
        if hasattr(card, 'debuffed') and card.debuffed:
            status += " (DEBUFFED)"
        print(f"{i}: {print_card(card)}{status}")
    print()

def print_jokers(jokers, title: str = "Jokers"):
    """Print a formatted display of jokers"""
    print(f"\n=== {title} ===")
    for i, joker in enumerate(jokers):
        print(f"{i}: {joker.name} (${joker.sell_value})")
    print()

def print_shop_items(shop):
    """Print the items available in the shop"""
    print("\n=== SHOP ===")
    for i, item in enumerate(shop.items):
        if item is not None:
            item_type = item.item_type.name
            
            if item.item_type == ShopItemType.JOKER and hasattr(item.item, 'name'):
                item_name = item.item.name
            elif item.item_type in [ShopItemType.TAROT, ShopItemType.PLANET] and hasattr(item.item, 'name'):
                item_name = item.item.name
            else:
                item_name = str(item.item) if item.item is not None else "Unknown"
                
            price = shop.get_item_price(i)
            print(f"{i}: {item_name} ({item_type}) - ${price}")
        else:
            print(f"{i}: [Empty]")
    print()


def print_planet_levels(inventory):
    """Print the current level of each planet type"""
    print("\n=== Planet Levels ===")
    for planet_type, level in sorted(inventory.planet_levels.items(), key=lambda x: x[0].name):
        mult_bonus, chip_bonus = inventory.get_planet_bonus(planet_type)
        print(f"{planet_type.name.title()}: Level {level} (+{mult_bonus} mult, +{chip_bonus} chips)")
    print()



def get_shop_item_contents(shop_item):
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
            
        return get_pack_contents(pack_type)
        
    return None


def handle_shop_interaction(game_manager, shop):
    """Handle player interaction with the shop with automatic planet usage"""
    inventory = game_manager.game.inventory
    print(f"Money: ${inventory.money}")
    print_shop_items(shop)
    
    print_planet_levels(inventory)
    
    if inventory.jokers:
        print_jokers(inventory.jokers)
    
    pending_tarots = []
        
    while True:
        joker_count = len(inventory.jokers)
        
        bought_item = False
        
        if joker_count >= 5:
            for i, item in enumerate(shop.items):
                if item is not None and item.item_type == ShopItemType.JOKER:
                    if random.random() < 0.5:
                        min_value_idx = min(range(len(inventory.jokers)), 
                                           key=lambda i: inventory.jokers[i].sell_value)
                        joker_name = inventory.jokers[min_value_idx].name
                        sell_value = shop.sell_item("joker", min_value_idx, inventory)
                        print(f"Sold {joker_name} for ${sell_value} to make space for a new joker")
                        joker_count -= 1
                        break
        
        if joker_count < 5:
            for i, item in enumerate(shop.items):
                if item is not None and item.item_type == ShopItemType.JOKER:
                    price = shop.get_item_price(i)
                    if inventory.money >= price and hasattr(item.item, 'name'):
                        success = shop.buy_item(i, inventory)
                        if success:
                            print(f"Bought {item.item.name} for ${price}")
                            bought_item = True
                            break
        
        if not bought_item:
            for i, item in enumerate(shop.items):
                if (item is not None and 
                    item.item_type == ShopItemType.PLANET and 
                    inventory.money >= shop.get_item_price(i)):
                    
                    planet_name = item.item.name
                    planet_type = item.item.planet_type
                    price = shop.get_item_price(i)
                    
                    shop.items[i] = None
                    inventory.money -= price
                    
                    current_level = inventory.planet_levels.get(planet_type, 1)
                    inventory.planet_levels[planet_type] = current_level + 1
                    
                    print(f"Bought and used {planet_name} planet for ${price}!")
                    print(f"Upgraded {planet_name} to level {current_level + 1}")
                    
                    mult_bonus, chip_bonus = inventory.get_planet_bonus(planet_type)
                    print(f"{planet_name} now provides: +{mult_bonus} mult, +{chip_bonus} chips")
                    
                    bought_item = True
                    break
        
        if not bought_item:
            for i, item in enumerate(shop.items):
                if (item is not None and 
                    item.item_type == ShopItemType.TAROT and 
                    inventory.money >= shop.get_item_price(i)):
                    
                    tarot_name = item.item.name
                    price = shop.get_item_price(i)
                    
                    success = shop.buy_item(i, inventory)
                    if success:
                        print(f"Bought {tarot_name} tarot for ${price} (will be used in next round)")
                        pending_tarots.append(tarot_name)
                        bought_item = True
                        break
        
        if not bought_item:
            for i, item in enumerate(shop.items):
                if item is not None:
                    price = shop.get_item_price(i)
                    if inventory.money >= price:
                        try:
                            if item.item_type == ShopItemType.BOOSTER:
                                pack_type = str(item.item)
                                
                                pack_contents = get_shop_item_contents(item)
                                
                                success = shop.buy_item(i, inventory)
                                if success:
                                    print(f"Bought {pack_type} for ${price}")
                                    result_message = handle_pack_opening(pack_type, pack_contents, inventory, game_manager)
                                    print(result_message)
                                    bought_item = True
                                    break
                            else:
                                item_name = item.get_name() if hasattr(item, 'get_name') else "Item"
                                success = shop.buy_item(i, inventory)
                                if success:
                                    print(f"Bought {item_name} for ${price}")
                                    bought_item = True
                                    break
                        except (AttributeError, TypeError) as e:
                            print(f"Error buying item: {e}")
                            continue
        
        if not bought_item and joker_count > 3 and random.random() < 0.3:
            if inventory.jokers:
                min_value_idx = min(range(len(inventory.jokers)), 
                                    key=lambda i: inventory.jokers[i].sell_value)
                joker_name = inventory.jokers[min_value_idx].name
                sell_value = shop.sell_item("joker", min_value_idx, inventory)
                print(f"Sold {joker_name} for ${sell_value}")
        
        break
    
    print("\nLeaving shop...")
    print(f"Money: ${inventory.money}")
    print(f"Jokers: {[j.name for j in inventory.jokers]}")
    print(f"Consumables: {len(inventory.consumables)}")
    
    #print_planet_levels(inventory)
    
    return pending_tarots 

def use_pending_tarots(game_manager, pending_tarots):
    """
    Use tarot cards that were purchased from the shop
    
    Args:
        game_manager: The game manager instance
        pending_tarots: List of tarot names to use
    """
    if not pending_tarots:
        return
    
    print("\n=== Using Tarot Cards From Shop ===")
    
    for tarot_name in pending_tarots:
        tarot_indices = game_manager.game.inventory.get_consumable_tarot_indices()
        tarot_index = None
        
        for idx in tarot_indices:
            consumable = game_manager.game.inventory.consumables[idx]
            if hasattr(consumable.item, 'name') and consumable.item.name.lower() == tarot_name.lower():
                tarot_index = idx
                break
        
        if tarot_index is None:
            print(f"Could not find tarot {tarot_name} in inventory")
            continue
            
        tarot = game_manager.game.inventory.consumables[tarot_index].item
        cards_required = tarot.selected_cards_required
        
        if cards_required > len(game_manager.current_hand):
            print(f"Not enough cards to use {tarot_name}, needs {cards_required}")
            continue
            
        selected_indices = []
        
        if cards_required > 0:
            card_values = [(i, card.rank.value) for i, card in enumerate(game_manager.current_hand)]
            card_values.sort(key=lambda x: x[1])
            selected_indices = [idx for idx, _ in card_values[:cards_required]]
        
        success, message = game_manager.use_tarot(tarot_index, selected_indices)
        if success:
            print(f"Used {tarot_name}: {message}")
        else:
            print(f"Failed to use {tarot_name}: {message}")



def get_pack_contents(pack_type):
    """
    Get the contents of a pack based on its type
    
    Args:
        pack_type: The type of pack (e.g., "Standard Pack")
        
    Returns:
        list: The contents of the pack
    """
    try:
        # Try to find pack_type in PackType enum
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
        
        # Search through all ante shops for matching pack
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


def get_shop_for_current_ante(game_manager, all_shops):
    """Get the appropriate shop for the current ante and blind"""
    current_ante = game_manager.game.current_ante
    ante_number = ((current_ante - 1) // 3) + 1
    
    blind_type_map = {
        0: "boss_blind",
        1: "small_blind", 
        2: "medium_blind"
    }
    
    blind_type = blind_type_map[current_ante % 3]
    
    if ante_number in all_shops and blind_type in all_shops[ante_number]:
        return all_shops[ante_number][blind_type]
    
    return Shop()

def handle_pack_opening(pack_type, pack_contents, inventory, game_manager=None):
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
            # Simple AI: randomly select a card from the pack
            selected_idx = random.randint(0, len(pack_contents) - 1)
            
            try:
                card_string = pack_contents[selected_idx]
                print(f"Processing card string: '{card_string}'")
                
                parts = card_string.split()
                if not parts:
                    print(f"WARNING: Empty card string")
                    continue
                
                # Map strings to proper Rank enums
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
                
                # Map strings to proper Suit enums
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
                
                # Get rank from first part of string
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

                # Get suit from last part of string
                suit_str = parts[-1].lower() if len(parts) > 1 else "hearts"
                suit = suit_map.get(suit_str, Suit.HEARTS)
                if suit_str not in suit_map:
                    print(f"WARNING: Invalid suit '{suit_str}', defaulting to HEARTS")
                
                # Create card with proper Suit and Rank enums
                if not isinstance(rank, Rank) or not isinstance(suit, Suit):
                    print(f"ERROR: Invalid rank or suit type - rank: {type(rank)}, suit: {type(suit)}")
                    continue
                    
                card = Card(suit, rank)
                
                # Debug print
                print(f"Created card: {card}, rank type: {type(card.rank)}, suit type: {type(card.suit)}")
                
                # Apply enhancement if specified
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
                    "blue": None,  # "blue stamp" is handled specially
                    "stamp": None  # Part of "blue stamp" or similar
                }
                
                for part in parts[1:-1]:
                    part_lower = part.lower()
                    if part_lower in enhancement_map and enhancement_map[part_lower] is not None:
                        card.enhancement = enhancement_map[part_lower]
                    elif part_lower == "blue" and "stamp" in [p.lower() for p in parts[1:-1]]:
                        # Handle special case for "blue stamp"
                        card.enhancement = CardEnhancement.FOIL
                
                # Validate card before adding
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
        # Existing code for Celestial packs
        for i in range(min(num_to_select, len(pack_contents))):
            selected_idx = random.randint(0, len(pack_contents) - 1)
            planet_name = pack_contents[selected_idx]
            
            try:
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
    
    elif "ARCANA" in pack_type.upper() or "BUFFOON" in pack_type.upper():
        # Existing code for Arcana and Buffoon packs
        # No changes needed here as it doesn't create cards
        pass
    
    return message

def simulate_game():
    """Run a simulation of the game"""
    game_manager = GameManager(seed=42)
    game_manager.start_new_game()
    
    all_shops = initialize_shops_for_game()
    
    jokers = []
    for joker_name in jokers:
        joker = create_joker(joker_name)
        if joker:
            game_manager.game.inventory.add_joker(joker)
            print(f"Added {joker_name} to inventory")
    
    game_manager.game.inventory.money = 0 #set value higher for debugging
    
    print("\n===== STARTING GAME =====")
    print(f"Current Ante: {game_manager.game.current_ante}, Blind: {game_manager.game.current_blind}")
    print(f"Jokers in inventory: {[j.name for j in game_manager.game.inventory.jokers]}")
    print(f"Money: ${game_manager.game.inventory.money}")
    print_planet_levels(game_manager.game.inventory)

    # Game loop
    max_rounds = 75
    rounds_played = 0
    max_loop_iterations = 75
    loop_count = 0
    last_ante = 0
    show_shop_next = False
    pending_tarots = []
    
    while not game_manager.game_over and rounds_played < max_rounds and loop_count < max_loop_iterations:
        rounds_played += 1
        loop_count += 1
        
        if show_shop_next:
            print("\n===== SHOP PHASE =====")
            shop = get_shop_for_current_ante(game_manager, all_shops)
            
            new_tarots = handle_shop_interaction(game_manager, shop)
            if new_tarots:
                pending_tarots.extend(new_tarots)
                
            show_shop_next = False
            
            if not game_manager.current_hand:
                print("Dealing new hand for tarot usage")
                game_manager.deal_new_hand()
            
            if pending_tarots and game_manager.current_hand:
                use_pending_tarots(game_manager, pending_tarots)
                pending_tarots = []
        
        blind_type = "Small"
        if game_manager.game.current_ante % 3 == 2:
            blind_type = "Medium"
        elif game_manager.game.current_ante % 3 == 0:
            blind_type = "Boss"
            
        current_ante_number = ((game_manager.game.current_ante - 1) // 3) + 1
        
        boss_blind_indicator = "🔥 BOSS BLIND 🔥" if blind_type == "Boss" else ""
        print(f"\n----- Ante {current_ante_number}, {blind_type} Blind: {game_manager.game.current_blind} {boss_blind_indicator} -----")
        print(f"Hand {game_manager.hands_played + 1}/{game_manager.max_hands_per_round}, " + 
              f"Discards Used: {game_manager.discards_used}/{game_manager.max_discards_per_round}, " + 
              f"Score: {game_manager.current_score}/{game_manager.game.current_blind}")
        
        if game_manager.current_ante_beaten:
            print(f"Blind beaten! Moving to next blind.")
            current_ante = game_manager.game.current_ante 
            game_manager.next_ante()
            loop_count = 0 
            show_shop_next = True
            continue
            
        print_hand(game_manager.current_hand)
        
        if not game_manager.current_hand:
            print("No cards in hand - dealing new hand")
            game_manager.deal_new_hand()
            continue
        
        try:
            best_hand_info = game_manager.get_best_hand_from_current()
            if best_hand_info:
                recommended_indices, reason = game_manager.get_recommended_play()
                print(f"Recommended play: {recommended_indices} - {reason}")
            else:
                print("No valid play found")
                recommended_indices = []
        except Exception as e:
            print(f"Error getting recommendation: {e}")
            recommended_indices = []
        

        has_forced_card = False
        if (game_manager.game.is_boss_blind and 
            game_manager.game.active_boss_blind_effect == BossBlindEffect.FORCE_CARD_SELECTION and 
            game_manager.game.forced_card_index is not None):
            has_forced_card = True
            forced_idx = game_manager.game.forced_card_index
            print(f"⚠️ BOSS BLIND: Must include card at index {forced_idx} in play/discard ⚠️")
            
            if forced_idx not in recommended_indices:
                recommended_indices.append(forced_idx)
                print(f"Added forced card at index {forced_idx} to recommended play")
        
        if game_manager.discards_used < game_manager.max_discards_per_round:
            best_hand_info = game_manager.get_best_hand_from_current()
            if best_hand_info:
                best_hand, _ = best_hand_info
                if best_hand.value <= HandType.PAIR.value and random.random() < 0.7:
                    discard_indices = []
                    for i, card in enumerate(game_manager.current_hand):
                        if card.rank.value < 10 and random.random() < 0.5:
                            discard_indices.append(i)
                                
                    if has_forced_card and game_manager.game.forced_card_index not in discard_indices:
                        discard_indices.append(game_manager.game.forced_card_index)
                                
                    if discard_indices:
                        success, message = game_manager.discard_cards(discard_indices)
                        print(f"DISCARD: {message}")
                        continue
        
        if recommended_indices:
            success, message = game_manager.play_cards(recommended_indices)
            print(f"PLAY: {message}")
            
            if game_manager.current_ante_beaten:
                print(f"Ante beaten! Score: {game_manager.current_score}/{game_manager.game.current_blind}")
                current_ante = game_manager.game.current_ante  
                game_manager.next_ante()
                loop_count = 0
                show_shop_next = True
            continue
        else:
            if game_manager.current_hand:
                play_indices = list(range(len(game_manager.current_hand)))
                success, message = game_manager.play_cards(play_indices)
                print(f"PLAY (all cards): {message}")
                
                if game_manager.current_ante_beaten:
                    print(f"Ante beaten! Score: {game_manager.current_score}/{game_manager.game.current_blind}")
                    current_ante = game_manager.game.current_ante
                    game_manager.next_ante()
                    loop_count = 0 
                    show_shop_next = True 
                continue
    
    print("\n===== GAME OVER =====")
    print(f"Final ante: {((game_manager.game.current_ante - 1) // 3) + 1}")
    print(f"Final score: {game_manager.current_score}")
    print(f"Final money: ${game_manager.game.inventory.money}")
    print(f"Final jokers: {[j.name for j in game_manager.game.inventory.jokers]}")
    print(f"Game over: {game_manager.game_over}")
if __name__ == "__main__":
    simulate_game()