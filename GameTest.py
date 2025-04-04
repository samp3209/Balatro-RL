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
    """Handle player interaction with the shop"""
    inventory = game_manager.game.inventory
    print(f"Money: ${inventory.money}")
    print_shop_items(shop)
    
    if inventory.jokers:
        print_jokers(inventory.jokers)
        
    while True:
        joker_count = len(inventory.jokers)
        
        bought_item = False
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

def get_pack_contents(pack_type):
    pack_enum = None
    for pt in PackType:
        if pt.value == pack_type:
            pack_enum = pt
            break
    
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
                if item.get("item_type") == ShopItemType.BOOSTER and item.get("pack_type") == pack_enum:
                    if "contents" in item:
                        return item["contents"]



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
    
    # Display the pack contents
    for i, item in enumerate(pack_contents):
        print(f"{i}: {item}")
    
    # Determine how many items to select based on pack type
    if "MEGA" in pack_type.upper():
        num_to_select = 2
    else:
        num_to_select = 1
    
    message = ""
    
    if "STANDARD" in pack_type.upper():
        # For standard packs, add cards to the deck
        for i in range(min(num_to_select, len(pack_contents))):
            # Simple AI: randomly select a card from the pack
            selected_idx = random.randint(0, len(pack_contents) - 1)
            
            try:
                # Try to parse the card string
                card_string = pack_contents[selected_idx]
                
                # Simple card parsing logic
                parts = card_string.split()
                
                rank_map = {"A": Rank.ACE, "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, 
                           "5": Rank.FIVE, "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT,
                           "9": Rank.NINE, "10": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING}
                
                suit_map = {"heart": Suit.HEARTS, "hearts": Suit.HEARTS, "♥": Suit.HEARTS,
                           "diamond": Suit.DIAMONDS, "diamonds": Suit.DIAMONDS, "♦": Suit.DIAMONDS,
                           "club": Suit.CLUBS, "clubs": Suit.CLUBS, "♣": Suit.CLUBS,
                           "spade": Suit.SPADES, "spades": Suit.SPADES, "♠": Suit.SPADES}
                
                # Extract rank
                rank_str = parts[0]
                rank = rank_map.get(rank_str)
                if not rank:
                    try:
                        rank_value = int(rank_str)
                        for r in Rank:
                            if r.value == rank_value:
                                rank = r
                                break
                    except ValueError:
                        rank = Rank.ACE  # Default to Ace if parsing fails
                
                # Extract suit (usually the last part)
                suit_str = parts[-1].lower() if len(parts) > 1 else "hearts"
                suit = suit_map.get(suit_str, Suit.HEARTS)  # Default to Hearts if not found
                
                # Create the card
                card = Card(suit, rank)
                
                # Check for enhancements
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
                    "bonus": CardEnhancement.BONUS
                }
                
                for part in parts[1:-1]:
                    if part.lower() in enhancement_map:
                        card.enhancement = enhancement_map[part.lower()]
                
                # Add card to deck
                inventory.add_card_to_deck(card)
                
                message += f"Added {card_string} to deck. "
                print(f"Selected and added {card_string} to deck")
                
            except Exception as e:
                print(f"Error processing card: {e}")
                message += f"Failed to add card: {e}. "
    
    elif "CELESTIAL" in pack_type.upper():
        # For celestial packs, immediately use the planet card
        for i in range(min(num_to_select, len(pack_contents))):
            # Select planet randomly
            selected_idx = random.randint(0, len(pack_contents) - 1)
            planet_name = pack_contents[selected_idx]
            
            try:
                planet = create_planet_by_name(planet_name)
                if planet and hasattr(planet, 'planet_type'):
                    # Use the planet immediately to increase its level
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
        # For arcana packs, use the tarot card immediately
        if game_manager is not None and game_manager.current_hand:
            for i in range(min(num_to_select, len(pack_contents))):
                # Select tarot randomly
                selected_idx = random.randint(0, len(pack_contents) - 1)
                tarot_name = pack_contents[selected_idx]
                
                try:
                    tarot = create_tarot_by_name(tarot_name)
                    if tarot:
                        # Special case for The Emperor - adds to inventory
                        if tarot.tarot_type == TarotType.THE_EMPEROR:
                            if inventory.get_available_space() > 0:
                                inventory.add_consumable(tarot)
                                message += f"Added {tarot_name} to inventory. "
                                print(f"Added {tarot_name} to inventory")
                            else:
                                message += f"No space to add {tarot_name}. "
                                print(f"No space to add {tarot_name}")
                        else:
                            # Get required number of cards for this tarot
                            cards_required = tarot.selected_cards_required
                            if cards_required > 0:
                                # Select random cards from current hand to apply tarot effect
                                hand = game_manager.current_hand
                                if len(hand) >= cards_required:
                                    selected_cards = random.sample(hand, cards_required)
                                    
                                    # Apply tarot effect
                                    game_state = {
                                        'money': inventory.money,
                                        'last_tarot_used': inventory.last_tarot,
                                        'last_planet_used': inventory.last_planet
                                    }
                                    
                                    effect = tarot.apply_effect(selected_cards, inventory, game_state)
                                    
                                    # Update message
                                    effect_msg = effect.message if hasattr(effect, 'message') else ""
                                    message += f"Used {tarot_name} tarot: {effect_msg} "
                                    print(f"Used {tarot_name} tarot on {cards_required} card(s): {effect_msg}")
                                    
                                    # Update money if needed
                                    if hasattr(effect, 'money_gained') and effect.money_gained > 0:
                                        inventory.money += effect.money_gained
                                        message += f"Gained ${effect.money_gained}. "
                                else:
                                    message += f"Not enough cards in hand to use {tarot_name}. "
                                    print(f"Not enough cards in hand to use {tarot_name}")
                            else:
                                # For tarots that don't require card selection
                                game_state = {
                                    'money': inventory.money,
                                    'last_tarot_used': inventory.last_tarot,
                                    'last_planet_used': inventory.last_planet
                                }
                                
                                effect = tarot.apply_effect([], inventory, game_state)
                                
                                # Update message
                                effect_msg = effect.message if hasattr(effect, 'message') else ""
                                message += f"Used {tarot_name} tarot: {effect_msg} "
                                print(f"Used {tarot_name} tarot: {effect_msg}")
                                
                                # Update money if needed
                                if hasattr(effect, 'money_gained') and effect.money_gained > 0:
                                    inventory.money += effect.money_gained
                                    message += f"Gained ${effect.money_gained}. "
                    else:
                        message += f"Failed to create tarot {tarot_name}. "
                        print(f"Failed to create tarot {tarot_name}")
                except Exception as e:
                    print(f"Error processing tarot: {e}")
                    message += f"Failed to use tarot {tarot_name}: {e}. "
        else:
            message += "No cards in hand to use tarot effects. "
            print("No cards in hand to use tarot effects")
    
    elif "BUFFOON" in pack_type.upper():
        # For buffoon packs, add joker cards to inventory
        for i in range(min(num_to_select, len(pack_contents))):
            # Select card randomly
            selected_idx = random.randint(0, len(pack_contents) - 1)
            item_name = pack_contents[selected_idx]
            
            try:
                # Try to create a joker
                joker = create_joker(item_name)
                if joker and inventory.has_joker_space():
                    inventory.add_joker(joker)
                    message += f"Added {item_name} joker to inventory. "
                    print(f"Added {item_name} joker to inventory")
                else:
                    message += f"Failed to add joker {item_name} (space full or invalid). "
                    print(f"Failed to add joker {item_name}")
            except Exception as e:
                print(f"Error processing item: {e}")
                message += f"Failed to add item {item_name}: {e}. "
    
    return message


def simulate_game():
    """Run a simulation of the game"""
    game_manager = GameManager(seed=42)
    game_manager.start_new_game()
    
    # Initialize all shops for the game
    all_shops = initialize_shops_for_game()
    
    jokers = ["Mr. Bones", "Clever", "Smiley"]
    for joker_name in jokers:
        joker = create_joker(joker_name)
        if joker:
            game_manager.game.inventory.add_joker(joker)
            print(f"Added {joker_name} to inventory")
    
    # Give starting money
    game_manager.game.inventory.money = 10
    
    print("\n===== STARTING GAME =====")
    print(f"Current Ante: {game_manager.game.current_ante}, Blind: {game_manager.game.current_blind}")
    print(f"Jokers in inventory: {[j.name for j in game_manager.game.inventory.jokers]}")
    print(f"Money: ${game_manager.game.inventory.money}")
    
    # Game loop
    max_rounds = 30
    rounds_played = 0
    max_loop_iterations = 30 # Prevent infinite loops
    loop_count = 0
    last_ante = 0
    show_shop_next = False
    
    while not game_manager.game_over and rounds_played < max_rounds and loop_count < max_loop_iterations:
        rounds_played += 1
        loop_count += 1
        
        # Check if we should show the shop (after beating a blind)
        if show_shop_next:
            print("\n===== SHOP PHASE =====")
            shop = get_shop_for_current_ante(game_manager, all_shops)
            handle_shop_interaction(game_manager, shop)
            show_shop_next = False
        
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
        
        # Check if blind is beaten, advance to next blind and show shop next iteration
        if game_manager.current_ante_beaten:
            print(f"Blind beaten! Moving to next blind.")
            current_ante = game_manager.game.current_ante  # Store current ante before advancing
            game_manager.next_ante()
            loop_count = 0  # Reset loop counter when advancing to next ante
            show_shop_next = True  # Show shop after beating a blind
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
        
        # Decide whether to discard or play
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
        
        # Play the recommended cards if we have a recommendation and haven't discarded
        if recommended_indices:
            success, message = game_manager.play_cards(recommended_indices)
            print(f"PLAY: {message}")
            
            if game_manager.current_ante_beaten:
                print(f"Ante beaten! Score: {game_manager.current_score}/{game_manager.game.current_blind}")
                current_ante = game_manager.game.current_ante  # Store current ante before advancing
                game_manager.next_ante()
                loop_count = 0  # Reset loop counter when advancing to next ante
                show_shop_next = True  # Show shop after beating a blind
            continue
        else:
            # If we have no recommendation but still have cards, try to play the best we can
            if game_manager.current_hand:
                # Just play all cards as a last resort
                play_indices = list(range(len(game_manager.current_hand)))
                success, message = game_manager.play_cards(play_indices)
                print(f"PLAY (all cards): {message}")
                
                if game_manager.current_ante_beaten:
                    print(f"Ante beaten! Score: {game_manager.current_score}/{game_manager.game.current_blind}")
                    current_ante = game_manager.game.current_ante  # Store current ante before advancing
                    game_manager.next_ante()
                    loop_count = 0  # Reset loop counter when advancing to next ante
                    show_shop_next = True  # Show shop after beating a blind
                continue
    
    print("\n===== GAME OVER =====")
    print(f"Final ante: {((game_manager.game.current_ante - 1) // 3) + 1}")
    print(f"Final score: {game_manager.current_score}")
    print(f"Final money: ${game_manager.game.inventory.money}")
    print(f"Final jokers: {[j.name for j in game_manager.game.inventory.jokers]}")
    print(f"Game over: {game_manager.game_over}")
if __name__ == "__main__":
    simulate_game()