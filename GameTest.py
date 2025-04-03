import random
from typing import List, Dict, Tuple
from Enums import *
from Card import Card
from Inventory import Inventory
from Game import Game
from HandEvaluator import *
from GameManager import GameManager
from JokerCreation import create_joker

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
        
    return f"{rank_str}{suit_str}{enhancement}"

def print_hand(cards: List[Card], title: str = "Current Hand"):
    """Print a formatted display of cards"""
    print(f"\n=== {title} ===")
    for i, card in enumerate(cards):
        status = ""
        if card.scored:
            status = " (scoring)"
        print(f"{i}: {print_card(card)}{status}")
    print()

def simulate_game():
    """Run a simulation of the game"""
    # Initialize with a fixed seed for reproducibility
    game_manager = GameManager(seed=42)
    game_manager.start_new_game()
    
    # Add some jokers to the inventory for testing
    jokers = ["Green Joker", "Even Steven", "Clever", "Smiley"]
    for joker_name in jokers:
        joker = create_joker(joker_name)
        if joker:
            game_manager.game.inventory.add_joker(joker)
            print(f"Added {joker_name} to inventory")
    
    # First ante - Small blind 300
    print("\n===== STARTING GAME =====")
    print(f"Current Ante: {game_manager.game.current_ante}, Blind: {game_manager.game.current_blind}")
    print(f"Jokers in inventory: {[j.name for j in game_manager.game.inventory.jokers]}")
    
    # Track total rounds played to avoid infinite loops
    max_rounds = 30
    rounds_played = 0
    
    # Game loop - now we'll play through multiple blinds
    max_antes = 3  # Play through 3 antes (9 blinds total)
    current_ante = 1
    
    while current_ante <= max_antes and rounds_played < max_rounds:
        # Print current blind info
        blind_type = "Boss"
        if game_manager.game.current_ante % 3 == 1:
            blind_type = "Small"
        elif game_manager.game.current_ante % 3 == 2:
            blind_type = "Medium"
        
        print(f"\n----- Ante {current_ante}, {blind_type} Blind: {game_manager.game.current_blind} -----")
        
        # Reset for a new ante if needed
        if game_manager.current_ante_beaten:
            print(f"Blind beaten! Moving to next blind.")
            game_manager.next_ante()
            if game_manager.game.current_ante % 3 == 1 and game_manager.game.current_ante > 1:
                current_ante += 1
            rounds_played = 0
            continue
            
        # Increment our round counter
        rounds_played += 1
        print(f"Round {rounds_played}/{max_rounds}")
        
        # Display the current hand
        print_hand(game_manager.current_hand)
        
        # Safety check for empty hand
        if not game_manager.current_hand:
            print("No cards in hand - dealing new hand")
            game_manager.deal_new_hand()
            continue
        
        # Get a recommendation
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
        
        # Decide whether to play or discard
        if game_manager.discards_used < game_manager.max_discards:
            # For this simulation, discard if we have a weak hand
            try:
                best_hand_info = game_manager.get_best_hand_from_current()
                if best_hand_info:
                    best_hand, _ = best_hand_info
                    if best_hand.value <= HandType.PAIR.value and random.random() < 0.7:
                        # Discard weak cards
                        discard_indices = []
                        for i, card in enumerate(game_manager.current_hand):
                            # Simple strategy: discard low cards that aren't part of a pair
                            if card.rank.value < 10 and random.random() < 0.5:
                                discard_indices.append(i)
                                
                        if discard_indices:
                            success, message = game_manager.discard_cards(discard_indices)
                            print(f"DISCARD: {message}")
                            continue
            except Exception as e:
                print(f"Error in discard strategy: {e}")
        
        # Play the recommended cards
        if recommended_indices:
            try:
                success, message = game_manager.play_cards(recommended_indices)
                print(f"PLAY: {message}")
                
                # Print the played hand
                print_hand(game_manager.played_cards, "Played Cards")
                
                # Print contained hand types
                if game_manager.contained_hand_types:
                    contained_types = [name for name, exists in game_manager.contained_hand_types.items() if exists]
                    print(f"Contained hand types: {contained_types}")
                
                # Display the score
                print(f"Current score: {game_manager.current_score}/{game_manager.game.current_blind}")
            except Exception as e:
                print(f"Error playing cards: {e}")
        else:
            # If there's no good play, try to discard if we still can
            if game_manager.discards_used < game_manager.max_discards and game_manager.current_hand:
                discard_indices = list(range(min(4, len(game_manager.current_hand))))
                success, message = game_manager.discard_cards(discard_indices)
                print(f"DISCARD: {message}")
            else:
                # We've run out of options, have to play something
                if game_manager.current_hand:
                    play_indices = list(range(min(5, len(game_manager.current_hand))))
                    success, message = game_manager.play_cards(play_indices)
                    print(f"FORCED PLAY: {message}")
                    print_hand(game_manager.played_cards, "Played Cards")
                else:
                    print("No cards left to play. Dealing new hand.")
                    game_manager.deal_new_hand()
    
    print("\n===== GAME SUMMARY =====")
    print(f"Final ante reached: {current_ante}")
    print(f"Final blind: {game_manager.game.current_blind}")
    print(f"Money earned: {game_manager.game.inventory.money}")
    print(f"Jokers collected: {len(game_manager.game.inventory.jokers)}")
    
if __name__ == "__main__":
    simulate_game()