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
        Rank.ACE.value: "A",
        Rank.TWO.value: "2", 
        Rank.THREE.value: "3",
        Rank.FOUR.value: "4",
        Rank.FIVE.value: "5",
        Rank.SIX.value: "6",
        Rank.SEVEN.value: "7",
        Rank.EIGHT.value: "8",
        Rank.NINE.value: "9",
        Rank.TEN.value: "10",
        Rank.JACK.value: "J",
        Rank.QUEEN.value: "Q",
        Rank.KING.value: "K"
    }
    
    suit_symbols = {
        Suit.HEARTS: "♥",
        Suit.DIAMONDS: "♦",
        Suit.CLUBS: "♣",
        Suit.SPADES: "♠",
        Suit.WILD: "★"
    }
    
    rank_str = rank_symbols.get(card.rank.value, str(card.rank.value))
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
    print(f"Ante: {game_manager.game.current_ante}, Blind: {game_manager.game.current_blind}")
    
    # Game loop
    while game_manager.game.current_ante <= 3:  # Just play the first few antes for demonstration
        print(f"\n----- Ante {game_manager.game.current_ante}: {game_manager.game.current_blind} -----")
        
        while not game_manager.current_ante_beaten:
            # Display the current hand
            print_hand(game_manager.current_hand)
            
            # Get a recommendation
            recommended_indices, reason = game_manager.get_recommended_play()
            print(f"Recommended play: {recommended_indices} - {reason}")
            
            # Decide whether to play or discard
            if game_manager.discards_used < game_manager.max_discards:
                # For this simulation, discard if we have a weak hand
                best_hand, _ = game_manager.get_best_hand_from_current()
                
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
            
            # Play the recommended cards
            if recommended_indices:
                success, message = game_manager.play_cards(recommended_indices)
                print(f"PLAY: {message}")
                
                # Print the played hand
                print_hand(game_manager.played_cards, "Played Cards")
                
                # Print contained hand types
                contained_types = [name for name, exists in game_manager.contained_hand_types.items() if exists]
                print(f"Contained hand types: {contained_types}")
                
                # Display the score
                print(f"Current score: {game_manager.current_score}/{game_manager.game.current_blind}")
                
                if game_manager.current_ante_beaten:
                    print(f"Ante beaten! Moving to next ante.")
                    game_manager.next_ante()
                    break
            else:
                # If there's no good play, try to discard if we still can
                if game_manager.discards_used < game_manager.max_discards:
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
                        print("No cards left to play. Moving to next ante.")
                        game_manager.next_ante()
                        break
    
    print("\n===== GAME SUMMARY =====")
    print(f"Final ante reached: {game_manager.game.current_ante}")
    print(f"Money earned: {game_manager.game.inventory.money}")
    print(f"Jokers collected: {len(game_manager.game.inventory.jokers)}")

if __name__ == "__main__":
    simulate_game()