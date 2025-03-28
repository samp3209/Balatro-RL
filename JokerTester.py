from JokerCreation import *
from Card import *
def GJtest():
    green_joker = create_joker("Green Joker")
    
    # Simulate a round
    round_info = {
        'hands_played': 3,
        'discards': 2,
        'hand_type': 'two_pair'
    }
    
    effect = green_joker.calculate_effect(
        hand=[],
        discards=4-round_info['discards'], 
        deck=[],
        round_info=round_info
    )
    
    print(f"Green Joker effect - Mult: {effect.mult}, Chips: {effect.chips}, Money: {effect.money}")

GJtest() # should print 1 mult 

def test_walkie_talkie_joker():
    print("Testing Walkie Talkie Joker:")
    walkie_talkie = WalkieTalkieJoker()
    
    # Scenario 1: Hand with no 10s or 4s
    hand1 = [
        Card(Suit.HEARTS, Rank.JACK),
        Card(Suit.DIAMONDS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.KING)
    ]
    effect1 = walkie_talkie.calculate_effect(hand1, 0, [], {'hand_type': 'high_card'})
    print("No 10s or 4s - Chips:", effect1.chips, "Mult:", effect1.mult)
    
    # Scenario 2: Hand with multiple 10s and 4s
    hand2 = [
        Card(Suit.HEARTS, Rank.TEN),
        Card(Suit.DIAMONDS, Rank.FOUR),
        Card(Suit.CLUBS, Rank.TEN)
    ]
    effect2 = walkie_talkie.calculate_effect(hand2, 0, [], {'hand_type': 'two_pair'})
    print("Two 10s and one 4 - Chips:", effect2.chips, "Mult:", effect2.mult)

def test_rocket_joker():
    print("\nTesting Rocket Joker:")
    rocket = RocketJoker()
    
    # Scenario 1: No boss blind defeated
    effect1 = rocket.calculate_effect([], 0, [], {})
    print("No boss blind defeated - Money:", effect1.money)
    
    # Scenario 2: One boss blind defeated
    rocket.boss_blind_defeated = 1
    effect2 = rocket.calculate_effect([], 0, [], {})
    print("One boss blind defeated - Money:", effect2.money)
    
    # Scenario 3: Multiple boss blinds defeated
    rocket.boss_blind_defeated = 3
    effect3 = rocket.calculate_effect([], 0, [], {})
    print("Three boss blinds defeated - Money:", effect3.money)

def test_clever_joker():
    print("\nTesting Clever Joker:")
    clever = CleverJoker()
    
    # Scenario 1: Not a two pair hand
    effect1 = clever.calculate_effect([], 0, [], {'hand_type': 'high_card'})
    print("High card hand - Chips:", effect1.chips)
    
    # Scenario 2: Two pair hand
    effect2 = clever.calculate_effect([], 0, [], {'hand_type': 'two_pair'})
    print("Two pair hand - Chips:", effect2.chips)

def test_delayed_gratification_joker():
    print("\nTesting Delayed Gratification Joker:")
    delayed_grat = DelayedGratificationJoker()
    
    # Scenario 1: Discards used
    effect1 = delayed_grat.calculate_effect([], 1, [], {'max_discards': 3})
    print("Discards used - Money:", effect1.money)
    
    # Scenario 2: No discards used
    effect2 = delayed_grat.calculate_effect([], 0, [], {'max_discards': 3})
    print("No discards used - Money:", effect2.money)
