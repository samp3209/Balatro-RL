from JokerCreation import *
from Card import *
import pytest
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
    
    assert(effect.mult_add == 1)

GJtest()

def test_walkie_talkie_joker():
    walkie_talkie = WalkieTalkieJoker()
    
    # Scenario 1: Hand with no 10s or 4s
    hand1 = [
        Card(Suit.HEARTS, Rank.JACK),
        Card(Suit.DIAMONDS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.KING)
    ]
    effect1 = walkie_talkie.calculate_effect(hand1, 0, [], {'hand_type': 'high_card'})
    assert(effect1.chips == 0)
    assert(effect1.mult_add == 0)
    
    # Scenario 2: Hand with multiple 10s and 4s
    hand2 = [
        Card(Suit.HEARTS, Rank.TEN),
        Card(Suit.DIAMONDS, Rank.FOUR),
        Card(Suit.CLUBS, Rank.TEN),
        Card(Suit.CLUBS, Rank.FOUR)
    ]
    hand2[0].scored = True
    hand2[1].scored = True
    hand2[2].scored = True
    hand2[3].scored = True
    effect2 = walkie_talkie.calculate_effect(hand2, 0, [], {'hand_type': 'two_pair'})
    assert(effect2.chips == 40)
    assert(effect2.mult_add == 16)

test_walkie_talkie_joker()

def test_rocket_joker():
    rocket = RocketJoker()
    
    # Scenario 1: No boss blind defeated
    effect1 = rocket.calculate_effect([], 0, [], {})
    assert(effect1.money == 1)
    
    # Scenario 2: One boss blind defeated
    rocket.boss_blind_defeated = 1
    effect2 = rocket.calculate_effect([], 0, [], {})
    assert(effect2.money == 3)
    
    # Scenario 3: Multiple boss blinds defeated
    rocket.boss_blind_defeated = 3
    effect3 = rocket.calculate_effect([], 0, [], {})
    assert(effect3.money == 7)    

test_rocket_joker()

def test_clever_joker():
    clever = CleverJoker()
    
    # Scenario 1: Not a two pair hand
    effect1 = clever.calculate_effect([], 0, [], {'hand_type': 'high_card'})
    assert(effect1.chips ==0)
    
    # Scenario 2: Two pair hand
    effect2 = clever.calculate_effect([], 0, [], {'hand_type': 'two_pair'})
    assert(effect2.chips == 80)

test_clever_joker()

def test_delayed_gratification_joker():
    delayed_grat = DelayedGratificationJoker()
    
    effect1 = delayed_grat.calculate_effect([], 1, [], {'max_discards': 4})
    assert(effect1.money == 0)
    
    effect2 = delayed_grat.calculate_effect([], 0, [], {'max_discards': 4})
    assert(effect2.money == 8)

test_delayed_gratification_joker()

def test_mad_joker():
    mad = MadJoker()
    hand1 = [
        Card(Suit.HEARTS, Rank.JACK),
        Card(Suit.DIAMONDS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.KING)
    ]
    hand2 = [
        Card(Suit.HEARTS, Rank.TEN),
        Card(Suit.DIAMONDS, Rank.FOUR),
        Card(Suit.CLUBS, Rank.TEN),
        Card(Suit.CLUBS, Rank.FOUR)
    ]
    effect1 = mad.calculate_effect(hand1, 0, [], {'hand_type': 'high_card'})
    effect2 = mad.calculate_effect(hand2, 0, [], {'hand_type': 'two_pair'})
    effect3 = mad.calculate_effect([], 0,[], {'hand_type': 'full_house'})
    assert(effect1.mult_add == 0 and effect2.mult_add == 10 and effect3.mult_add == 10)

test_mad_joker()

def test_Wily_joker():
    wily = WilyJoker()
    effect1 = wily.calculate_effect([], 0, [], {'hand_type': 'three_of_kind'})
    effect2 = wily.calculate_effect([], 0, [], {'hand_type': 'full_house'})
    effect3 = wily.calculate_effect([], 0, [], {'hand_type': 'four_of_kind'})
    effect4 = wily.calculate_effect([], 0, [], {'hand_type': 'high_card'})
    assert(effect1.chips == 100 and effect2.chips == 100 and effect3.chips == 100 and effect4.chips == 0)

test_Wily_joker()

def test_crafty():
    crafty = CraftyJoker()
    effect1 = crafty.calculate_effect([], 0, [], {'hand_type': 'flush'})
    effect2 = crafty.calculate_effect([], 0, [], {'hand_type': 'high_card'})
    assert(effect1.chips == 80 and effect2 == 0)

test_crafty()

def test_misprint():
    misprint = MisprintJoker()
    effect1 = misprint.calculate_effect([], 0, [], {'hand_type': 'three_of_kind'})
    assert(0 <= effect1.mult_add <= 23)

test_misprint()