from JokerCreation import *
from Card import *
import pytest
def GJtest():
    green_joker = create_joker("Green Joker")
    
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
    
    hand1 = [
        Card(Suit.HEARTS, Rank.JACK),
        Card(Suit.DIAMONDS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.KING)
    ]
    effect1 = walkie_talkie.calculate_effect(hand1, 0, [], {'hand_type': 'high_card'})
    assert(effect1.chips == 0)
    assert(effect1.mult_add == 0)
    
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
    
    effect1 = rocket.calculate_effect([], 0, [], {})
    assert(effect1.money == 1)
    
    rocket.boss_blind_defeated = 1
    effect2 = rocket.calculate_effect([], 0, [], {})
    assert(effect2.money == 3)
    
    rocket.boss_blind_defeated = 3
    effect3 = rocket.calculate_effect([], 0, [], {})
    assert(effect3.money == 7)    

test_rocket_joker()

def test_clever_joker():
    clever = CleverJoker()
    
    effect1 = clever.calculate_effect([], 0, [], {'hand_type': 'high_card'})
    assert(effect1.chips ==0)
    
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
    assert(effect1.chips == 80 and effect2.chips == 0)

test_crafty()

def test_misprint():
    misprint = MisprintJoker()
    effect1 = misprint.calculate_effect([], 0, [], {'hand_type': 'three_of_kind'})
    assert(0 <= effect1.mult_add <= 23)

test_misprint()


def test_square_joker():
    square = SquareJoker()
    
    hand1 = [
        Card(Suit.HEARTS, Rank.JACK),
        Card(Suit.DIAMONDS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.KING),
        Card(Suit.SPADES, Rank.ACE)
    ]
    effect1 = square.calculate_effect(hand1, 0, [], {'hand_type': 'high_card'})
    assert(effect1.chips == 4)
    
    hand2 = [
        Card(Suit.HEARTS, Rank.JACK),
        Card(Suit.DIAMONDS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.KING)
    ]
    effect2 = square.calculate_effect(hand2, 0, [], {'hand_type': 'high_card'})
    assert(effect2.chips == 0)
    
    hand3 = [
        Card(Suit.HEARTS, Rank.JACK),
        Card(Suit.DIAMONDS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.KING),
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.HEARTS, Rank.TEN)
    ]
    effect3 = square.calculate_effect(hand3, 0, [], {'hand_type': 'high_card'})
    assert(effect3.chips == 0)
    

test_square_joker()


def test_faceless_joker():
    faceless = FacelessJoker()
    
    round_info = {'face_cards_discarded_count': 0}
    effect1 = faceless.calculate_effect([], 0, [], round_info)
    assert effect1.money == 0, f"Expected 0 money, got {effect1.money}"
    
    round_info = {'face_cards_discarded_count': 2}
    effect2 = faceless.calculate_effect([], 0, [], round_info)
    assert effect2.money == 0, f"Expected 0 money, got {effect2.money}"
    
    round_info = {'face_cards_discarded_count': 3}
    effect3 = faceless.calculate_effect([], 0, [], round_info)
    assert effect3.money == 6, f"Expected 6 money, got {effect3.money}"
    
    round_info = {'face_cards_discarded_count': 5}
    effect4 = faceless.calculate_effect([], 0, [], round_info)
    assert effect4.money == 6, f"Expected 6 money, got {effect4.money}"
    
test_faceless_joker()

def test_socks_buskin_retrigger():
    """Test the Socks and Buskin joker's retrigger effect"""
    from Card import Card
    from Enums import Suit, Rank, CardEnhancement, HandType
    from JokerCreation import SocksAndBuskinJoker
    from GameManager import GameManager
    
    gm = GameManager(seed=42)
    gm.start_new_game()
    
    socks_joker = SocksAndBuskinJoker()
    gm.game.inventory.add_joker(socks_joker)
    
    test_hand = [
        Card(Suit.HEARTS, Rank.JACK),
        Card(Suit.DIAMONDS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.KING),
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.HEARTS, Rank.TEN)
    ]
    
    gm.current_hand = []
    for card in test_hand:
        card.in_hand = True
        card.in_deck = False
        gm.current_hand.append(card)
    
    indices = list(range(len(gm.current_hand)))
    success, message = gm.play_cards(indices)
    
    print(f"Play result: {success}, {message}")
    print(f"Score: {gm.current_score}")
    
    retriggered_cards = [card for card in gm.played_cards if card.retrigger]
    print(f"Retriggered cards: {[str(card) for card in retriggered_cards]}")
    
    face_card_count = sum(1 for card in gm.played_cards if card.face)
    retrigger_count = len(retriggered_cards)
    
    assert retrigger_count == face_card_count, \
        f"Expected {face_card_count} retriggered cards, got {retrigger_count}"
    
    expected_rank_value = 0
    for card in gm.played_cards:
        if card.scored:
            if card.rank == Rank.ACE:
                expected_rank_value += 11
            elif card.face:
                expected_rank_value += 10
            else:
                expected_rank_value += card.rank.value
    
    expected_retrigger_value = 0
    for card in gm.played_cards:
        if card.retrigger and card.scored:
            if card.face:
                expected_retrigger_value += 10
            elif card.rank == Rank.ACE:
                expected_retrigger_value += 11
            else:
                expected_retrigger_value += card.rank.value
    
    print(f"Expected rank value contribution: {expected_rank_value}")
    print(f"Expected retrigger value contribution: {expected_retrigger_value}")
    print("Test passed! Socks and Buskin joker correctly marked face cards for retrigger.")
    
test_socks_buskin_retrigger()