from JokerCreation import *
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

GJtest()