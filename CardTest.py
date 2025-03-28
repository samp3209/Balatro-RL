from Card import *
from Enums import *
def cardTest():
    card = Card(Suit.SPADES, Rank.KING)
    print(f"Is face card: {card.face}")
    
    card.apply_enhancement(CardEnhancement.FOIL)
    
    print(f"Chip bonus: {card.get_chip_bonus()}")
    print(f"Mult bonus: {card.get_mult_add()}")
    
    card.reset_state()
    print(f"Chip bonus: {card.get_chip_bonus()}") 

cardTest()

def aceTest():
    card = Card(Suit.HEARTS, Rank.ACE)
    print(f"is face card: {card.face}")

aceTest()