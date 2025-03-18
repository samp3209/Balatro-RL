class Card(object):
    def __init__(self):
        self.spade = False
        self.club = False
        self.heart = False
        self.diamond = False
        self.rank = 0
        self.foil = False # foil +50 chips
        self.holo = False # holographic +10 mult
        self.poly = False # polychrome x1.5 mult
        self.face = False # is face card?
        self.played_this_ante = False # has this card been played in this round
        self.in_deck = True # is the card in the deck still
        self.in_hand = False # is the card in the hand played
        self.played = False # was the card played
        self.scored = False # did this card score when it was played?
        self.discarded = False # was the card discarded

        

