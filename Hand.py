class hand(object):
    def __init__(self):
        self.hands_left = 4
        self.discards_left = 4
        self.played_hand = None #what kind of hand was played 
        self.cards_in_deck = [] #what cards are still yet to be drawn
        self.cards_in_hand = [] #list of cards in hand
        self.best_hand_in_hand = None #what is the best hand that can be made out of cards in hand
        self.current_score = 0 #what is our score
        self.played_hand_size = 0 # size of hand played