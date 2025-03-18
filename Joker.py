class Joker(object):
    def __init__(self):
        self.rarity = 'base' # 'base', 'foil' +50 chips, 'holo' +10 mult, 'poly' 1.5x mult, 'negative' -1 joker space
        self.mult_effect = 0 
        self.chips_effect = 0
        self.played_hand_effect = 0
        self.left_most = False #left most joker for the brainstorm joker
        self.boss_blind_defeated = 0 #counts how many times this joker beat a boss blind for rocket joker
        self.price = 0 #shop price
        self.sell_value = 0 #shop selling value
        self.retrigger = False #does the joker get retriggered
        