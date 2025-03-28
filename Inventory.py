class inventory(object):
    def __init__(self):
        self.jokers = [] #list of jokers
        self.consumables = [] #max is 2
        self.last_tarot = None #last played tarot card 
        self.joker_sell_values = [] #list of joker sell values
        self.pluto_lvl = 1 #default 1 mult x 5 chips, each increment increases pluto effect by 1+ mult 10+ chips
        self.mercury_lvl = 1 #default 2 mult x 10 chips, each increase by 1 mult 15 chips
        self.uranus_lvl = 1 #default 2 mutl x 20 chips, each increase by +1 mult +20 chips
        self.venus_lvl = 1 #default 3 mult x 30 chips, each increase by +2 mult +20 chips
        self.saturn_lvl = 1 #default 4 mult x 30 chips, each increase by +3 mult +30 chips
        self.earth_lvl = 1 #default 4 mult x 40 chips, each increase by +2 mult +30 chips
        self.mars_lvl = 1 #default 7 mult x 60 chips, each increase by +3 mult +30 chips
        self.neptune_lvl = 1 #default 8 mult x 100 chips, each increase by +4 mult +40 chips
        self.booster_skip = 0 #counts how many boosters we skipped
        self.tarot_used = 0 #counts tarot cards used
        self.uncommon_joker_count = 0

