import random
from typing import List, Optional, Tuple, Dict, Set
from Enums import *
from Card import Card
from Hand import hand
from Inventory import Inventory

class Game:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            
        self.inventory = Inventory()
        self.current_ante = 1
        self.current_blind = 300
        self.stake_multiplier = 1
        self.hands_played = 0
        self.hands_discarded = 0
        self.face_cards_discarded_count = 0
        
        self.initialize_deck()
        
    def initialize_deck(self):
        """Initialize a standard 52-card deck"""
        self.inventory.initialize_standard_deck()
        self.inventory.shuffle_deck()
        
    def deal_hand(self, count=8) -> List[Card]:
        """Deal a specified number of cards from the deck to the hand"""
        hand = []
        
        if len(self.inventory.deck) < count:
            print(f"WARNING: Not enough cards in deck ({len(self.inventory.deck)}), need to reset deck")

        
        for _ in range(min(count, len(self.inventory.deck))):
            card = self.inventory.deck.pop(0)
            card.in_deck = False
            card.in_hand = True
            hand.append(card)
            
        return hand
    
    def play_cards(self, cards: List[Card]) -> bool:
        """
        Mark specified cards as played
        Returns True if successful
        """
        if not cards:
            return False
            
        for card in cards:
            if card.in_hand and not card.played:
                card.played = True
                card.in_hand = False
                card.played_this_ante = True
                
        return True
    
    def discard_cards(self, cards: List[Card]) -> bool:
        """
        Mark specified cards as discarded
        Returns True if successful
        """
        if not cards:
            return False
            
        for card in cards:
            if card.in_hand and not card.discarded:
                card.discarded = True
                card.in_hand = False
                
                if card.face:
                    self.face_cards_discarded_count += 1
                
        self.hands_discarded += 1
        return True
    
    def evaluate_played_hand(self, played_cards: List[Card]) -> Tuple[HandType, Dict[str, bool]]:
        """
        Evaluate the played cards to determine the best possible hand type
        Also identify all possible hand types contained within the played cards
        
        Returns:
            Tuple of (best_hand_type, contained_hand_types)
            where contained_hand_types is a dict of {hand_type_name: exists_bool}
        """
        if not played_cards or len(played_cards) < 5:
            return (HandType.HIGH_CARD, {"high_card": True})
        
        sorted_cards = sorted(played_cards, key=lambda card: card.rank.value)
        
        contained_hands = {
            "high_card": True,
            "pair": False,
            "two_pair": False,
            "three_of_kind": False,
            "straight": False,
            "flush": False,
            "full_house": False,
            "four_of_kind": False,
            "straight_flush": False
        }
        
        rank_counts = {}
        suit_counts = {}
        
        for card in played_cards:
            if card.enhancement == CardEnhancement.WILD:
                for suit in Suit:
                    if suit != Suit.WILD:
                        suit_counts[suit] = suit_counts.get(suit, 0) + 1
            else:
                suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
                
            rank_counts[card.rank.value] = rank_counts.get(card.rank.value, 0) + 1
        
        pairs = []
        three_of_kinds = []
        four_of_kinds = []
        
        for rank, count in rank_counts.items():
            if count == 2:
                pairs.append(rank)
            elif count == 3:
                three_of_kinds.append(rank)
            elif count == 4:
                four_of_kinds.append(rank)
        
        flush = any(count >= 5 for count in suit_counts.values())
        if flush:
            contained_hands["flush"] = True
        
        straight = False
        unique_ranks = set(card.rank.value for card in played_cards)
        
        #Check for A-2-3-4-5 straight 
        if {14, 2, 3, 4, 5}.issubset(unique_ranks):
            straight = True
        
        #Check for  straight
        for i in range(2, 11):
            if all(r in unique_ranks for r in range(i, i+5)):
                straight = True
                break
                
        if straight:
            contained_hands["straight"] = True
        
        if len(pairs) >= 1:
            contained_hands["pair"] = True
            
        if len(pairs) >= 2:
            contained_hands["two_pair"] = True
            
        if len(three_of_kinds) >= 1:
            contained_hands["three_of_kind"] = True
            
        if len(four_of_kinds) >= 1:
            contained_hands["four_of_kind"] = True
            
        if (len(three_of_kinds) >= 1 and len(pairs) >= 1) or len(three_of_kinds) >= 2:
            contained_hands["full_house"] = True
            
        best_hand = HandType.HIGH_CARD
        
        if contained_hands["straight_flush"]:
            best_hand = HandType.STRAIGHT_FLUSH
        elif contained_hands["four_of_kind"]:
            best_hand = HandType.FOUR_OF_A_KIND
        elif contained_hands["full_house"]:
            best_hand = HandType.FULL_HOUSE
        elif contained_hands["flush"]:
            best_hand = HandType.STRAIGHT
        elif contained_hands["straight"]:
            best_hand = HandType.STRAIGHT
        elif contained_hands["three_of_kind"]:
            best_hand = HandType.THREE_OF_A_KIND
        elif contained_hands["two_pair"]:
            best_hand = HandType.TWO_PAIR
        elif contained_hands["pair"]:
            best_hand = HandType.PAIR
            
        self._mark_scoring_cards(played_cards, best_hand)
            
        return (best_hand, contained_hands)
    
    def _mark_scoring_cards(self, played_cards: List[Card], hand_type: HandType):
        """Mark cards that contribute to the scoring hand"""
        if hand_type == HandType.HIGH_CARD:
            highest_card = max(played_cards, key=lambda card: card.rank.value)
            highest_card.scored = True
            return
            

        sorted_cards = sorted(played_cards, key=lambda card: card.rank.value, reverse=True)
        
        if hand_type == HandType.PAIR:
            rank_counts = {}
            for card in played_cards:
                rank_counts[card.rank.value] = rank_counts.get(card.rank.value, 0) + 1
                
            pair_rank = max([rank for rank, count in rank_counts.items() if count >= 2])
            
            for card in played_cards:
                if card.rank.value == pair_rank:
                    card.scored = True
        

        
    def calculate_hand_score(self, hand_type: HandType) -> Tuple[int, int]:
        """
        Calculate the score for a hand type, applying inventory bonuses
        Returns (multiplier, chips)
        """
        return self.inventory.calculate_hand_value(hand_type, {
            'stake_multiplier': self.stake_multiplier,
            'count_all_played': self.count_all_played
        })
        
    def apply_joker_effects(self, played_cards: List[Card], hand_type: HandType, contained_hands: Dict[str, bool]) -> Tuple[int, int, int]:
        """
        Apply all joker effects based on the current state
        
        Returns:
            Tuple of (total_mult, base_chips, money_gained)
            The caller should multiply total_mult by base_chips to get the final score
        """
        base_mult, base_chips = self.inventory.calculate_hand_value(hand_type, {
            'stake_multiplier': self.stake_multiplier
        })
        
        total_mult = base_mult
        money_gained = 0
        
        print(f"Base values: {base_mult} mult, {base_chips} chips")
        
        count_all_played = False
        round_info = {
            'hand_type': hand_type.name.lower(),
            'contained_hands': contained_hands,
            'hands_played': self.hands_played,
            'inventory': self.inventory,
            'max_discards': 4,
            'face_cards_discarded_count': self.face_cards_discarded_count
        }
        
        for joker in self.inventory.jokers:
            print(f"Applying {joker.name} effect...")
            effect = joker.calculate_effect(
                played_cards, 
                self.hands_discarded, 
                self.inventory.deck, 
                round_info
            )
            if hasattr(effect, 'count_all_played') and effect.count_all_played:
                count_all_played = True
            old_mult = total_mult
            
            total_mult += effect.mult_add
            base_chips += effect.chips
            
            total_mult *= effect.mult_mult
            
            money_gained += effect.money
            
            print(f"  • {joker.name}: +{effect.mult_add} mult, x{effect.mult_mult} mult, +{effect.chips} chips, +${effect.money}")
            print(f"  • Result: {old_mult} → {total_mult} mult, {base_chips} chips")
            
        if count_all_played:
            base_mult, base_chips = self.inventory.calculate_hand_value(hand_type, {
                'stake_multiplier': self.stake_multiplier,
                'count_all_played': True
            })
            print(f"Splash Joker: Recalculated with all played cards: {base_mult} mult, {base_chips} chips")
       
        rank_chips = 0
        for card in played_cards:
            if card.scored:
                card_value = self.calculate_rank_chip_value(card)
                rank_chips += card_value
                print(f"  • Card {card}: +{card_value} chips")
        
        if rank_chips > 0:
            base_chips += rank_chips
            print(f"  • Total rank value: +{rank_chips} chips")
        
        retrigger_chips = 0
        for card in played_cards:
            if card.retrigger and card.scored:
                card_value = self.calculate_rank_chip_value(card)
                retrigger_chips += card_value
                print(f"  • Retrigger effect from {card}: +{card_value} chips")
        
        if retrigger_chips > 0:
            base_chips += retrigger_chips
        print(f"  • Total retrigger bonus: +{retrigger_chips} chips")

        return (total_mult, base_chips, money_gained)
        
    def reset_for_new_round(self):
        """Reset game state for a new round"""
        for card in self.inventory.deck + [c for c in self.played_cards] + [c for c in self.discarded_cards]:
            card.reset_state()
            
        self.hands_played = 0
        self.hands_discarded = 0
        self.face_cards_discarded_count = 0
        
        for joker in self.inventory.jokers:
            if hasattr(joker, 'reset'):
                joker.reset()
                
        self.inventory.reset_deck(self.played_cards, self.discarded_cards, [])
        self.played_cards = []
        self.discarded_cards = []

    def calculate_rank_chip_value(self, card: Card) -> int:
        """
        Calculate the chip value of a card based on its rank.
        Face cards (J, Q, K) are worth 10, Ace is worth 11, and 
        numbered cards are worth their rank value.
        """
        if card.rank == Rank.ACE:
            return 11
        elif card.face:
            return 10
        else:
            return card.rank.value