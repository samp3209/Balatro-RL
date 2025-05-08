import random
from typing import List, Optional, Tuple, Dict, Set
from Enums import *
from Card import Card
from Hand import hand
from Inventory import Inventory
from EnhancementManager import EnhancementManager


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
        
        #boss blind effects
        self.active_boss_blind_effect = None
        self.is_boss_blind = False
        self.face_down_cards = set() 
        self.forced_card_index = None 
        self.first_hand_dealt = False
        self.scored_cards_this_ante = [] 

        self.played_cards = []
        self.discarded_cards = []

        self.initialize_deck()
        
    def initialize_deck(self):
        """Initialize a standard 52-card deck"""
        self.inventory.initialize_standard_deck()
        self.inventory.shuffle_deck()

    def set_boss_blind_effect(self):
        """Set a random boss blind effect when a boss blind is reached"""
        is_boss_blind = (self.current_ante % 3 == 0)
        
        if is_boss_blind:
            self.is_boss_blind = True
            
            boss_effects = list(BossBlindEffect)
            effect_index = (self.current_ante // 3 - 1) % len(boss_effects)
            self.active_boss_blind_effect = boss_effects[effect_index]
            
            print(f"\n==================================================")
            print(f"🔥 BOSS BLIND ACTIVE: {self.active_boss_blind_effect.name} 🔥")
            print(f"==================================================\n")
            
            self.face_down_cards = set()
            self.forced_card_index = None
            self.first_hand_dealt = False
            self.scored_cards_this_ante = []
            
            if self.active_boss_blind_effect == BossBlindEffect.CLUB_DEBUFF:
                print("All Club cards are debuffed (contribute no value to hands)")
            elif self.active_boss_blind_effect == BossBlindEffect.HALVE_VALUES:
                print("All hand base values (chips and mult) are halved")
            elif self.active_boss_blind_effect == BossBlindEffect.FACE_CARDS_DOWN:
                print("All face cards (J, Q, K) are dealt face down")
            elif self.active_boss_blind_effect == BossBlindEffect.RANDOM_CARDS_DOWN:
                print("1 in 7 cards are dealt face down")
            elif self.active_boss_blind_effect == BossBlindEffect.DISCARD_RANDOM:
                print("2 random cards will be discarded whenever a hand is played")
            elif self.active_boss_blind_effect == BossBlindEffect.FIRST_HAND_DOWN:
                print("First 8 cards dealt are face down")
            elif self.active_boss_blind_effect == BossBlindEffect.PREVIOUS_CARDS_DEBUFF:
                print("Cards that were played previously in this ante are debuffed")
                self.scored_cards_this_ante = []
            elif self.active_boss_blind_effect == BossBlindEffect.FORCE_CARD_SELECTION:
                print("One card must be selected to be played/discarded each hand")
                
            return self.active_boss_blind_effect
        else:
            self.is_boss_blind = False
            self.active_boss_blind_effect = None
            return None
        
    def deal_hand(self, count=8) -> List[Card]:
            """Deal a specified number of cards from the deck to the hand"""
            hand = []
            
            if len(self.inventory.deck) < count:
                print(f"WARNING: Not enough cards in deck ({len(self.inventory.deck)}), need to reset deck")

            
            for _ in range(min(count, len(self.inventory.deck))):
                card = self.inventory.deck.pop(0)
                card.in_deck = False
                card.in_hand = True
                card.debuffed = False

                if self.is_boss_blind:
                    if self.active_boss_blind_effect == BossBlindEffect.CLUB_DEBUFF and card.suit == Suit.CLUBS:
                        card.debuffed = True
                        print(f"Boss Blind CLUB_DEBUFF: {card} is debuffed")
                    elif self.active_boss_blind_effect == BossBlindEffect.FACE_CARDS_DOWN and card.face:
                        self.face_down_cards.add(id(card))
                        print(f"Boss Blind FACE_CARDS_DOWN: {card} is face down")
                    elif self.active_boss_blind_effect == BossBlindEffect.RANDOM_CARDS_DOWN and random.randint(1, 7) == 1:
                        self.face_down_cards.add(id(card))
                        print(f"Boss Blind RANDOM_CARDS_DOWN: {card} is face down")
                    elif self.active_boss_blind_effect == BossBlindEffect.FIRST_HAND_DOWN and not self.first_hand_dealt:
                        self.face_down_cards.add(id(card))
                        print(f"Boss Blind FIRST_HAND_DOWN: {card} is face down")

                hand.append(card)

            if not self.first_hand_dealt:
                self.first_hand_dealt = True
            
            if self.is_boss_blind and self.active_boss_blind_effect == BossBlindEffect.FORCE_CARD_SELECTION:
                if hand:
                    self.forced_card_index = random.randint(0, len(hand) - 1)
                    print(f"Boss Blind FORCE_CARD_SELECTION: Card at index {self.forced_card_index} must be played/discarded")
                
            return hand
    
    def apply_card_enhancements(self, played_cards: List[Card], hand_type: HandType, 
                            base_mult: int, base_chips: int) -> Tuple[int, int]:
        """
        Apply all card enhancement effects to the score
        """
        total_mult, total_chips = EnhancementManager.apply_enhancement_effects(
            played_cards, 
            hand_type, 
            base_mult, 
            base_chips,
            self.is_boss_blind,
            self.active_boss_blind_effect
        )
        
        return total_mult, total_chips
    

    def process_post_hand_enhancements(self, played_cards: List[Card], hand_cards: List[Card]) -> Dict:
        """
        Process enhancement effects that happen after a hand is played or discarded
        """
        all_cards = played_cards + hand_cards
        result = EnhancementManager.process_enhancement_after_hand(all_cards, self.inventory)
        
        if 'money_gained' in result and result['money_gained'] > 0:
            self.inventory.money += result['money_gained']
            
        return result
    
    def handle_enhanced_deck_reset(self):
        """
        Handle special cases when resetting the deck
        (removing broken glass card)
        """
        cards_to_remove = []
        
        for card in self.inventory.deck:
            if card.enhancement == CardEnhancement.GLASS and card.check_glass_break():
                cards_to_remove.append(card)
                
                for master_card in list(self.inventory.master_deck):
                    if master_card is card:
                        self.inventory.master_deck.remove(master_card)
        
        for card in cards_to_remove:
            if card in self.inventory.deck:
                self.inventory.deck.remove(card)
        
        if cards_to_remove:
            print(f"Removed {len(cards_to_remove)} broken glass cards from the deck")


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
                
                if self.is_boss_blind:
                    if self.active_boss_blind_effect == BossBlindEffect.CLUB_DEBUFF and card.suit == Suit.CLUBS:
                        card.debuffed = True
                        print(f"Card {card} is a club - debuffed due to CLUB_DEBUFF effect!")
                    
                    elif self.active_boss_blind_effect == BossBlindEffect.PREVIOUS_CARDS_DEBUFF:
                        card_signature = (card.rank, card.suit)
                        
                        if hasattr(self, 'scored_cards_this_ante') and card_signature in self.scored_cards_this_ante:
                            card.debuffed = True
                            print(f"Card {card} was played previously in this ante - debuffed!")
        
        self.played_cards.extend(cards)
        
        if self.is_boss_blind and self.active_boss_blind_effect == BossBlindEffect.DISCARD_RANDOM:
            discarded = self.discard_random_cards(2)
            if discarded:
                print(f"Boss Blind DISCARD_RANDOM: Discarded {len(discarded)} random cards")
                for card in discarded:
                    print(f"  - Discarded {card}")
        
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
                card.played_this_ante = True
                if card.face:
                    self.face_cards_discarded_count += 1
        
        if self.is_boss_blind and self.active_boss_blind_effect == BossBlindEffect.DISCARD_RANDOM:
            discarded = self.discard_random_cards(2)
            if discarded:
                print(f"Boss Blind DISCARD_RANDOM: Discarded {len(discarded)} random cards after user discard")
                for card in discarded:
                    print(f"  - Discarded {card}")
                
        self.hands_discarded += 1

        return True
    

    def discard_random_cards(self, count: int) -> List[Card]:
        """
        Discard random cards from the player's hand
        Returns the list of discarded cards
        """
        cards_in_hand = [card for card in self.inventory.deck if card.in_hand and not card.played and not card.discarded]
        
        cards_to_discard = min(count, len(cards_in_hand))
        
        if cards_to_discard == 0:
            return []
            
        discard_indices = random.sample(range(len(cards_in_hand)), cards_to_discard)
        discarded_cards = [cards_in_hand[i] for i in discard_indices]
        
        for card in discarded_cards:
            card.discarded = True
            card.in_hand = False
            
            if card.face:
                self.face_cards_discarded_count += 1
        
        return discarded_cards

    
    def evaluate_played_hand(self, played_cards: List[Card]) -> Tuple[HandType, Dict[str, bool]]:
        """
        Evaluate the played cards to determine the best possible hand type
        Also identify all possible hand types contained within the played cards
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
            best_hand = HandType.FLUSH
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
            for card in played_cards:
                card.scored = False
                
            if hand_type == HandType.HIGH_CARD:
                highest_card = max(played_cards, key=lambda card: card.rank.value)
                highest_card.scored = True
                return
                
            if hand_type == HandType.PAIR or hand_type == HandType.TWO_PAIR or hand_type == HandType.THREE_OF_A_KIND or hand_type == HandType.FOUR_OF_A_KIND or hand_type == HandType.FULL_HOUSE:
                rank_counts = {}
                for card in played_cards:
                    rank_counts[card.rank.value] = rank_counts.get(card.rank.value, 0) + 1
                
                sorted_ranks = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
                
                if hand_type == HandType.PAIR:
                    pair_rank = sorted_ranks[0][0]
                    for card in played_cards:
                        if card.rank.value == pair_rank:
                            card.scored = True
                            
                elif hand_type == HandType.TWO_PAIR:
                    first_pair_rank = sorted_ranks[0][0]
                    second_pair_rank = sorted_ranks[1][0]
                    for card in played_cards:
                        if card.rank.value == first_pair_rank or card.rank.value == second_pair_rank:
                            card.scored = True
                            
                elif hand_type == HandType.THREE_OF_A_KIND:
                    three_rank = sorted_ranks[0][0]
                    for card in played_cards:
                        if card.rank.value == three_rank:
                            card.scored = True
                            
                elif hand_type == HandType.FOUR_OF_A_KIND:
                    four_rank = sorted_ranks[0][0]
                    for card in played_cards:
                        if card.rank.value == four_rank:
                            card.scored = True
                            
                elif hand_type == HandType.FULL_HOUSE:
                    three_rank = sorted_ranks[0][0]
                    pair_rank = sorted_ranks[1][0]
                    for card in played_cards:
                        if card.rank.value == three_rank or card.rank.value == pair_rank:
                            card.scored = True
                            
            elif hand_type == HandType.STRAIGHT or hand_type == HandType.STRAIGHT_FLUSH:
                unique_ranks = sorted(set(card.rank.value for card in played_cards))
                
                # Check for A-2-3-4-5 straight
                if set([14, 2, 3, 4, 5]).issubset(set(unique_ranks)):
                    straight_ranks = {14, 2, 3, 4, 5}
                    for card in played_cards:
                        if card.rank.value in straight_ranks:
                            card.scored = True
                    return
                    
                for i in range(len(unique_ranks) - 4):
                    if unique_ranks[i+4] - unique_ranks[i] == 4:
                        straight_ranks = set(range(unique_ranks[i], unique_ranks[i] + 5))
                        for card in played_cards:
                            if card.rank.value in straight_ranks:
                                card.scored = True
                        return
                        
            elif hand_type == HandType.FLUSH:
                suit_counts = {}
                for card in played_cards:
                    suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
                    
                most_common_suit = max(suit_counts.items(), key=lambda x: x[1])[0]
                
                suit_cards = [card for card in played_cards if card.suit == most_common_suit]
                suit_cards = sorted(suit_cards, key=lambda card: card.rank.value, reverse=True)
                for card in suit_cards[:5]:
                    card.scored = True

    def calculate_hand_score(self, hand_type: HandType) -> Tuple[int, int]:
        """
        Calculate the score for a hand type, applying inventory bonuses and boss blind effect
        Returns (multiplier, chips)
        """
        mult, chips = self.inventory.calculate_hand_value(hand_type, {
            'stake_multiplier': self.stake_multiplier,
            'count_all_played': getattr(self, 'count_all_played', False)
        })
        
        if self.is_boss_blind and self.active_boss_blind_effect == BossBlindEffect.HALVE_VALUES:
            print(f"Boss Blind effect HALVE_VALUES applied: {mult} → {max(1, mult // 2)} mult, {chips} → {max(5, chips // 2)} chips")
            mult = max(1, mult // 2)
            chips = max(5, chips // 2)
            
        return (mult, chips)
        
    def apply_joker_effects(self, played_cards: List[Card], hand_type: HandType, contained_hands: Dict[str, bool]) -> Tuple[int, int, int]:
        """
        Apply all joker effects based on the current state
        """
        base_mult, base_chips = self.inventory.calculate_hand_value(hand_type, {
            'stake_multiplier': self.stake_multiplier
        })
        
        if self.is_boss_blind and self.active_boss_blind_effect == BossBlindEffect.HALVE_VALUES:
            original_mult = base_mult
            original_chips = base_chips
            base_mult = max(1, base_mult // 2)
            base_chips = max(5, base_chips // 2)
            print(f"Boss Blind HALVE_VALUES: base values reduced from {original_mult} mult, {original_chips} chips to {base_mult} mult, {base_chips} chips")
        
        enhanced_mult, enhanced_chips = self.apply_card_enhancements(
            played_cards, 
            hand_type, 
            base_mult, 
            base_chips
        )
        
        total_mult = enhanced_mult
        money_gained = 0
        base_chips = enhanced_chips
        
        print(f"After enhancements: {total_mult} mult, {base_chips} chips")
        
        debuffed_cards = []
        for card in played_cards:
            if card.scored:
                is_debuffed = card.debuffed
                
                if (self.is_boss_blind and 
                    self.active_boss_blind_effect == BossBlindEffect.CLUB_DEBUFF and 
                    card.suit == Suit.CLUBS):
                    is_debuffed = True
                    card.debuffed = True
                
                if (self.is_boss_blind and 
                    self.active_boss_blind_effect == BossBlindEffect.PREVIOUS_CARDS_DEBUFF and 
                    card.played_this_ante):
                    is_debuffed = True
                    card.debuffed = True
                
                if is_debuffed:
                    debuffed_cards.append(card)
                    print(f"Card {card} is debuffed - will not contribute to joker effects or scoring")
        
        count_all_played = False
        round_info = {
            'hand_type': hand_type.name.lower(),
            'contained_hands': contained_hands,
            'hands_played': self.hands_played,
            'inventory': self.inventory,
            'max_discards': 4,
            'face_cards_discarded_count': self.face_cards_discarded_count,
            'debuffed_cards': debuffed_cards
        }
        
        non_debuffed_cards = []
        for card in played_cards:
            is_debuffed = card.debuffed
            if (self.is_boss_blind and 
                self.active_boss_blind_effect == BossBlindEffect.CLUB_DEBUFF and 
                card.suit == Suit.CLUBS):
                is_debuffed = True
            
            if (self.is_boss_blind and 
                self.active_boss_blind_effect == BossBlindEffect.PREVIOUS_CARDS_DEBUFF and 
                card.played_this_ante):
                is_debuffed = True
                
            if not is_debuffed:
                non_debuffed_cards.append(card)
        
        for joker in self.inventory.jokers:
            effect = joker.calculate_effect(
                non_debuffed_cards, 
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

        if count_all_played:
            base_mult, base_chips = self.inventory.calculate_hand_value(hand_type, {
                'stake_multiplier': self.stake_multiplier,
                'count_all_played': True
            })
            print(f"Splash Joker: Recalculated with all played cards: {base_mult} mult, {base_chips} chips")
    
        rank_chips = 0
        for card in played_cards:
            if card.scored:
                is_debuffed = card.debuffed
                
                if (self.is_boss_blind and 
                    self.active_boss_blind_effect == BossBlindEffect.CLUB_DEBUFF and 
                    card.suit == Suit.CLUBS):
                    is_debuffed = True
                    card.debuffed = True
                
                if (self.is_boss_blind and 
                    self.active_boss_blind_effect == BossBlindEffect.PREVIOUS_CARDS_DEBUFF and 
                    card.played_this_ante):
                    is_debuffed = True
                    card.debuffed = True 

                if is_debuffed:
                    card_value = 0
                    print(f"  • Card {card}: +0 chips (debuffed)")
                else:
                    card_value = self.calculate_rank_chip_value(card)
                    rank_chips += card_value
                    print(f"  • Card {card}: +{card_value} chips")
                    
                if self.is_boss_blind and self.active_boss_blind_effect == BossBlindEffect.HALVE_VALUES and not is_debuffed:
                    rank_chips = max(1, rank_chips // 2)
                    print(f"  • Boss Blind HALVE_VALUES applied: halved chip value")
        
        if rank_chips > 0:
            base_chips += rank_chips
            print(f"  • Total rank value: +{rank_chips} chips")
        
        retrigger_chips = 0
        for card in played_cards:
            if card.retrigger and card.scored:
                is_debuffed = card.debuffed
                
                if (self.is_boss_blind and 
                    self.active_boss_blind_effect == BossBlindEffect.CLUB_DEBUFF and 
                    card.suit == Suit.CLUBS):
                    is_debuffed = True
                
                if (self.is_boss_blind and 
                    self.active_boss_blind_effect == BossBlindEffect.PREVIOUS_CARDS_DEBUFF and 
                    card.played_this_ante):
                    is_debuffed = True
                
                if is_debuffed:
                    print(f"  • Retrigger effect from {card}: +0 chips (debuffed)")
                else:
                    card_value = self.calculate_rank_chip_value(card)
                    retrigger_chips += card_value
                    print(f"  • Retrigger effect from {card}: +{card_value} chips")
                
                if self.is_boss_blind and self.active_boss_blind_effect == BossBlindEffect.HALVE_VALUES and not is_debuffed:
                    retrigger_chips = max(1, retrigger_chips // 2)
                    print("  • Boss Blind HALVE_VALUES applied: halved retrigger value")

        if retrigger_chips > 0:
            base_chips += retrigger_chips
        print(f"  • Total retrigger bonus: +{retrigger_chips} chips")
        
        post_hand_effects = self.process_post_hand_enhancements(played_cards, [])
        if 'money_gained' in post_hand_effects:
            money_gained += post_hand_effects['money_gained']
        
        return (total_mult, base_chips, money_gained)

        
    def reset_for_new_round(self):
        """Reset game state for a new round with special handling for enhanced cards"""
        self.handle_enhanced_deck_reset()
        
        for card in self.inventory.deck + self.played_cards + self.discarded_cards:
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
        
        self.face_down_cards = set()
        self.forced_card_index = None
        self.first_hand_dealt = False
        self.scored_cards_this_ante = []
        
        if self.current_ante % 3 == 1 and not self.is_boss_blind:
            if hasattr(self, 'ante_played_cards'):
                self.ante_played_cards = []
        
        self.set_boss_blind_effect()

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
        

    def is_card_face_down(self, card: Card) -> bool:
        """
        Check if a card is face down due to a boss blind effect
        """
        return id(card) in self.face_down_cards
    
    def get_forced_card(self, hand: List[Card]) -> Optional[Card]:
        """
        Get the forced card from the hand if the FORCE_CARD_SELECTION boss blind effect is active
        """
        if (self.is_boss_blind and 
            self.active_boss_blind_effect == BossBlindEffect.FORCE_CARD_SELECTION and 
            self.forced_card_index is not None and 
            0 <= self.forced_card_index < len(hand)):
            return hand[self.forced_card_index]
        return None
    
    def track_scored_cards(self):
            """Track which cards have been scored in this ante for PREVIOUS_CARDS_DEBUFF effect"""
            if not hasattr(self, 'scored_cards_this_ante'):
                self.scored_cards_this_ante = []
            
            if self.is_boss_blind and self.active_boss_blind_effect == BossBlindEffect.PREVIOUS_CARDS_DEBUFF:
                for card in self.played_cards:
                    if card.scored:
                        card_signature = (card.rank, card.suit)
                        if card_signature not in self.scored_cards_this_ante:
                            self.scored_cards_this_ante.append(card_signature)
                            print(f"Tracking {card} as scored in this ante")