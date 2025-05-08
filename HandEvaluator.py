from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
from Enums import *
from Card import Card

class HandPattern:
    """Class to represent a detected pattern in a hand of cards"""
    def __init__(self, pattern_type: HandType, cards: List[Card]):
        self.pattern_type = pattern_type
        self.cards = cards
        
    def __repr__(self):
        return f"{self.pattern_type.name}: {[str(card) for card in self.cards]}"

class HandEvaluator:
    """
    Advanced analyzer that identifies all possible hand patterns within a set of cards
    with special handling for enhanced cards
    """
    
    @staticmethod
    def analyze_hand(cards: List[Card]) -> Tuple[HandType, Dict[HandType, List[HandPattern]]]:
        """
        Analyze a hand to find all patterns within it, accounting for card enhancements
        
        Args:
            cards: List of cards to analyze
            
        Returns:
            Tuple containing:
            - The best hand type found
            - Dictionary mapping hand types to lists of patterns of that type
        """
        if not cards:
            return (HandType.HIGH_CARD, {})
            
        patterns = {
            HandType.HIGH_CARD: [],
            HandType.PAIR: [],
            HandType.TWO_PAIR: [],
            HandType.THREE_OF_A_KIND: [],
            HandType.STRAIGHT: [],
            HandType.FLUSH: [],
            HandType.FULL_HOUSE: [],
            HandType.FOUR_OF_A_KIND: [],
            HandType.STRAIGHT_FLUSH: []
        }
        
        HandEvaluator._find_high_cards(cards, patterns)
        HandEvaluator._find_pairs(cards, patterns)
        HandEvaluator._find_three_of_a_kinds(cards, patterns)
        HandEvaluator._find_four_of_a_kinds(cards, patterns)
        HandEvaluator._find_straights(cards, patterns)
        HandEvaluator._find_flushes(cards, patterns)
        
        HandEvaluator._find_two_pairs(patterns)
        HandEvaluator._find_full_houses(patterns)
        HandEvaluator._find_straight_flushes(cards, patterns)
        
        best_type = HandType.HIGH_CARD
        for hand_type in [HandType.STRAIGHT_FLUSH, HandType.FOUR_OF_A_KIND, 
                        HandType.FULL_HOUSE, HandType.FLUSH, HandType.STRAIGHT,
                        HandType.THREE_OF_A_KIND, HandType.TWO_PAIR, HandType.PAIR]:
            if patterns[hand_type]:
                best_type = hand_type
                break
                
        return (best_type, patterns)
    
    @staticmethod
    def _find_high_cards(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find high cards in the hand, accounting for STONE cards"""
        sorted_cards = sorted(cards, key=lambda card: card.rank.value, reverse=True)
        
        non_stone_cards = [card for card in sorted_cards if card.enhancement != CardEnhancement.STONE]
        
        if non_stone_cards:
            patterns[HandType.HIGH_CARD].append(HandPattern(HandType.HIGH_CARD, [non_stone_cards[0]]))
        elif sorted_cards:
            patterns[HandType.HIGH_CARD].append(HandPattern(HandType.HIGH_CARD, [sorted_cards[0]]))
    
    @staticmethod
    def _find_pairs(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find all pairs in the hand, handling STONE cards and WILD cards"""
        rank_groups = defaultdict(list)
        for card in cards:
            if card.enhancement != CardEnhancement.STONE:
                rank_groups[card.rank.value].append(card)
            
        for rank, group in rank_groups.items():
            if len(group) >= 2:
                for i in range(len(group) - 1):
                    for j in range(i + 1, len(group)):
                        pair_cards = [group[i], group[j]]
                        patterns[HandType.PAIR].append(HandPattern(HandType.PAIR, pair_cards))
    
    @staticmethod
    def _find_three_of_a_kinds(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find all three of a kinds in the hand, handling STONE and WILD cards"""
        rank_groups = defaultdict(list)
        for card in cards:
            if card.enhancement != CardEnhancement.STONE:
                rank_groups[card.rank.value].append(card)
            
        for rank, group in rank_groups.items():
            if len(group) >= 3:
                for i in range(len(group) - 2):
                    for j in range(i + 1, len(group) - 1):
                        for k in range(j + 1, len(group)):
                            three_cards = [group[i], group[j], group[k]]
                            patterns[HandType.THREE_OF_A_KIND].append(
                                HandPattern(HandType.THREE_OF_A_KIND, three_cards)
                            )
    
    @staticmethod
    def _find_four_of_a_kinds(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find all four of a kinds in the hand, handling STONE and WILD cards"""
        rank_groups = defaultdict(list)
        for card in cards:
            if card.enhancement != CardEnhancement.STONE:
                rank_groups[card.rank.value].append(card)
            
        for rank, group in rank_groups.items():
            if len(group) >= 4:
                four_cards = group[:4]  
                patterns[HandType.FOUR_OF_A_KIND].append(
                    HandPattern(HandType.FOUR_OF_A_KIND, four_cards)
                )
    
    @staticmethod
    def _find_straights(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find all straights in the hand, handling STONE and WILD cards"""
        non_stone_cards = [card for card in cards if card.enhancement != CardEnhancement.STONE]
        
        unique_ranks = sorted(set(card.rank.value for card in non_stone_cards))
        
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i+4] - unique_ranks[i] == 4:
                straight_ranks = set(range(unique_ranks[i], unique_ranks[i] + 5))
                straight_cards = [card for card in non_stone_cards if card.rank.value in straight_ranks]
                
                final_straight = []
                for rank in range(unique_ranks[i], unique_ranks[i] + 5):
                    for card in straight_cards:
                        if card.rank.value == rank:
                            final_straight.append(card)
                            break
                            
                if len(final_straight) == 5:
                    patterns[HandType.STRAIGHT].append(
                        HandPattern(HandType.STRAIGHT, final_straight)
                    )
        
        if set([14, 2, 3, 4, 5]).issubset(set(unique_ranks)):
            straight_cards = []
            for rank in [14, 2, 3, 4, 5]:
                for card in non_stone_cards:
                    if card.rank.value == rank:
                        straight_cards.append(card)
                        break
                        
            if len(straight_cards) == 5:
                patterns[HandType.STRAIGHT].append(
                    HandPattern(HandType.STRAIGHT, straight_cards)
                )
    
    @staticmethod
    def _find_flushes(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find all flushes in the hand, handling WILD and STONE cards"""
        non_stone_cards = [card for card in cards if card.enhancement != CardEnhancement.STONE]
        
        suit_groups = defaultdict(list)
        wild_cards = []
        
        for card in non_stone_cards:
            if card.enhancement == CardEnhancement.WILD:
                wild_cards.append(card)
            else:
                suit_groups[card.suit].append(card)
        
        for suit in suit_groups:
            suit_groups[suit].extend(wild_cards)
        
        for suit, group in suit_groups.items():
            if len(group) >= 5:
                sorted_group = sorted(group, key=lambda card: card.rank.value, reverse=True)
                flush_cards = sorted_group[:5] 
                
                patterns[HandType.FLUSH].append(
                    HandPattern(HandType.FLUSH, flush_cards)
                )
    
    @staticmethod
    def _find_two_pairs(patterns: Dict[HandType, List[HandPattern]]):
        """Find all two pair combinations using already identified pairs"""
        pairs = patterns[HandType.PAIR]
        
        if len(pairs) >= 2:
            for i in range(len(pairs) - 1):
                for j in range(i + 1, len(pairs)):
                    pair1 = pairs[i]
                    pair2 = pairs[j]
                    
                    pair1_ids = {id(card) for card in pair1.cards}
                    pair2_ids = {id(card) for card in pair2.cards}
                    
                    if not pair1_ids.intersection(pair2_ids):
                        pair1_ranks = {card.rank.value for card in pair1.cards 
                                      if card.enhancement != CardEnhancement.WILD}
                        pair2_ranks = {card.rank.value for card in pair2.cards 
                                      if card.enhancement != CardEnhancement.WILD}
                        
                        if not pair1_ranks.intersection(pair2_ranks) or len(pair1_ranks) == 0 or len(pair2_ranks) == 0:
                            two_pair_cards = pair1.cards + pair2.cards
                            patterns[HandType.TWO_PAIR].append(
                                HandPattern(HandType.TWO_PAIR, two_pair_cards)
                            )
    
    @staticmethod
    def _find_full_houses(patterns: Dict[HandType, List[HandPattern]]):
        """Find all full house combinations using three of a kinds and pairs"""
        three_of_a_kinds = patterns[HandType.THREE_OF_A_KIND]
        pairs = patterns[HandType.PAIR]
        
        if not three_of_a_kinds or not pairs:
            return
            
        for three in three_of_a_kinds:
            three_rank = None
            for card in three.cards:
                if card.enhancement != CardEnhancement.WILD:
                    three_rank = card.rank.value
                    break
            
            if three_rank is None: 
                three_rank = -1
            
            for pair in pairs:
                pair_rank = None
                for card in pair.cards:
                    if card.enhancement != CardEnhancement.WILD:
                        pair_rank = card.rank.value
                        break
                
                if pair_rank is None:
                    pair_rank = -2
                
                three_ids = {id(card) for card in three.cards}
                pair_ids = {id(card) for card in pair.cards}
                
                if not three_ids.intersection(pair_ids) and (three_rank != pair_rank or three_rank == -1 or pair_rank == -2):
                    full_house_cards = three.cards + pair.cards
                    patterns[HandType.FULL_HOUSE].append(
                        HandPattern(HandType.FULL_HOUSE, full_house_cards)
                    )
    
    @staticmethod
    def _find_straight_flushes(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find straight flushes by looking for straights that are also flushes, handling enhanced cards"""
        non_stone_cards = [card for card in cards if card.enhancement != CardEnhancement.STONE]
        
        suit_groups = defaultdict(list)
        wild_cards = []
        
        for card in non_stone_cards:
            if card.enhancement == CardEnhancement.WILD:
                wild_cards.append(card)
            else:
                suit_groups[card.suit].append(card)
        
        for suit in suit_groups:
            suit_groups[suit].extend(wild_cards)
        
        for suit, suited_cards in suit_groups.items():
            if len(suited_cards) >= 5:
                rank_values = sorted(set(card.rank.value for card in suited_cards))
                
                for i in range(len(rank_values) - 4):
                    if rank_values[i+4] - rank_values[i] == 4:
                        straight_ranks = set(range(rank_values[i], rank_values[i] + 5))
                        straight_flush_cards = []
                        
                        for rank in straight_ranks:
                            natural_card = None
                            for card in suited_cards:
                                if card.rank.value == rank and card.enhancement != CardEnhancement.WILD:
                                    natural_card = card
                                    break
                            
                            if natural_card:
                                straight_flush_cards.append(natural_card)
                            else:
                                for card in suited_cards:
                                    if card.enhancement == CardEnhancement.WILD and not any(id(card) == id(c) for c in straight_flush_cards):
                                        straight_flush_cards.append(card)
                                        break
                        
                        if len(straight_flush_cards) == 5:
                            patterns[HandType.STRAIGHT_FLUSH].append(
                                HandPattern(HandType.STRAIGHT_FLUSH, straight_flush_cards)
                            )
                
                if set([14, 2, 3, 4, 5]).issubset(set(rank_values)):
                    straight_flush_cards = []
                    for rank in [14, 2, 3, 4, 5]:
                        natural_card = None
                        for card in suited_cards:
                            if card.rank.value == rank and card.enhancement != CardEnhancement.WILD:
                                natural_card = card
                                break
                        
                        if natural_card:
                            straight_flush_cards.append(natural_card)
                        else:
                            for card in suited_cards:
                                if card.enhancement == CardEnhancement.WILD and not any(id(card) == id(c) for c in straight_flush_cards):
                                    straight_flush_cards.append(card)
                                    break
                    
                    if len(straight_flush_cards) == 5:
                        patterns[HandType.STRAIGHT_FLUSH].append(
                            HandPattern(HandType.STRAIGHT_FLUSH, straight_flush_cards)
                        )
    
    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[HandType, Dict[str, bool], List[Card]]:
        """
        Analyze a hand to determine the best hand type, all contained hand types,
        and the cards that make up the best hand.
        """
        best_type, patterns = HandEvaluator.analyze_hand(cards)
        
        best_pattern = HandEvaluator.get_best_pattern(patterns)
        
        scoring_cards = best_pattern.cards if best_pattern else []
        
        contained_types = HandEvaluator.get_contained_hand_types(patterns)
        
        return (best_type, contained_types, scoring_cards)

    @staticmethod
    def mark_scoring_cards(cards: List[Card], scoring_cards: List[Card]):
        """
        Mark cards that are part of the scoring hand
        """
        for card in cards:
            card.scored = False
            
        scoring_card_ids = set(id(card) for card in scoring_cards)
        
        for card in cards:
            if id(card) in scoring_card_ids:
                card.scored = True
    
    @staticmethod
    def get_contained_hand_types(patterns: Dict[HandType, List[HandPattern]]) -> Dict[str, bool]:
        """
        Convert patterns dictionary to a simpler format of contained hand types
        """
        contained_hands = {
            "high_card": bool(patterns[HandType.HIGH_CARD]),
            "pair": bool(patterns[HandType.PAIR]),
            "two_pair": bool(patterns[HandType.TWO_PAIR]),
            "three_of_kind": bool(patterns[HandType.THREE_OF_A_KIND]),
            "straight": bool(patterns[HandType.STRAIGHT]),
            "flush": bool(patterns[HandType.FLUSH]),
            "full_house": bool(patterns[HandType.FULL_HOUSE]),
            "four_of_kind": bool(patterns[HandType.FOUR_OF_A_KIND]),
            "straight_flush": bool(patterns[HandType.STRAIGHT_FLUSH])
        }
        
        return contained_hands
    
    @staticmethod
    def get_best_pattern(patterns: Dict[HandType, List[HandPattern]]) -> Optional[HandPattern]:
        """
        Get the best hand pattern from all identified patterns
        """
        for hand_type in [HandType.STRAIGHT_FLUSH, HandType.FOUR_OF_A_KIND, 
                         HandType.FULL_HOUSE, HandType.FLUSH, HandType.STRAIGHT, 
                         HandType.THREE_OF_A_KIND, HandType.TWO_PAIR, HandType.PAIR, HandType.HIGH_CARD]:
            if patterns[hand_type]:
                return patterns[hand_type][0]
                
        return None