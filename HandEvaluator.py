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
        
        # Find basic patterns
        HandEvaluator._find_high_cards(cards, patterns)
        HandEvaluator._find_pairs(cards, patterns)
        HandEvaluator._find_three_of_a_kinds(cards, patterns)
        HandEvaluator._find_four_of_a_kinds(cards, patterns)
        HandEvaluator._find_straights(cards, patterns)
        HandEvaluator._find_flushes(cards, patterns)
        
        # Find complex patterns that build on simpler ones
        HandEvaluator._find_two_pairs(patterns)
        HandEvaluator._find_full_houses(patterns)
        HandEvaluator._find_straight_flushes(cards, patterns)
        
        # Determine best hand type
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
        # Sort by rank value (highest first)
        sorted_cards = sorted(cards, key=lambda card: card.rank.value, reverse=True)
        
        # Skip STONE cards as they have no rank for high card
        non_stone_cards = [card for card in sorted_cards if card.enhancement != CardEnhancement.STONE]
        
        # Take the highest card as the high card pattern
        if non_stone_cards:
            patterns[HandType.HIGH_CARD].append(HandPattern(HandType.HIGH_CARD, [non_stone_cards[0]]))
        elif sorted_cards:  # If we only have STONE cards, use one anyway
            patterns[HandType.HIGH_CARD].append(HandPattern(HandType.HIGH_CARD, [sorted_cards[0]]))
    
    @staticmethod
    def _find_pairs(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find all pairs in the hand, handling STONE cards and WILD cards"""
        # Group cards by rank, excluding STONE cards which have no rank
        rank_groups = defaultdict(list)
        for card in cards:
            if card.enhancement != CardEnhancement.STONE:
                rank_groups[card.rank.value].append(card)
            
        # Find pairs
        for rank, group in rank_groups.items():
            if len(group) >= 2:
                # Create a pattern for each possible pair combination
                for i in range(len(group) - 1):
                    for j in range(i + 1, len(group)):
                        pair_cards = [group[i], group[j]]
                        patterns[HandType.PAIR].append(HandPattern(HandType.PAIR, pair_cards))
    
    @staticmethod
    def _find_three_of_a_kinds(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find all three of a kinds in the hand, handling STONE and WILD cards"""
        # Group cards by rank, excluding STONE cards which have no rank
        rank_groups = defaultdict(list)
        for card in cards:
            if card.enhancement != CardEnhancement.STONE:
                rank_groups[card.rank.value].append(card)
            
        # Find three of a kinds
        for rank, group in rank_groups.items():
            if len(group) >= 3:
                # Create a pattern for each possible three of a kind combination
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
        # Group cards by rank, excluding STONE cards which have no rank
        rank_groups = defaultdict(list)
        for card in cards:
            if card.enhancement != CardEnhancement.STONE:
                rank_groups[card.rank.value].append(card)
            
        # Find four of a kinds
        for rank, group in rank_groups.items():
            if len(group) >= 4:
                four_cards = group[:4]  # Take the first four cards
                patterns[HandType.FOUR_OF_A_KIND].append(
                    HandPattern(HandType.FOUR_OF_A_KIND, four_cards)
                )
    
    @staticmethod
    def _find_straights(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find all straights in the hand, handling STONE and WILD cards"""
        # Skip STONE cards as they have no rank for straights
        non_stone_cards = [card for card in cards if card.enhancement != CardEnhancement.STONE]
        
        # Get unique rank values
        unique_ranks = sorted(set(card.rank.value for card in non_stone_cards))
        
        # Find 5-card straights
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i+4] - unique_ranks[i] == 4:
                # This is a straight
                straight_ranks = set(range(unique_ranks[i], unique_ranks[i] + 5))
                straight_cards = [card for card in non_stone_cards if card.rank.value in straight_ranks]
                
                # Need to select exactly one card of each rank
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
        
        # Check for A-2-3-4-5 straight (Ace = 1)
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
        # Skip STONE cards as they have no suit for flushes
        non_stone_cards = [card for card in cards if card.enhancement != CardEnhancement.STONE]
        
        # Group cards by suit, with special handling for WILD cards
        suit_groups = defaultdict(list)
        wild_cards = []
        
        for card in non_stone_cards:
            if card.enhancement == CardEnhancement.WILD:
                wild_cards.append(card)
            else:
                suit_groups[card.suit].append(card)
        
        # Add wild cards to all suit groups
        for suit in suit_groups:
            suit_groups[suit].extend(wild_cards)
        
        # Find flushes
        for suit, group in suit_groups.items():
            if len(group) >= 5:
                # Sort by rank (highest first) for best flush
                sorted_group = sorted(group, key=lambda card: card.rank.value, reverse=True)
                flush_cards = sorted_group[:5]  # Take the five highest cards
                
                patterns[HandType.FLUSH].append(
                    HandPattern(HandType.FLUSH, flush_cards)
                )
    
    @staticmethod
    def _find_two_pairs(patterns: Dict[HandType, List[HandPattern]]):
        """Find all two pair combinations using already identified pairs"""
        pairs = patterns[HandType.PAIR]
        
        if len(pairs) >= 2:
            # Check all combinations of pairs
            for i in range(len(pairs) - 1):
                for j in range(i + 1, len(pairs)):
                    pair1 = pairs[i]
                    pair2 = pairs[j]
                    
                    # Check if the pairs don't share any cards
                    pair1_ids = {id(card) for card in pair1.cards}
                    pair2_ids = {id(card) for card in pair2.cards}
                    
                    if not pair1_ids.intersection(pair2_ids):
                        # Check that the ranks are different
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
            # Get the rank of the three of a kind, ignoring WILD cards
            three_rank = None
            for card in three.cards:
                if card.enhancement != CardEnhancement.WILD:
                    three_rank = card.rank.value
                    break
            
            if three_rank is None:  # All wild cards - can be any rank
                three_rank = -1
            
            for pair in pairs:
                # Get the rank of the pair, ignoring WILD cards
                pair_rank = None
                for card in pair.cards:
                    if card.enhancement != CardEnhancement.WILD:
                        pair_rank = card.rank.value
                        break
                
                if pair_rank is None:  # All wild cards - can be any rank
                    pair_rank = -2
                
                # Check for shared cards between the three of a kind and pair
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
        # Skip STONE cards as they have no rank or suit
        non_stone_cards = [card for card in cards if card.enhancement != CardEnhancement.STONE]
        
        # Group cards by suit, with special handling for WILD cards
        suit_groups = defaultdict(list)
        wild_cards = []
        
        for card in non_stone_cards:
            if card.enhancement == CardEnhancement.WILD:
                wild_cards.append(card)
            else:
                suit_groups[card.suit].append(card)
        
        # Add wild cards to all suit groups
        for suit in suit_groups:
            suit_groups[suit].extend(wild_cards)
        
        # Check each suit group for straights
        for suit, suited_cards in suit_groups.items():
            if len(suited_cards) >= 5:
                rank_values = sorted(set(card.rank.value for card in suited_cards))
                
                # Find 5-card sequences
                for i in range(len(rank_values) - 4):
                    if rank_values[i+4] - rank_values[i] == 4:
                        straight_ranks = set(range(rank_values[i], rank_values[i] + 5))
                        straight_flush_cards = []
                        
                        for rank in straight_ranks:
                            # First try to find a natural card of this rank and suit
                            natural_card = None
                            for card in suited_cards:
                                if card.rank.value == rank and card.enhancement != CardEnhancement.WILD:
                                    natural_card = card
                                    break
                            
                            if natural_card:
                                straight_flush_cards.append(natural_card)
                            else:
                                # If no natural card, use a wild card
                                for card in suited_cards:
                                    if card.enhancement == CardEnhancement.WILD and not any(id(card) == id(c) for c in straight_flush_cards):
                                        straight_flush_cards.append(card)
                                        break
                        
                        if len(straight_flush_cards) == 5:
                            patterns[HandType.STRAIGHT_FLUSH].append(
                                HandPattern(HandType.STRAIGHT_FLUSH, straight_flush_cards)
                            )
                
                # Check for A-2-3-4-5 straight flush
                if set([14, 2, 3, 4, 5]).issubset(set(rank_values)):
                    straight_flush_cards = []
                    for rank in [14, 2, 3, 4, 5]:
                        # First try to find a natural card of this rank and suit
                        natural_card = None
                        for card in suited_cards:
                            if card.rank.value == rank and card.enhancement != CardEnhancement.WILD:
                                natural_card = card
                                break
                        
                        if natural_card:
                            straight_flush_cards.append(natural_card)
                        else:
                            # If no natural card, use a wild card
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
        
        Args:
            cards: List of cards to evaluate
            
        Returns:
            Tuple containing:
            - The best hand type
            - Dictionary of all contained hand types
            - List of cards that make up the best hand
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
        
        Args:
            cards: All cards being played
            scoring_cards: Cards that make up the best hand
        """
        # Reset scoring flag for all cards
        for card in cards:
            card.scored = False
            
        # Get identifying information for scoring cards (accounting for WILD cards)
        scoring_card_ids = set(id(card) for card in scoring_cards)
        
        # Mark cards that match
        for card in cards:
            if id(card) in scoring_card_ids:
                card.scored = True
    
    @staticmethod
    def get_contained_hand_types(patterns: Dict[HandType, List[HandPattern]]) -> Dict[str, bool]:
        """
        Convert patterns dictionary to a simpler format of contained hand types
        
        Args:
            patterns: Dictionary mapping hand types to lists of patterns
            
        Returns:
            Dictionary mapping hand type names to boolean indicating presence
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
        
        Args:
            patterns: Dictionary mapping hand types to lists of patterns
            
        Returns:
            The best hand pattern or None if no patterns exist
        """
        for hand_type in [HandType.STRAIGHT_FLUSH, HandType.FOUR_OF_A_KIND, 
                         HandType.FULL_HOUSE, HandType.FLUSH, HandType.STRAIGHT, 
                         HandType.THREE_OF_A_KIND, HandType.TWO_PAIR, HandType.PAIR, HandType.HIGH_CARD]:
            if patterns[hand_type]:
                return patterns[hand_type][0]
                
        return None