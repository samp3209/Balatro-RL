from typing import List, Dict, Tuple, Set
from collections import defaultdict
from Enums import *
from Card import Card

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
    and tracks which cards are part of each pattern
    """
    
    @staticmethod
    def analyze_hand(cards: List[Card]) -> Tuple[HandType, Dict[HandType, List[HandPattern]]]:
        """
        Analyze a hand to find all patterns within it
        
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
                         HandType.FULL_HOUSE, HandType.STRAIGHT, 
                         HandType.THREE_OF_A_KIND, HandType.TWO_PAIR, HandType.PAIR]:
            if patterns[hand_type]:
                best_type = hand_type
                break
                
        return (best_type, patterns)
    
    @staticmethod
    def _find_high_cards(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find high cards in the hand"""
        # Sort by rank value (highest first)
        sorted_cards = sorted(cards, key=lambda card: card.rank.value, reverse=True)
        
        # Take the highest card as the high card pattern
        if sorted_cards:
            patterns[HandType.HIGH_CARD].append(HandPattern(HandType.HIGH_CARD, [sorted_cards[0]]))
    
    @staticmethod
    def _find_pairs(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find all pairs in the hand"""
        # Group cards by rank
        rank_groups = defaultdict(list)
        for card in cards:
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
        """Find all three of a kinds in the hand"""
        # Group cards by rank
        rank_groups = defaultdict(list)
        for card in cards:
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
        """Find all four of a kinds in the hand"""
        # Group cards by rank
        rank_groups = defaultdict(list)
        for card in cards:
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
        """Find all straights in the hand"""
        # Get unique rank values
        unique_ranks = sorted(set(card.rank.value for card in cards))
        
        # Find 5-card straights
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i+4] - unique_ranks[i] == 4:
                # This is a straight
                straight_ranks = set(range(unique_ranks[i], unique_ranks[i] + 5))
                straight_cards = [card for card in cards if card.rank.value in straight_ranks]
                
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
                for card in cards:
                    if card.rank.value == rank:
                        straight_cards.append(card)
                        break
                        
            if len(straight_cards) == 5:
                patterns[HandType.STRAIGHT].append(
                    HandPattern(HandType.STRAIGHT, straight_cards)
                )
    
    @staticmethod
    def _find_flushes(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find all flushes in the hand"""
        # Group cards by suit
        suit_groups = defaultdict(list)
        for card in cards:
            suit_groups[card.suit].append(card)
            
        # Add wild cards to all suits
        wild_cards = [card for card in cards if card.enhancement == CardEnhancement.WILD]
        for suit in Suit:
            if suit != Suit.WILD:
                suit_groups[suit].extend(wild_cards)
                
        # Find flushes (5 or more cards of the same suit)
        for suit, group in suit_groups.items():
            if len(group) >= 5:
                # Create a flush with the 5 highest cards
                sorted_group = sorted(group, key=lambda card: card.rank.value, reverse=True)
                flush_cards = sorted_group[:5]
                
                patterns[HandType.STRAIGHT].append(  # Flushes are stored as straights
                    HandPattern(HandType.STRAIGHT, flush_cards)
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
                    pair1_ranks = {card.rank.value for card in pair1.cards}
                    pair2_ranks = {card.rank.value for card in pair2.cards}
                    
                    if not pair1_ranks.intersection(pair2_ranks):
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
            three_rank = three.cards[0].rank.value
            
            for pair in pairs:
                pair_rank = pair.cards[0].rank.value
                
                if three_rank != pair_rank:
                    full_house_cards = three.cards + pair.cards
                    patterns[HandType.FULL_HOUSE].append(
                        HandPattern(HandType.FULL_HOUSE, full_house_cards)
                    )
    
    @staticmethod
    def _find_straight_flushes(cards: List[Card], patterns: Dict[HandType, List[HandPattern]]):
        """Find straight flushes by looking for straights that are also flushes"""
        # Get all straights
        straights = patterns[HandType.STRAIGHT]
        
        for straight in straights:
            # Check if all cards in the straight have the same suit
            suits = {card.suit for card in straight.cards}
            
            # Allow wild cards to contribute to any suit
            non_wild_suits = {card.suit for card in straight.cards 
                             if card.enhancement != CardEnhancement.WILD}
            
            if len(non_wild_suits) <= 1:  # All cards have the same suit or are wild
                patterns[HandType.STRAIGHT_FLUSH].append(
                    HandPattern(HandType.STRAIGHT_FLUSH, straight.cards)
                )
    
    @staticmethod
    def mark_scoring_cards(cards: List[Card], best_pattern: HandPattern):
        """
        Mark cards that are part of the scoring hand
        
        Args:
            cards: All cards being played
            best_pattern: The pattern that makes up the best hand
        """
        # Reset scoring flag for all cards
        for card in cards:
            card.scored = False
            
        # Get identifying information for cards in the best pattern
        scoring_card_ids = {(card.rank.value, card.suit) for card in best_pattern.cards}
        
        # Mark cards that match
        for card in cards:
            if (card.rank.value, card.suit) in scoring_card_ids:
                card.scored = True
                # Remove this ID to avoid marking duplicates
                scoring_card_ids.remove((card.rank.value, card.suit))
    
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
            "full_house": bool(patterns[HandType.FULL_HOUSE]),
            "four_of_kind": bool(patterns[HandType.FOUR_OF_A_KIND]),
            "straight_flush": bool(patterns[HandType.STRAIGHT_FLUSH])
        }
        
        # Add "flush" specifically for jokers that care about flushes
        # In this implementation, flushes are stored as straights
        contained_hands["flush"] = any(
            len(set(card.suit for card in pattern.cards if card.enhancement != CardEnhancement.WILD)) == 1
            for pattern in patterns[HandType.STRAIGHT]
        )
        
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
                         HandType.FULL_HOUSE, HandType.STRAIGHT, 
                         HandType.THREE_OF_A_KIND, HandType.TWO_PAIR, HandType.PAIR, HandType.HIGH_CARD]:
            if patterns[hand_type]:
                # For simplicity, just return the first pattern of the best type
                # A more sophisticated implementation would rank patterns of the same type
                return patterns[hand_type][0]
                
        return None