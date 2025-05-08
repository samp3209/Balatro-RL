from typing import List, Tuple, Dict, Set, Optional
from Enums import *
from Card import Card
import random  

class EnhancementManager:
    """
    Manages the application and effects of card enhancements.
    This class serves as a central place to handle all enhancement-related logic.
    """
    
    @staticmethod
    def apply_enhancement_effects(cards: List[Card], 
                                 hand_type: HandType, 
                                 base_mult: int, 
                                 base_chips: int,
                                 is_boss_blind: bool = False,
                                 active_boss_blind_effect: Optional[BossBlindEffect] = None) -> Tuple[int, int]:
        """
        Apply enhancement effects to the score based on cards in hand
        """
        total_mult = base_mult
        total_chips = base_chips
        
        scoring_cards = [card for card in cards if card.scored]
        
        for card in scoring_cards:
            if card.debuffed:
                continue
                
            if (is_boss_blind and 
                active_boss_blind_effect == BossBlindEffect.CLUB_DEBUFF and 
                card.suit == Suit.CLUBS):
                continue
                
            total_chips += card.get_chip_bonus()
            
            total_mult += card.get_mult_add()
            
            total_mult *= card.get_mult_mult()
            
        glass_cards = [card for card in scoring_cards if card.enhancement == CardEnhancement.GLASS and not card.debuffed]
        for card in glass_cards:
            total_mult *= 2
            
            if random.random() < 0.25:
                card.in_deck = False
                print(f"Glass card {card} broke and was removed from the deck")
        
        steel_cards = [card for card in cards if card.enhancement == CardEnhancement.STEEL and card.in_hand and not card.debuffed]
        for card in steel_cards:
            total_mult *= 1.5
            
        stone_cards = [card for card in scoring_cards if card.enhancement == CardEnhancement.STONE and not card.debuffed]
        for card in stone_cards:
            total_chips += 50
            
        lucky_cards = [card for card in scoring_cards if card.enhancement == CardEnhancement.LUCKY and not card.debuffed]
        for card in lucky_cards:
            if random.random() < 0.2:
                total_mult += 20
                print(f"Lucky card {card} gave +20 mult!")
                
        if is_boss_blind and active_boss_blind_effect == BossBlindEffect.HALVE_VALUES:
            total_mult = max(1, total_mult // 2)
            total_chips = max(5, total_chips // 2)
            
        return (total_mult, total_chips)
    
    @staticmethod
    def process_enhancement_after_hand(cards: List[Card], inventory) -> Dict[str, int]:
        """
        Process enhancement effects that happen after a hand is played
        """
        result = {'money_gained': 0}
        
        gold_cards = [card for card in cards if card.enhancement == CardEnhancement.GOLD and card.in_hand]
        result['money_gained'] += len(gold_cards) * 3
        
        if gold_cards:
            print(f"Gained ${len(gold_cards) * 3} from {len(gold_cards)} gold cards held in hand")
        
        for card in cards:
            if card.enhancement == CardEnhancement.GLASS and card.played and random.random() < 0.25:
                card.in_deck = False
                print(f"Glass card {card} broke and was removed from the deck")
        
        return result
    
    @staticmethod
    def apply_wild_card_effects(cards: List[Card]) -> None:

        pass


def enhance_card(card: Card, enhancement: CardEnhancement) -> Card:
    """
    Apply an enhancement to a card and return the enhanced card
    
    """
    card.apply_enhancement(enhancement)
    return card

def count_enhanced_cards(cards: List[Card], enhancement: CardEnhancement) -> int:
    """
    Count cards with a specific enhancement
    """
    return sum(1 for card in cards if card.enhancement == enhancement)

def get_enhanced_cards(cards: List[Card], enhancement: CardEnhancement) -> List[Card]:
    """
    Get all cards with a specific enhancement
    """
    return [card for card in cards if card.enhancement == enhancement]
