from typing import List, Dict, Tuple, Optional
from Enums import *
from Card import Card
from Inventory import Inventory
from Game import Game
from HandEvaluator import HandEvaluator
from collections import defaultdict

class GameManager:
    """
    High-level manager for the game that coordinates game flow and provides
    an interface for playing the game.
    """
    
    def __init__(self, seed=None):
        """Initialize the game manager with an optional seed for reproducibility"""
        self.game = Game(seed)
        self.hand_evaluator = HandEvaluator()
        
        self.current_hand = []
        self.played_cards = []
        self.discarded_cards = []
        self.hand_result = None
        self.contained_hand_types = {}
        self.scoring_cards = []
        
        self.max_hand_size = 8
        self.max_hands_per_round = 4
        self.max_discards_per_round = 4
        self.discards_used = 0
        self.hands_played = 0
        self.current_score = 0
        self.current_ante_beaten = False
        self.game_over = False
        
        self.boss_blind_message = ""

    def start_new_game(self):
        """Start a new game with a fresh deck and initial ante"""
        self.game.initialize_deck()
        self.deal_new_hand()
        self.game_over = False

        self.apply_boss_blind_effect()

    
    def apply_boss_blind_effect(self):
        """Apply boss blind effect if at a boss blind"""
        effect = self.game.set_boss_blind_effect()
        if effect:
            self.boss_blind_message = f"Boss Blind Effect: {effect.name}"
            print(f"\n⚠️ {self.boss_blind_message} ⚠️\n")
            
            if effect == BossBlindEffect.DISCARD_RANDOM:
                self.boss_blind_message += " - 2 random cards will be discarded per hand played"
            elif effect == BossBlindEffect.HALVE_VALUES:
                self.boss_blind_message += " - Base chips and mult are halved"
            elif effect == BossBlindEffect.CLUB_DEBUFF:
                self.boss_blind_message += " - All Club cards are debuffed (reduced value)"
            elif effect == BossBlindEffect.FACE_CARDS_DOWN:
                self.boss_blind_message += " - All face cards are face down"
            elif effect == BossBlindEffect.RANDOM_CARDS_DOWN:
                self.boss_blind_message += " - 1 in 7 cards are face down"
            elif effect == BossBlindEffect.FIRST_HAND_DOWN:
                self.boss_blind_message += " - First 8 cards dealt are face down"
            elif effect == BossBlindEffect.PREVIOUS_CARDS_DEBUFF:
                self.boss_blind_message += " - Cards played previously in this ante are debuffed"
            elif effect == BossBlindEffect.FORCE_CARD_SELECTION:
                self.boss_blind_message += " - One card must be selected every hand"
        else:
            self.boss_blind_message = ""


    def deal_new_hand(self):
        """Deal a new hand of cards from the deck"""
        self.current_hand = []
        self.played_cards = []
        
        self.current_hand = self.game.deal_hand(self.max_hand_size)
        
        if self.game.is_boss_blind:
            if self.game.active_boss_blind_effect == BossBlindEffect.FORCE_CARD_SELECTION:
                self.forced_card = self.game.get_forced_card(self.current_hand)
                if self.forced_card:
                    print(f"Boss Blind Effect: Card must be selected: {self.forced_card}")
        
        self.hand_result = None
        self.contained_hand_types = {}
        self.scoring_cards = []
        
    def play_cards(self, card_indices: List[int]) -> Tuple[bool, str]:
        """
        Play cards from the current hand
        """
        if not card_indices:
            return (False, "No cards selected to play")
            
        if len(self.played_cards) > 0:
            return (False, "Cards have already been played this hand")
            
        if self.hands_played >= self.max_hands_per_round:
            return (False, "Already played maximum number of hands for this round")
            
        if (self.game.is_boss_blind and 
            self.game.active_boss_blind_effect == BossBlindEffect.FORCE_CARD_SELECTION and 
            self.game.forced_card_index is not None):
            forced_idx = self.game.forced_card_index
            if forced_idx not in card_indices:
                return (False, f"Boss Blind Effect: You must include card at index {forced_idx} in your play")
            
        cards_to_play = []
        for idx in sorted(card_indices, reverse=True):
            if 0 <= idx < len(self.current_hand):
                cards_to_play.append(self.current_hand.pop(idx))
                
        if not cards_to_play:
            return (False, "No valid cards selected")
            
        self.played_cards.extend(cards_to_play)
        
        self.game.play_cards(cards_to_play)
        
        hand_type, contained_hands, scoring_cards = self.hand_evaluator.evaluate_hand(self.played_cards)
        
        self.hand_evaluator.mark_scoring_cards(self.played_cards, scoring_cards)
        
        self.game._mark_scoring_cards(self.played_cards, hand_type)
        
        retrigger_applied = False
        retrigger_msg = ""
        for joker in self.game.inventory.jokers:
            if joker.name == "Socks and Buskin":
                for card in self.played_cards:
                    if card.face and card.scored:
                        card.retrigger = True
                        retrigger_applied = True
        
        if retrigger_applied:
            retrigger_msg = " (with retrigger effect)"
        
        self.hand_result = hand_type
        self.contained_hand_types = contained_hands
        self.scoring_cards = scoring_cards
        
        round_info = {
            'hand_type': hand_type.name.lower(),
            'contained_hands': contained_hands,
            'hands_played': self.game.hands_played,
            'inventory': self.game.inventory,
            'max_discards': self.max_discards_per_round,
            'face_cards_discarded_count': self.game.face_cards_discarded_count
        }
        
        total_mult, chips, money_gained = self.game.apply_joker_effects(
            self.played_cards, hand_type, contained_hands
        )
        
        final_score = int(chips * total_mult)

        print("\n=== Joker Effects ===")
        for joker in self.game.inventory.jokers:
            effect = joker.calculate_effect(
                self.played_cards, 
                self.discards_used, 
                self.game.inventory.deck, 
                round_info
            )
            
            details = []
            if effect.mult_add > 0:
                details.append(f"+{effect.mult_add} mult")
            if effect.mult_mult > 1:
                details.append(f"x{effect.mult_mult} mult")
            if effect.chips > 0:
                details.append(f"+{effect.chips} chips")
            if effect.money > 0:
                details.append(f"+${effect.money} money")
                
            if details:
                print(f"{joker.name}: {', '.join(details)}")
            else:
                print(f"{joker.name}: No effect")
        
        self.current_score += final_score
        self.game.inventory.money += money_gained
        self.hands_played += 1
        self.game.hands_played += 1
        
        if self.game.is_boss_blind and self.game.active_boss_blind_effect == BossBlindEffect.PREVIOUS_CARDS_DEBUFF:
            self.game.track_scored_cards()
        
        if self.current_score >= self.game.current_blind:
            self.current_ante_beaten = True
            message = f"Played {hand_type.name} for {final_score} chips ({chips} x {total_mult})"
            return (True, message)
        
        if self.hands_played >= self.max_hands_per_round and not self.current_ante_beaten:
            mr_bones_index = None
            for i, joker in enumerate(self.game.inventory.jokers):
                if joker.name == "Mr. Bones":
                    mr_bones_index = i
                    break
                    
            if mr_bones_index is not None and self.current_score >= (self.game.current_blind * 0.25):
                #Mr. Bones saves the game but gets removed
                self.current_ante_beaten = True
                removed_joker = self.game.inventory.remove_joker(mr_bones_index)
                message = f"Played {hand_type.name} for {final_score} chips ({chips} x {total_mult}) - Mr. Bones saved you and vanished!"
            else:
                self.game_over = True
                message = f"Played {hand_type.name} for {final_score} chips ({chips} x {total_mult}) - GAME OVER: Failed to beat the ante"
            return (True, message)
        
        message = f"Played {hand_type.name} for {final_score} chips ({chips} x {total_mult})"
        self.deal_new_hand()
        if retrigger_applied:
            message += retrigger_msg
        return (True, message)

    def reset_hand_state(self):
        """Reset the hand state for a new hand within the same ante."""
        self.played_cards = []
                
        for card in self.current_hand:
            card.reset_state()
            self.game.inventory.add_card_to_deck(card)
            
        self.current_hand = []
        
        self.game.inventory.shuffle_deck()
        
        self.current_hand = self.game.deal_hand(self.max_hand_size)
        
        self.hand_result = None
        self.contained_hand_types = {}
        self.scoring_cards = []
    
    def discard_cards(self, card_indices: List[int]) -> Tuple[bool, str]:
        """
        Discard cards from the current hand and draw replacements
        """
        self.discarded_indices = card_indices.copy()

        if not card_indices:
            return (False, "No cards selected to discard")
            
        if self.discards_used >= self.max_discards_per_round:
            return (False, "Maximum discards already used for this round")
            
        if (self.game.is_boss_blind and 
            self.game.active_boss_blind_effect == BossBlindEffect.FORCE_CARD_SELECTION and 
            self.game.forced_card_index is not None):
            forced_idx = self.game.forced_card_index
            if forced_idx not in card_indices:
                return (False, f"Boss Blind Effect: You must include card at index {forced_idx} in your discard")
            
        cards_to_discard = []
        for idx in sorted(card_indices, reverse=True):
            if 0 <= idx < len(self.current_hand):
                cards_to_discard.append(self.current_hand.pop(idx))
                
        if not cards_to_discard:
            return (False, "No valid cards selected")
            
        self.discarded_cards.extend(cards_to_discard)
        
        self.game.discard_cards(cards_to_discard)
        
        replacement_cards = self.game.deal_hand(len(cards_to_discard))
        self.current_hand.extend(replacement_cards)
        
        self.discards_used += 1
        
        return (True, f"Discarded {len(cards_to_discard)} cards and drew replacements")
    
    def get_best_hand_from_current(self) -> Optional[Tuple[HandType, List[Card]]]:
        """
        Evaluate the current hand to determine the best possible hand
        """
        if not self.current_hand:
            return None
            
        hand_type, _, scoring_cards = self.hand_evaluator.evaluate_hand(self.current_hand)
        
        return (hand_type, scoring_cards)
    
    def get_recommended_play(self) -> Tuple[List[int], str]:
        """
        Get a recommended play based on the current hand
        """
        forced_card = None
        if (self.game.is_boss_blind and 
            self.game.active_boss_blind_effect == BossBlindEffect.FORCE_CARD_SELECTION and 
            self.game.forced_card_index is not None):
            forced_index = self.game.forced_card_index
            if 0 <= forced_index < len(self.current_hand):
                forced_card = self.current_hand[forced_index]
                print(f"Boss Blind FORCE_CARD_SELECTION: Card at index {forced_index} must be played/discarded")

        best_hand = self.get_best_hand_from_current()
        
        if not best_hand:
            return ([], "No cards to play")
            
        hand_type, cards = best_hand
        
        indices = []
        for card in cards:
            for i, hand_card in enumerate(self.current_hand):
                if (hand_card.rank.value == card.rank.value and 
                    hand_card.suit == card.suit and
                    i not in indices):
                    indices.append(i)
                    break
        
        if forced_card and self.game.forced_card_index not in indices:
            indices.append(self.game.forced_card_index)
                    
        explanation = f"Play {hand_type.name} for the best chance of winning"
        if forced_card:
            explanation += f" (including forced card at index {self.game.forced_card_index})"
        
        return (indices, explanation)
    
    def get_game_state(self) -> Dict:
        """
        Get the current game state as a dictionary
        """
        return {
            'current_ante': self.game.current_ante,
            'current_blind': self.game.current_blind,
            'hands_played': self.hands_played,
            'discards_used': self.discards_used,
            'max_discards': self.max_discards,
            'current_score': self.current_score,
            'ante_beaten': self.current_ante_beaten,
            'money': self.game.inventory.money,
            'hand_size': len(self.current_hand),
            'played_cards_count': len(self.played_cards),
            'discarded_cards_count': len(self.discarded_cards),
            'deck_size': len(self.game.inventory.deck),
            'joker_count': len(self.game.inventory.jokers),
            'consumable_count': len(self.game.inventory.consumables),
            'is_boss_blind': self.game.is_boss_blind,
            'boss_blind_effect': self.game.active_boss_blind_effect.name if self.game.active_boss_blind_effect else "None"
        }
        
    def next_ante(self):
        """
        Move to the next ante if the current one is beaten
        Returns True if successful
        """
        if not self.current_ante_beaten:
            print("Cannot advance ante - current ante not beaten")
            return False
        
        current_ante = self.game.current_ante
        print(f"next_ante(): Moving from Ante {current_ante} to {current_ante + 1}")
        
        hands_left = self.max_hands_per_round - self.hands_played
        
        blind_type = "Small"
        if self.game.current_ante % 3 == 2:
            blind_type = "Medium"
        elif self.game.current_ante % 3 == 0:
            blind_type = "Boss"
        
        base_money = 0
        if blind_type == "Small":
            base_money = 3
        elif blind_type == "Medium":
            base_money = 4
        elif blind_type == "Boss":
            base_money = 5
        
        money_earned = base_money + hands_left
        self.game.inventory.money += money_earned
        
        print(f"Earned ${money_earned} for beating the blind with {hands_left} hands left to play")
        
        self.game.current_ante += 1
        blind_progression = {
            # Ante 1
            1: 300,   
            2: 450,   
            3: 600,   # Boss blind
            # Ante 2
            4: 800,   
            5: 1200,  
            6: 1600,  # Boss blind
            # Ante 3
            7: 2000,
            8: 3000,
            9: 4000,  # Boss blind
            #Ante 4
            10: 5000, 
            11: 7500,
            12: 10000, # Boss
            #Ante 5
            13: 11000,
            14: 17500,
            15: 22000, # Boss
            #Ante 6
            16: 20000,
            17: 30000,
            18: 40000,
            # Ante 7
            19: 35000,
            20: 52500,
            21: 70000,
            #Ante 8
            22: 50000,
            23: 75000,
            24: 100000
        }
        
        self.game.current_blind = blind_progression.get(self.game.current_ante, 5000)
        self.current_score = 0
        self.current_ante_beaten = False
        self.reset_for_new_round()
        
        return True
    
    def use_tarot(self, tarot_index: int, selected_card_indices: List[int]) -> Tuple[bool, str]:
        """
        Use a tarot card with selected cards
        """
        tarot_indices = self.game.inventory.get_consumable_tarot_indices()
        
        if tarot_index not in tarot_indices:
            return (False, "Invalid tarot card selected")
            
        selected_cards = []
        for idx in selected_card_indices:
            if 0 <= idx < len(self.current_hand):
                selected_cards.append(self.current_hand[idx])
                
        game_state = {
            'money': self.game.inventory.money,
            'last_tarot_used': self.game.inventory.last_tarot,
            'last_planet_used': self.game.inventory.last_planet
        }
        
        effect = self.game.inventory.use_tarot(tarot_index, selected_cards, game_state)
        
        if not effect:
            return (False, "Failed to apply tarot effect")
            
        message = effect.message if hasattr(effect, 'message') else 'Tarot card used successfully'

        if hasattr(effect, 'money_gained') and effect.money_gained > 0:
            self.game.inventory.money += effect.money_gained
            message += f" Gained ${effect.money_gained}."
            

        
        return (True, message)
    
    def use_planet(self, planet_index: int) -> Tuple[bool, str]:
        """
        Use a planet card with the current hand type
        """
        planet_indices = self.game.inventory.get_consumable_planet_indices()
        
        if planet_index not in planet_indices:
            return (False, "Invalid planet card selected")
            
        if not self.hand_result:
            return (False, "No hand has been played yet")
            
        game_state = {
            'money': self.game.inventory.money,
            'stake_multiplier': self.game.stake_multiplier
        }
        
        effect = self.game.inventory.use_planet(planet_index, self.hand_result, game_state)
        
        if not effect:
            return (False, "Failed to apply planet effect")
            
        message = effect.get('message', 'Planet card used successfully')
        
        if 'mult_bonus' in effect:
            self.game.stake_multiplier += effect['mult_bonus']
            
        if 'chip_bonus' in effect:
            self.current_score += effect['chip_bonus']
            
        if self.current_score >= self.game.current_blind:
            self.current_ante_beaten = True
            message += f" Ante beaten! ({self.current_score}/{self.game.current_blind})"
            
        return (True, message)
    
    def reset_for_new_round(self):
        """Reset the game state for a new round (after playing max hands or beating the ante)"""
        self.discards_used = 0
        self.hands_played = 0
        self.current_score = 0
        self.current_ante_beaten = False
        
        #self._log_card_distribution("BEFORE RESET")
        
        self.game.inventory.reset_deck(
            played_cards=self.played_cards,
            discarded_cards=self.discarded_cards,
            hand_cards=self.current_hand
        )
        
        self.played_cards = []
        self.discarded_cards = []
        self.current_hand = []
        
        self._log_card_distribution("AFTER RESET")
        
        self.deal_new_hand()
        
        for joker in self.game.inventory.jokers:
            if hasattr(joker, 'reset'):
                joker.reset()
                
        self.apply_boss_blind_effect()
                
    def _log_card_distribution(self, prefix=""):
        """Log the distribution of cards in the deck, hand, played and discarded piles"""
        rank_counts = defaultdict(int)
        suit_counts = defaultdict(int)
        
        deck_size = len(self.game.inventory.deck)
        for card in self.game.inventory.deck:
            rank_counts[card.rank.name] += 1
            suit_counts[card.suit.name] += 1
            
        hand_size = len(self.current_hand)
        
        played_size = len(self.played_cards)
        
        discarded_size = len(self.discarded_cards)
        
        total_cards = deck_size + hand_size + played_size + discarded_size
        
        #print(f"\n{prefix} CARD DISTRIBUTION:")
        #print(f"Deck: {deck_size}, Hand: {hand_size}, Played: {played_size}, Discarded: {discarded_size}")
        print(f"Total cards: {total_cards} (should be 52 in base case)")
        #print(f"Ranks: {dict(rank_counts)}")
        #print(f"Suits: {dict(suit_counts)}")