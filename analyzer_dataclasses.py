from dataclasses import dataclass
from typing import Iterable

import numpy as np
from enums import Hero, Team, Action, Ability


@dataclass
class TemplateData:
    template: np.ndarray
    width: int
    height: int
    mask: np.ndarray = None


@dataclass
class KillFeedHero:
    hero: Hero
    point: tuple  # Tuple of two elements x = point[0] and y = point[1]
    confidence: float
    team: Team = Team.UNKNOWN
    template_data: TemplateData = None

    def __post_init__(self):
        if len(self.point) != 2:
            raise ValueError("KilLFeedItem point has to be tuple of length 2 (x, y)")

    def __eq__(self, other):
        return isinstance(other, KillFeedHero) and self.hero == other.hero and self.team == other.team

    def __str__(self):
        return f"{self.team.value} {self.hero.value}"


@dataclass
class KillFeedAbility:
    ability: Ability
    point: tuple  # Tuple of two elements x = point[0] and y = point[1]
    confidence: float
    template_data: TemplateData = None

    def __post_init__(self):
        if len(self.point) != 2:
            raise ValueError("KilLFeedItem point has to be tuple of length 2 (x, y)")

    def __eq__(self, other):
        return isinstance(other, KillFeedAbility) and self.ability == other.ability

    def __str__(self):
        return f"{self.ability.value} {self.confidence}"


@dataclass
class KillFeedLine:
    timestamp: float
    hero_left: KillFeedHero
    hero_right: KillFeedHero
    action: Action = Action.KILLED
    critical_hit: bool = False
    ability: Ability = None
    reaction_sent: bool = False

    def __post_init__(self):
        # TODO: use cv2 to detect action
        # Can't be sure about other team if the left hero is Mercy because it can be ress or kill.
        if self.hero_left.hero != Hero.MERCY:
            if self.hero_left.team == Team.UNKNOWN:
                if self.hero_right.team == Team.RED:
                    self.hero_left.team = Team.BLUE
                elif self.hero_right.team == Team.BLUE:
                    self.hero_left.team = Team.RED
            if self.hero_right.team == Team.UNKNOWN:
                if self.hero_left.team == Team.RED:
                    self.hero_right.team = Team.BLUE
                elif self.hero_left.team == Team.BLUE:
                    self.hero_right.team = Team.RED

    def add_abilities(self, abilities: Iterable[KillFeedAbility]):
        best_match = 0
        for a in abilities:
            if a.ability == Ability.CRITICAL_HIT:
                self.critical_hit = True
            if (not self.ability or a.confidence > best_match) and a.ability != Ability.CRITICAL_HIT:
                self.ability = a.ability

    def __hash__(self):
        return hash(f"{self.hero_left}{self.hero_right}{self.action}")

    def __eq__(self, other):
        return isinstance(other, KillFeedLine) and self.hero_right == other.hero_right and \
               self.hero_left == other.hero_left and self.action == other.action

    def __str__(self):
        ability = self.ability if self.ability else ''
        return f"[{self.timestamp}] {self.hero_left} {self.action.value} <{ability}>" \
               f"{'CRITICAL HIT!' if self.critical_hit else ''} {self.hero_right}"
