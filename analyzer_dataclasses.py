from dataclasses import dataclass
from typing import Iterable, Any

import cv2
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
        return self.hero == other.hero and self.team == other.team

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
