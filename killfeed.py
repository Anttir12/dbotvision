from dataclasses import dataclass

from enums import Hero, Team, Action


@dataclass
class KillFeedItem:
    hero: Hero
    point: tuple  # Tuple of two elements x = point[0] and y = point[1]
    confidence: float
    team: Team = Team.UNKNOWN

    def __post_init__(self):
        if len(self.point) != 2:
            raise ValueError("KilLFeedItem point has to be tuple of length 2 (x, y)")

    def __eq__(self, other):
        return isinstance(other, KillFeedItem) and self.hero == other.hero and self.team == other.team

    def __str__(self):
        return f"{self.team.value} {self.hero.value}"


@dataclass
class KillFeedLine:
    timestamp: float
    hero_left: KillFeedItem
    hero_right: KillFeedItem
    action: Action = Action.KILLED

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

    def __hash__(self):
        return hash(f"{self.hero_left}{self.hero_right}{self.action}")

    def __eq__(self, other):
        return isinstance(other, KillFeedLine) and self.hero_right == other.hero_right and \
               self.hero_left == other.hero_left and self.action == other.action

    def __str__(self):
        return f"[{self.timestamp}] {self.hero_left} {self.action.value} {self.hero_right}"
