from collections import defaultdict
from enum import Enum
from typing import List, Dict


class Action(Enum):
    KILLED = "killed"
    RESSED = "resurrected"


class Team(Enum):
    RED = "red"
    BLUE = "blue"
    UNKNOWN = "unknown"


class Hero(Enum):
    ANA = "Ana"
    ASHE = "Ashe"
    BAPTISTE = "Baptiste"
    BASTION = "Bastion"
    BRIGITTE = "Brigitte"
    DOOMFIST = "Doomfist"
    DVA = "Dva"
    #DVA_MECH = "Dva_mech"
    ECHO = "Echo"
    GENJI = "Genji"
    HANZO = "Hanzo"
    JUNKERQUEEN = "Junkerqueen"
    JUNKRAT = "Junkrat"
    LUCIO = "Lucio"
    KIRIKO = "Kiriko"
    MCREE = "Mcree"
    MEI = "Mei"
    MERCY = "Mercy"
    MOIRA = "Moira"
    ORISA = "Orisa"
    PHARAH = "Pharah"
    REAPER = "Reaper"
    REINHARDT = "Reinhardt"
    ROADHOG = "Roadhog"
    SIGMA = "Sigma"
    SOLDIER = "Soldier"
    SOMBRA = "Sombra"
    SOJOURN = "Sojourn"
    SYMMETRA = "Symmetra"
    TORBJORN = "Torbjorn"
    TRACER = "Tracer"
    WIDOWMAKER = "Widowmaker"
    WINSTON = "Winston"
    WRECKINGBALL = "Wreckingball"
    ZARYA = "Zarya"
    ZENYATTA = "Zenyatta"


class Ability(Enum):

    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, abilityname, hero):
        self.ability = abilityname
        self.hero = hero

    CRITICAL_HIT = "Critical_hit", None
    MELEE = "Melee", None
    BOB = "Bob", Hero.ASHE
    DYNAMITE = "Dynamite", Hero.ASHE
    HOOK = "Hook", Hero.ROADHOG
    TRAP = "Trap", Hero.JUNKRAT
    CONCUSSION_MINE = "Concussion_mine", Hero.JUNKRAT
    FIRE_STRIKE = "Fire_strike", Hero.REINHARDT
    CHARGE = "Charge", Hero.REINHARDT
    RAILGUN = "Railgun", Hero.SOJOURN
    OVERCLOCK = "Overclock", Hero.SOJOURN
    WHIP_SHOT = "Whip_shot", Hero.BRIGITTE
    JAVELIN = "Javelin", Hero.ORISA
    SOUNDWAVE = "Soundwave", Hero.LUCIO
    PULSE_BOMB = "Pulse_bomb", Hero.TRACER
    GRENADE = "Grenade", Hero.BASTION
    ARTILLERY = "Artillery", Hero.BASTION
    TURRET = "Turret", Hero.TORBJORN
    MOLTEN_CORE = "Molten_core", Hero.TORBJORN
    PRIMAL_RAGE = "Primal_rage", Hero.WINSTON
    JUMP_PACK = "Jump_pack", Hero.WINSTON
    STORM_ARROW = "Storm_arrow", Hero.HANZO
    CALL_MECH = "Call_mech", Hero.DVA
    BOOSTERS = "Boosters", Hero.DVA
    MICRO_MISSILES = "Micro_missiles", Hero.DVA
    SELF_DESTRUCT = "Self-destruct", Hero.DVA


HERO_ABILITY_MAP: Dict[Hero, List[Ability]] = dict()
for h in Hero:
    HERO_ABILITY_MAP[h] = list()
    for a in Ability:
        # Echo can use any ability during ult
        if h == Hero.ECHO:
            HERO_ABILITY_MAP[h].append(a)
        # None is a special case which means it is a common ability for everyone
        if a.hero == h or a.hero is None:
            HERO_ABILITY_MAP[h].append(a)
