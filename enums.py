from enum import Enum


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
    JUNKRAT = "Junkrat"
    LUCIO = "Lucio"
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
    SYMMETRA = "Symmetra"
    TORBJORN = "Torbjorn"
    TRACER = "Tracer"
    WIDOWMAKER = "Widowmaker"
    WINSTON = "Winston"
    WRECKINGBALL = "Wreckingball"
    ZARYA = "Zarya"
    ZENYATTA = "Zenyatta"
