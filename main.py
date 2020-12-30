import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from multiprocessing.pool import ThreadPool
from time import time
from typing import NamedTuple, Optional

from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
from mss import mss
from matplotlib import pyplot as plt


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
    DVA_MECH = "Dva_mech"
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


h_icons = {hero: cv2.cvtColor(cv2.imread("ow_icons/{}.png".format(hero.value.lower())), cv2.COLOR_BGR2GRAY) for hero in
           Hero}


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
        return f"{self.hero_left} {self.action.value} {self.hero_right}"


def analyze_image(img_rgb, thread_count):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    threshold = 0.80
    heroes_found = list()
    chunks = chunkify_list(list(h_icons.items()), thread_count)
    tpool = ThreadPool(processes=thread_count)
    threads = list()
    for chunk in chunks:
        img_copy = img_rgb.copy()
        res = tpool.apply_async(find_heroes, (chunk, img_copy, img_gray, threshold))
        threads.append(res)
    for t in threads:
        heroes_found.extend(t.get())
    heroes_found.sort(reverse=True, key=lambda kf_item: kf_item.point[1])
    return heroes_found


def find_heroes(hero_icons, img_rgb, img_gray, threshold):
    heroes_found = dict()
    width = 66
    height = 46
    for hero, template in hero_icons:
        mask_num = 1
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

        loc = np.where(res >= threshold)
        mask = np.zeros(img_gray.shape[:2], np.uint8)
        for pt in zip(*loc[::-1]):
            confidence = res[pt[1], pt[0]]
            midx = pt[0] + int(round(width / 2))
            midy = pt[1] + int(round(height / 2))
            mask_point = mask[midy, midx]
            # New match
            if mask_point == 0:
                mask[pt[1]:pt[1] + height, pt[0]:pt[0] + width] = mask_num
                heroes_found[mask_num] = KillFeedItem(hero=hero, point=pt, confidence=confidence)
                mask_num += 1
            # Not new but better match
            elif heroes_found[mask_point].confidence < confidence:
                heroes_found[mask_point].confidence = confidence
                heroes_found[mask_point].point = pt
    for hero_found in heroes_found.values():
        hero_found.team = get_team_color(hero_found, width, height, img_rgb)
    return heroes_found.values()


def get_team_color(kf_item: KillFeedItem, width: int, height: int, img_rbg):
    cropped = img_rbg[kf_item.point[1]-2: kf_item.point[1]+height+2, kf_item.point[0]-2:kf_item.point[0]+width+2]
    cropped[2:-2, 2:-2] = 0
    w, h = cropped.shape[:2]
    count = w*4+(h-4)*4
    bgr = np.array([0, 0, 0, 0])
    for row in cropped:
        for i in row:
            if i.any():
                bgr += i
    average_color = bgr / count
    if average_color[1] < average_color[0] > average_color[2]:
        return Team.BLUE
    if average_color[1] < average_color[2] > average_color[0]:
        return Team.RED
    return Team.UNKNOWN


def chunkify_list(some_list, chunk_count):
    for i in range(0, len(some_list), chunk_count):
        yield some_list[i:i + chunk_count]


def main():
    with mss() as sct:
        showim = True
        mon = sct.monitors[1]
        w = 640
        h = 448
        monitor = {
            "top": mon["top"] + 45,
            "left": mon["left"] + 2520-w,
            "width": w,
            "height": h,
            "mon": 1,
        }
        color = (0, 255, 0)
        width = 66
        height = 46
        killfeed = deque()
        while True:
            start = time()
            img = np.array(sct.grab(monitor))
            #img = cv2.imread("test_images/test1.png")
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            heroes_found = analyze_image(img, 4)
            timestamp = time()
            prev = None
            for i, kf_item in enumerate(heroes_found):
                if showim:
                    pass
                    cv2.rectangle(img, (kf_item.point[0], kf_item.point[1]), (kf_item.point[0]+width, kf_item.point[1]+height), color, 2)
                if not prev:
                    prev = kf_item
                    continue
                if kf_item.point[1] - 10 < prev.point[1] < kf_item.point[1] + 10:
                    if kf_item.point[0] < prev.point[0]:
                        kfl = KillFeedLine(hero_left=kf_item, hero_right=prev, timestamp=timestamp)
                    else:
                        kfl = KillFeedLine(hero_left=prev, hero_right=kf_item, timestamp=timestamp)
                    if kfl not in killfeed:
                        killfeed.append(kfl)
                        print(kfl)
                    prev = None

            while killfeed and timestamp - killfeed[0].timestamp > 12:
                killfeed.popleft()
            if showim:
                cv2.imshow("Matched image", img)
                cv2.waitKey(20)
            #print(f"analyze took {time()-start} seconds")


main()
