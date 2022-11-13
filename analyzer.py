from collections import deque, Counter, defaultdict
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool, ApplyResult
from time import time, sleep
from typing import Dict, List, Iterable, Tuple, Optional, Set

import cv2
import logging
import numpy as np
from mss import mss

from dbot_api_client import DbotApiClient
from enums import Hero, Team, Action, Ability, HERO_ABILITY_MAP
from analyzer_dataclasses import KillFeedHero, TemplateData, KillFeedAbility


@dataclass
class KillFeedLine:
    threshold: float
    timestamp: float
    hero_left_bgr: np.ndarray = None
    hero_left_gray: np.ndarray = None
    hero_right_bgr: np.ndarray = None
    hero_right_gray: np.ndarray = None
    hero_right: KillFeedHero = None
    hero_left: KillFeedHero = None
    team_left: Team = None
    team_right: Team = None
    action: Action = Action.KILLED
    critical_hit: bool = False
    ability: Ability = None
    ability_bgr: np.ndarray = None
    reaction_sent: bool = False

    def analyze(self):
        self.determine_heroes()
        if self.hero_left:
            possible_abilities = [(a, a_icons[a]) for a in HERO_ABILITY_MAP[self.hero_left.hero]]
            self.determine_ability(possible_abilities)
        return self.hero_left and self.hero_right

    def determine_heroes(self):
        self.hero_left = self._find_hero(self.hero_left_gray)
        if self.hero_left:
            self.hero_left.team = self.team_left
            self.hero_right = self._find_hero(self.hero_right_gray)
            if self.hero_right:
                self.hero_right.team = self.team_right

    def _find_hero(self, img_gray) -> Optional[KillFeedHero]:
        for hero, template_data in h_icons.items():
            res = cv2.matchTemplate(img_gray, template_data.template, cv2.TM_CCOEFF_NORMED, mask=template_data.mask)
            loc = np.where((res >= self.threshold) & (res != float('inf')))
            for pt in zip(*loc[::-1]):
                confidence = res[pt[1], pt[0]]
                hero_found = KillFeedHero(hero=hero, point=pt, confidence=confidence, template_data=template_data)
                return hero_found

        return None

    def add_abilities(self, abilities: Iterable[KillFeedAbility]):
        best_match = 0
        for a in abilities:
            if a.ability == Ability.CRITICAL_HIT:
                self.critical_hit = True
            if (not self.ability or a.confidence > best_match) and a.ability != Ability.CRITICAL_HIT:
                self.ability = a.ability

    def determine_ability(self, possible_abilities: List[Tuple[Ability, TemplateData]],
                          threshold=0.70) -> Iterable[KillFeedAbility]:

        img_rgb = self.ability_bgr.copy()
        # Make the picture black & Red. Everything close to red is red. Rest is black. Increases crit detection
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(img_hsv, (0, 70, 140), (10, 255, 255))
        red_mask2 = cv2.inRange(img_hsv, (170, 70, 140), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        white_mask = cv2.inRange(img_hsv, (0, 0, 230), (255, 25, 255))
        img_rgb[red_mask > 0] = (0, 0, 255, 255)
        img_rgb[white_mask > 0] = (255, 255, 255, 255)
        img_rgb[(red_mask == 0) & (white_mask == 0)] = (0, 0, 0, 255)

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        # self.debug_image(img_rgb, "rgb", None)
        # self.debug_image(img_gray, "gray")
        abilities_found: Dict[int, KillFeedAbility] = dict()
        mask_num = 0
        for ability, template_data in possible_abilities:
            if template_data.width > img_gray.shape[1] or template_data.height > img_gray.shape[0]:
                logger.error(f"ERRROR!!! img shape: {img_gray.shape}  ability shape: {template_data.template.shape}")
                continue
            res = cv2.matchTemplate(img_gray, template_data.template, cv2.TM_CCOEFF_NORMED)
            # self.debug_image(img_crit_gray, "gray", None)
            # self.debug_image(template_data.template, "template", None)
            # self.debug_image(img_crit_rgb, "brg")
            # TODO All the below. I don't remember how it works
            loc = np.where((res >= threshold) & (res != float('inf')))
            red_mask = np.zeros(img_gray.shape[:2], np.uint8)
            for pt in zip(*loc[::-1]):
                confidence = res[pt[1], pt[0]]
                midx = pt[0] + int(round(template_data.width / 2))
                midy = pt[1] + int(round(template_data.height / 2))
                mask_point = red_mask[midy, midx]
                # New match
                if mask_point == 0:
                    red_mask[pt[1]:pt[1] + template_data.height, pt[0]:pt[0] + template_data.width] = mask_num
                    abilities_found[mask_num] = KillFeedAbility(ability=ability, point=pt, confidence=confidence,
                                                                template_data=template_data)
                    mask_num += 1
                # Not new but better match
                elif abilities_found[mask_point].confidence < confidence:
                    abilities_found[mask_point].confidence = confidence
                    abilities_found[mask_point].point = pt
        return abilities_found.values()

    def __hash__(self):
        return hash(f"{self.hero_left}{self.hero_right}{self.action}")

    def __eq__(self, other):
        return self.hero_right == other.hero_right and self.hero_left == other.hero_left and self.action == other.action

    def __str__(self):
        ability = self.ability if self.ability else ''
        return f"[{self.timestamp}] {self.hero_left} {self.action.value} <{ability}>" \
               f"{'CRITICAL HIT!' if self.critical_hit else ''} {self.hero_right}"


def load_hero_icons():
    icons = dict()
    for h in Hero:
        img = cv2.imread(f"icons_with_mask/{h.value.lower()}.png")
        hero_icon = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        icon_mask = cv2.cvtColor(cv2.imread(f"icons_with_mask/{h.value.lower()}_mask.png"), cv2.COLOR_BGR2GRAY)
        icons[h] = TemplateData(template=hero_icon, mask=icon_mask, width=img.shape[1], height=img.shape[0])
    return icons


def load_ability_icons():
    icons = dict()
    for a in Ability:
        img = cv2.imread(f"ability_icons/{a.ability.lower()}.png")
        ability_icon = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        icons[a] = TemplateData(template=ability_icon, width=img.shape[1], height=img.shape[0])
    return icons


h_icons: Dict[Hero, TemplateData] = load_hero_icons()
h_icons_list = list(h_icons.items())
a_icons: Dict[Ability, TemplateData] = load_ability_icons()


logger = logging.getLogger(__name__)


class KillFeedAnalyzer:

    kill_events = {
        2: "double_kill",
        3: "triple_kill",
        4: "quadruple_kill",
        5: "quintuple_kill",
        6: "sextuple_kill",
    }

    def _if_debug_print(self):
        return self.debug and self.i % 3 == 0

    def __init__(self, api_client, act_instantly=False, show_debug_img=False, debug=False, print_killfeed=True,
                 combo_cutoff=2, threshold=0.78, thread_count=7):
        self.color = (0, 255, 0)
        self.hero_width = 66
        self.hero_height = 46
        self.debug = debug
        self.show_debug_img = show_debug_img
        self.killfeed = deque()
        self.multikilldetection = Counter()
        self.short_event_history = set()
        self.killfeed_clear_time = 0
        self.combo_cutoff = combo_cutoff
        self.api_client: DbotApiClient = api_client
        self.act_instantly = act_instantly
        self.print_killfeed = print_killfeed
        self.threshold = threshold
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.disable_debug_images = False
        self.thread_count = thread_count
        self.tpool = ThreadPool(processes=thread_count)
        self.i = 0

    def start_analyzer(self):
        with mss() as sct:
            mon = sct.monitors[1]
            w = 550
            h = 500
            monitor = {
                "top": mon["top"] + 15,
                "left": mon["left"] + 3820 - w,
                "width": w,
                "height": h,
                "mon": 1,
            }
            while True:
                self.i += 1
                if self.debug:
                    start = time()
                filename = None
                #filename = "test_images/sextuple.png"
                if not filename:
                    img = np.array(sct.grab(monitor))
                else:
                    img = cv2.imread(filename)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                self.update_killfeed(img)
                self.analyze_killfeed()
                if self._if_debug_print():
                    logger.debug(f"Analyzing loop took {time()-start} seconds")

    def update_killfeed(self, img):
        start = time()
        killfeed_lines = self.analyze_image(img)
        if self._if_debug_print():
            logger.debug(f"Analyzing image took {time() - start} seconds")
        timestamp = time()
        for kfl in killfeed_lines:
            if kfl not in self.killfeed:
                self.killfeed.append(kfl)
                if kfl.action == Action.KILLED:
                    self.multikilldetection[str(kfl.hero_left)] += 1
                    self.update_multikillcutoff()
                if self.print_killfeed:
                    logger.info(kfl)

        while self.killfeed and timestamp - self.killfeed[0].timestamp > 10:
            self.killfeed.popleft()
        if self.show_debug_img:
            cv2.imshow("Matched image", img)
            cv2.waitKey(1)

    def analyze_image(self, img_rgb):
        start = time()
        cpy = img_rgb.copy()
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        gray_copy = img_gray.copy()
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(img_hsv, (170, 175, 220), (180, 205, 240))
        blue_mask = cv2.inRange(img_hsv, (85, 180, 205), (105, 235, 250))
        # Used to find the nearly constant color stripes next to hero icons to make the analyzed img smaller
        kf_stripes = img_gray.copy()
        kf_stripes[(blue_mask > 0)] = 255
        kf_stripes[(red_mask > 0)] = 100
        kf_stripes[(blue_mask == 0) & (red_mask == 0)] = 0
        visited: Set[Tuple[int, int]] = set()
        lines: Dict[Tuple[int, int], Tuple[int, Team]] = dict()
        dem_spots = {(i[1], i[0]): None for i in np.argwhere(kf_stripes > 0)}
        for x, y in dem_spots.keys():
            if (x, y) in visited:
                continue
            line_start = y
            temp_y = y+1
            line_end = None
            visited.add((x, y))
            while (x, temp_y) in dem_spots and (x, temp_y) not in visited:
                visited.add((x, temp_y))
                line_end = temp_y
                temp_y += 1
            if line_end and line_end - y > 30:
                color = Team.BLUE if kf_stripes[y][x] == 255 else Team.RED
                lines[(x, line_start)] = line_end, color

        merged_lines: Dict[Tuple[int, int], Tuple[int, Team]] = dict()
        todels = set()
        for line_start, (end_y, color) in lines.items():
            if line_start in todels:
                continue
            merged_lines[line_start] = end_y, color
            x, start_y = line_start
            for temp_x in range(x-3, x+6):
                for temp_y in range(start_y-5, start_y + 5):
                    if (temp_x, temp_y) in lines:
                        todels.add((temp_x, temp_y))
                        break

        heroes_found: List[KillFeedLine] = list()
        pairs: Dict[Tuple[int, int, int, int, Team, Team]] = dict()
        killfeed_lines: List[KillFeedLine] = list()
        for (start1_x, start1_y), (end1_y, color1) in merged_lines.items():
            for (start2_x, start2_y), (end2_y, color2) in merged_lines.items():
                if (start1_x, start1_y) == (start2_x, start2_y):
                    continue
                if start1_x - start2_x > 40 and start1_y - 10 < start2_y and end2_y < end1_y + 10:
                    if start1_x < start2_x:
                        pairs[(start1_x, start1_y - 10, start2_x, end1_y + 10, color1, color2)] = None
                    else:
                        pairs[start2_x, start1_y - 10, start1_x, end1_y + 10, color2, color1] = None
                    break

        timestamp = time()
        for i, pair in enumerate(pairs):
            left_hero_bgr = cpy[pair[1]:pair[3], pair[0] - 80:pair[0] + 5]
            left_hero_gray = gray_copy[pair[1]:pair[3], pair[0] - 80:pair[0] + 5]
            right_hero_bgr = cpy[pair[1]:pair[3], pair[2] - 5:pair[2] + 80]
            right_hero_gray = gray_copy[pair[1]:pair[3], pair[2] - 5:pair[2] + 80]
            ability_bgr = cpy[pair[1]:pair[3], pair[0]-10:pair[2]+10]
            kfl = KillFeedLine(threshold=self.threshold, timestamp=timestamp, hero_left_bgr=left_hero_bgr,
                               hero_left_gray=left_hero_gray, hero_right_bgr=right_hero_bgr,
                               hero_right_gray=right_hero_gray, ability_bgr=ability_bgr, team_left=pair[4],
                               team_right=pair[5])
            killfeed_lines.append(kfl)

        threads: List[Tuple[ApplyResult, KillFeedLine]] = list()
        start_find = time()
        for kfl in killfeed_lines:
            res = self.tpool.apply_async(kfl.analyze)
            threads.append((res, kfl))
        for i, (res, kfl) in enumerate(threads):
            if res.get():
                heroes_found.append(kfl)
        heroes_found.sort(reverse=True, key=lambda _kfl: _kfl.hero_left.point[1])
        return heroes_found

    def find_heroes(self, img_rgb, img_gray, threshold) -> Optional[KillFeedHero]:
        for hero, template_data in h_icons.items():
            res = cv2.matchTemplate(img_gray, template_data.template, cv2.TM_CCOEFF_NORMED, mask=template_data.mask)
            loc = np.where((res >= threshold) & (res != float('inf')))
            for pt in zip(*loc[::-1]):
                confidence = res[pt[1], pt[0]]
                hero_found = KillFeedHero(hero=hero, point=pt, confidence=confidence, template_data=template_data)
                hero_found.team = self.get_team_color(hero_found, img_rgb)
                return hero_found

        return None

    def find_abilities(self, source_img, possible_abilities: List[Tuple[Ability, TemplateData]],
                       threshold=0.70) -> Iterable[KillFeedAbility]:

        img_rgb = source_img.copy()
        # Make the picture black & Red. Everything close to red is red. Rest is black. Increases crit detection
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(img_hsv, (0, 70, 140), (10, 255, 255))
        red_mask2 = cv2.inRange(img_hsv, (170, 70, 140), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        white_mask = cv2.inRange(img_hsv, (0, 0, 230), (255, 25, 255))
        img_rgb[red_mask > 0] = (0, 0, 255, 255)
        img_rgb[white_mask > 0] = (255, 255, 255, 255)
        img_rgb[(red_mask == 0) & (white_mask == 0)] = (0, 0, 0, 255)

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        #self.debug_image(img_rgb, "rgb", None)
        #self.debug_image(img_gray, "gray")
        abilities_found: Dict[int, KillFeedAbility] = dict()
        mask_num = 0
        for ability, template_data in possible_abilities:
            if template_data.width > img_gray.shape[1] or template_data.height > img_gray.shape[0]:
                logger.error(f"ERRROR!!! img shape: {img_gray.shape}  ability shape: {template_data.template.shape}")
                continue
            res = cv2.matchTemplate(img_gray, template_data.template, cv2.TM_CCOEFF_NORMED)
            #self.debug_image(img_crit_gray, "gray", None)
            #self.debug_image(template_data.template, "template", None)
            #self.debug_image(img_crit_rgb, "brg")
            # TODO All the below. I don't remember how it works
            loc = np.where((res >= threshold) & (res != float('inf')))
            red_mask = np.zeros(img_gray.shape[:2], np.uint8)
            for pt in zip(*loc[::-1]):
                confidence = res[pt[1], pt[0]]
                midx = pt[0] + int(round(template_data.width / 2))
                midy = pt[1] + int(round(template_data.height / 2))
                mask_point = red_mask[midy, midx]
                # New match
                if mask_point == 0:
                    red_mask[pt[1]:pt[1] + template_data.height, pt[0]:pt[0] + template_data.width] = mask_num
                    abilities_found[mask_num] = KillFeedAbility(ability=ability, point=pt, confidence=confidence,
                                                                template_data=template_data)
                    mask_num += 1
                # Not new but better match
                elif abilities_found[mask_point].confidence < confidence:
                    abilities_found[mask_point].confidence = confidence
                    abilities_found[mask_point].point = pt
        return abilities_found.values()

    def get_team_color(self, kf_item: KillFeedHero, img_rbg) -> Team:
        width = kf_item.template_data.width
        height = kf_item.template_data.height
        cropped = img_rbg[kf_item.point[1] - 2: kf_item.point[1] + height + 2,
                          kf_item.point[0] - 2:kf_item.point[0] + width + 2]
        cropped[2:-2, 2:-2] = 0
        w, h = cropped.shape[:2]
        count = w * 4 + (h - 4) * 4
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

    def _chunkify_list(self, some_list, chunk_count):
        return [some_list[i::chunk_count] for i in range(chunk_count)]

    def analyze_killfeed(self):
        kfl: KillFeedLine
        for kfl in self.killfeed:
            if not kfl.reaction_sent:
                if kfl.ability:
                    kfl.reaction_sent = True
                    self.api_client.send_ow_event(kfl.hero_left.hero.value, f"{kfl.ability.ability.lower()}_kill", kfl.hero_left.team.value)
                elif kfl.critical_hit:
                    kfl.reaction_sent = True
                    self.api_client.send_ow_event(kfl.hero_left.hero.value, "critical_hit", kfl.hero_left.team.value)

        if self.killfeed_clear_time and self.multikilldetection and \
                (self.act_instantly or time() > self.killfeed_clear_time):
            for key, kill_count in self.multikilldetection.items():
                team, hero = key.split(" ")
                if kill_count > 6:
                    kill_count = 6
                if self.api_client and f"{key}{kill_count}" not in self.short_event_history:
                    if kill_count > 1:
                        self.api_client.send_ow_event(hero, self.kill_events[kill_count], team)
                        self.short_event_history.add(f"{key}{kill_count}")
        if time() > self.killfeed_clear_time:
            self.multikilldetection.clear()
            self.short_event_history.clear()

    def update_multikillcutoff(self):
        self.killfeed_clear_time = time() + self.combo_cutoff

    def debug_image(self, img, text="debug", wait: Optional[int] = 0):
        if not self.disable_debug_images:
            cv2.imshow(text, img)
            if wait is not None:
                cv2.waitKey(wait)
