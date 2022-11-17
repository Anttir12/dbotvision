import traceback
from collections import deque, Counter
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool, ApplyResult
from time import time
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
    ability_area: Tuple[int, int, int, int]
    hero_left_area: Tuple[int, int, int, int]
    hero_right_area: Tuple[int, int, int, int]
    gray_img: np.ndarray
    bgr_img: np.ndarray
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

    def __post_init__(self):
        self.hero_left_bgr = self.bgr_img[self.hero_left_area[0]:self.hero_left_area[1],
                                          self.hero_left_area[2]:self.hero_left_area[3]]
        self.hero_left_gray = self.gray_img[self.hero_left_area[0]:self.hero_left_area[1],
                                            self.hero_left_area[2]:self.hero_left_area[3]]
        self.hero_right_bgr = self.bgr_img[self.hero_right_area[0]:self.hero_right_area[1],
                                           self.hero_right_area[2]:self.hero_right_area[3]]
        self.hero_right_gray = self.gray_img[self.hero_right_area[0]:self.hero_right_area[1],
                                             self.hero_right_area[2]:self.hero_right_area[3]]
        self.ability_bgr = self.bgr_img[self.ability_area[0]:self.ability_area[1],
                                        self.ability_area[2]:self.ability_area[3]]

    def is_valid(self):
        return self.hero_left_gray.shape[0] >= 48 and \
               self.hero_left_gray.shape[1] >= 66 and \
               self.hero_right_gray.shape[0] >= 48 and \
               self.hero_right_gray.shape[1] >= 66

    def draw_debug_line(self, img, font):
        hero_height, hero_width, color = 48, 66, (0, 255, 0)

        for kf_item in [self.hero_left, self.hero_right]:
            if kf_item is None:
                continue
            point = kf_item.point[0], kf_item.point[1]
            text_point = (point[0], point[1] + hero_height + 10)
            text = '{} {:.2f}'.format(kf_item.hero.value, kf_item.confidence)
            cv2.putText(img, text, text_point, font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (point[0], point[1]), (point[0] + hero_width, point[1] + hero_height), color, 2)

    def draw_search_area(self, img):
        color = (255, 0, 0)
        cv2.rectangle(img, (self.hero_left_area[2], self.hero_left_area[0]),
                      (self.hero_left_area[3], self.hero_left_area[1]), color, 2)
        cv2.rectangle(img, (self.hero_right_area[2], self.hero_right_area[0]),
                      (self.hero_right_area[3], self.hero_right_area[1]), color, 2)

    def analyze(self):
        self.determine_heroes()
        if self.hero_left:
            possible_abilities = [(a, a_icons[a]) for a in HERO_ABILITY_MAP[self.hero_left.hero]]
            self.determine_ability(possible_abilities)
        return self.hero_left and self.hero_right

    def determine_heroes(self):
        self.hero_left = self._find_hero(self.hero_left_gray, self.hero_left_area)
        if self.hero_left:
            self.hero_left.team = self.team_left
            if self.hero_left.hero == Hero.MERCY or self.team_left != self.team_right:
                self.hero_right = self._find_hero(self.hero_right_gray, self.hero_right_area)
                if self.hero_right:
                    self.hero_right.team = self.team_right
                    if self.hero_left.team == self.hero_right.team:
                        self.action = Action.RESSED

    def _find_hero(self, img_gray, area) -> Optional[KillFeedHero]:
        for hero, template_data in h_icons.items():
            res = cv2.matchTemplate(img_gray, template_data.template, cv2.TM_CCOEFF_NORMED, mask=template_data.mask)
            loc = np.where((res >= self.threshold) & (res != float('inf')))
            for pt in zip(*loc[::-1]):
                confidence = res[pt[1], pt[0]]
                point = (pt[0] + area[2], pt[1] + area[0])
                hero_found = KillFeedHero(hero=hero, point=point, confidence=confidence, template_data=template_data)
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
                          threshold=0.70):

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
        for ability, template_data in possible_abilities:
            if template_data.width > img_gray.shape[1] or template_data.height > img_gray.shape[0]:
                logger.error(f"ERRROR!!! img shape: {img_gray.shape}  ability shape: {template_data.template.shape}")
                continue
            res = cv2.matchTemplate(img_gray, template_data.template, cv2.TM_CCOEFF_NORMED)
            # debug_image(img_crit_gray, "gray", None)
            # debug_image(template_data.template, "template", None)
            # debug_image(img_crit_rgb, "brg")
            loc = np.where((res >= threshold) & (res != float('inf')))
            for pt in zip(*loc[::-1]):
                confidence = res[pt[1], pt[0]]
                if confidence > threshold:
                    if ability == Ability.CRITICAL_HIT:
                        self.critical_hit = True
                    else:
                        self.ability = ability

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
        return self.debug and self.i % 60 == 0

    def __init__(self, api_client, act_instantly=False, show_debug_img=False, debug=False, print_killfeed=True,
                 combo_cutoff=2, threshold=0.73, thread_count=6):
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
        self.width = 550
        self.height = 600

    def __del__(self):
        self.tpool.close()
        self.tpool.terminate()

    def start_analyzer(self):
        with mss() as sct:
            mon = sct.monitors[1]
            monitor = {
                "top": mon["top"] + 15,
                "left": mon["left"] + 3820 - self.width,
                "width": self.width,
                "height": self.height,
                "mon": 1,
            }
            img = None
            fps = '-'
            fps_counter = 0
            fps_start = time()
            try:
                while True:
                    self.i += 1
                    start = time()
                    filename = None
                    filename = "test_images/sextuple.png"
                    if not filename:
                        img = np.array(sct.grab(monitor))
                    else:
                        img = cv2.imread(filename)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    self.update_killfeed(img)
                    self.analyze_killfeed()
                    if self._if_debug_print():
                        logger.debug(f"Analyzing loop took {time()-start} seconds")
                    if self.show_debug_img:
                        fps_counter += 1
                        if time() - fps_start > 1:
                            fps = f'{fps_counter}'
                            fps_counter = 0
                            fps_start = time()
                        cv2.putText(img, fps, (30, 30), self.font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.imshow("Matched image", img)
                        cv2.waitKey(1)

            except Exception as e:
                traceback.print_exc()
                logger.error(e)
                cv2.imshow("last_image", img)

    def update_killfeed(self, img):
        start = time()
        killfeed_lines = self.analyze_image(img)
        if self._if_debug_print():
            logger.debug(f"Analyzing image took {time() - start} seconds")
        timestamp = time()
        for kfl in killfeed_lines:
            if self.show_debug_img:
                kfl.draw_debug_line(img, self.font)
            if kfl not in self.killfeed:
                self.killfeed.append(kfl)
                if kfl.action == Action.KILLED:
                    self.multikilldetection[str(kfl.hero_left)] += 1
                    self.update_multikillcutoff()
                if self.print_killfeed:
                    logger.info(kfl)

        while self.killfeed and timestamp - self.killfeed[0].timestamp > 10:
            self.killfeed.popleft()

    def analyze_image(self, img_rgb):
        cpy = img_rgb.copy()
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        gray_copy = img_gray.copy()
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        #  ORIGINAL
        #  red_mask = cv2.inRange(img_hsv, (170, 175, 220), (180, 205, 240))
        #  blue_mask = cv2.inRange(img_hsv, (85, 180, 205), (105, 235, 250))
        red_mask = cv2.inRange(img_hsv, (130, 130, 210), (180, 210, 250))
        blue_mask = cv2.inRange(img_hsv, (80, 175, 200), (110, 255, 255))
        # Used to find the nearly constant color stripes next to hero icons to make the analyzed img smaller
        kf_stripes = img_gray.copy()
        kf_stripes[(blue_mask > 0)] = 255
        kf_stripes[(red_mask > 0)] = 100
        kf_stripes[(blue_mask == 0) & (red_mask == 0)] = 0
        if self.show_debug_img:
            debug_image(kf_stripes, wait=None)
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
            skips = 4
            color = Team.BLUE if kf_stripes[y][x] == 255 else Team.RED

            def skip_possible(_y):
                possible = False
                for _temp_y in range(_y + 1, _y + skips):
                    if (x, _temp_y) in dem_spots:
                        possible = True
                        break
                return possible

            while (x, temp_y) in dem_spots or skip_possible(temp_y) and (x, temp_y) not in visited:
                visited.add((x, temp_y))
                line_end = temp_y
                temp_y += 1
            if line_end and line_end - y > 30:
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

        pairs: Dict[Tuple[int, int, int, int, Team, Team]] = dict()
        killfeed_lines: List[KillFeedLine] = list()
        for (start1_x, start1_y), (end1_y, color1) in merged_lines.items():
            for (start2_x, start2_y), (end2_y, color2) in merged_lines.items():
                if (start1_x, start1_y) == (start2_x, start2_y):
                    continue
                if start1_x - start2_x > 33 and start1_y - 20 < start2_y and end2_y < end1_y + 30:
                    if start1_x < start2_x:
                        pairs[(start1_x, max(0, start1_y - 20), start2_x, end1_y + 20, color1, color2)] = None
                    else:
                        pairs[start2_x, max(0, start1_y - 20), start1_x, end1_y + 20, color2, color1] = None
                    break

        debug_copy = None
        if self.show_debug_img:
            debug_copy = cpy.copy()
        for i, pair in enumerate(pairs):
            timestamp = time()
            kfl = KillFeedLine(threshold=self.threshold, timestamp=timestamp,
                               hero_left_area=(pair[1], pair[3], pair[0] - 160, pair[0] + 5),
                               hero_right_area=(pair[1], pair[3], pair[2] - 5, pair[2] + 160),
                               ability_area=(pair[1], pair[3], pair[0]-10, pair[2]+10),
                               bgr_img=cpy, gray_img=gray_copy, team_left=pair[4], team_right=pair[5])
            if kfl.is_valid():
                killfeed_lines.append(kfl)
                if self.show_debug_img:
                    kfl.draw_search_area(debug_copy)

        if self.show_debug_img:
            cv2.imshow("test123", debug_copy)

        threads: List[Tuple[ApplyResult, KillFeedLine]] = list()
        for kfl in killfeed_lines:
            res = self.tpool.apply_async(kfl.analyze)
            threads.append((res, kfl))
        heroes_found_sorted_set: Dict[KillFeedLine, None] = dict()
        for i, (res, kfl) in enumerate(threads):
            if res.get():
                heroes_found_sorted_set[kfl] = None
        heroes_found: List[KillFeedLine] = list(heroes_found_sorted_set.keys())
        heroes_found.sort(reverse=True, key=lambda _kfl: _kfl.hero_left.point[1])
        return heroes_found

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


def debug_image(img, text="debug", wait: Optional[int] = 0):
    cv2.imshow(text, img)
    if wait is not None:
        cv2.waitKey(wait)
