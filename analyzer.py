from collections import deque, Counter
from multiprocessing.pool import ThreadPool
from time import time
from typing import Dict, Any

import cv2
import logging
import numpy as np
from mss import mss

from dbot_api_client import DbotApiClient
from enums import Hero, Team, Action
from killfeed import KillFeedItem, KillFeedLine

h_icons: Dict[Hero, Any] = dict()
for hero in Hero:
    #print(f"Hero: {hero}")
    hero_icon = cv2.cvtColor(cv2.imread("icons_with_mask/{}.png".format(hero.value.lower())), cv2.COLOR_BGR2GRAY)
    icon_mask = cv2.cvtColor(cv2.imread("icons_with_mask/{}_mask.png".format(hero.value.lower())), cv2.COLOR_BGR2GRAY)
    h_icons[hero] = (hero_icon, icon_mask)

logger = logging.getLogger(__name__)


class KillFeedAnalyzer:

    kill_events = {
        2: "double_kill",
        3: "triple_kill",
        4: "quadruple_kill",
        5: "quintuple_kill",
        6: "sextuple_kill",
    }

    def __init__(self, api_client, act_instantly=False, show_debug_img=False, debug=False, print_killfeed=True,
                 combo_cutoff=2, threshold=0.80):
        self.color = (0, 255, 0)
        self.width = 66
        self.height = 46
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
        self.font = cv2.FONT_HERSHEY_SIMPLEX


    def start_analyzer(self):
        with mss() as sct:
            mon = sct.monitors[1]
            w = 550
            h = 450
            monitor = {
                "top": mon["top"] + 15,
                "left": mon["left"] + 3820 - w,
                "width": w,
                "height": h,
                "mon": 1,
            }
            while True:
                if self.debug:
                    start = time()
                img = np.array(sct.grab(monitor))
                #img = cv2.imread("test_images/hook.jpg")
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                self.update_killfeed(img)
                self.analyze_killfeed()
                if self.debug:
                    logger.debug(f"Analyzing image took {time()-start} seconds")


    def update_killfeed(self, img):
        heroes_found = self.analyze_image(img, 8)
        timestamp = time()
        prev = None
        for i, kf_item in enumerate(heroes_found):
            if self.show_debug_img:
                #print(f"Confidence for {kf_item.hero} is {kf_item.confidence}")
                text_point = ((kf_item.point[0], kf_item.point[1] + self.height + 10))
                text = '{} {:.2f}'.format(kf_item.hero.value, kf_item.confidence)
                cv2.putText(img, text, text_point, self.font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(img, (kf_item.point[0], kf_item.point[1]),
                              (kf_item.point[0] + self.width, kf_item.point[1] + self.height), self.color, 2)
            if not prev:
                prev = kf_item
                continue
            if kf_item.point[1] - 10 < prev.point[1] < kf_item.point[1] + 10:
                if kf_item.point[0] < prev.point[0]:
                    left = kf_item
                    right = prev
                else:
                    left = prev
                    right = kf_item
                if left.hero == Hero.MERCY and left.team == right.team:
                    action = Action.RESSED
                else:
                    action = Action.KILLED
                if not (left.team == right.team and left.hero != Hero.MERCY):
                    kfl = KillFeedLine(hero_left=left, hero_right=right, timestamp=timestamp, action=action)
                    if kfl not in self.killfeed:
                        self.killfeed.append(kfl)
                        if kfl.action == Action.KILLED:
                            self.multikilldetection[str(kfl.hero_left)] += 1
                            self.update_multikillcutoff()
                        if self.print_killfeed:
                            logger.info(kfl)
                prev = None

        while self.killfeed and timestamp - self.killfeed[0].timestamp > 10:
            self.killfeed.popleft()
        if self.show_debug_img:
            cv2.imshow("Matched image", img)
            cv2.waitKey(1)

    def analyze_image(self, img_rgb, thread_count):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        heroes_found = list()
        chunks = self._chunkify_list(list(h_icons.items()), thread_count)
        tpool = ThreadPool(processes=thread_count)
        threads = list()
        for chunk in chunks:
            img_copy = img_rgb.copy()
            res = tpool.apply_async(self.find_heroes, (chunk, img_copy, img_gray, self.threshold))
            threads.append(res)
        for t in threads:
            heroes_found.extend(t.get())
        heroes_found.sort(reverse=True, key=lambda kf_item: kf_item.point[1])
        return heroes_found


    def find_heroes(self, hero_icons, img_rgb, img_gray, threshold):
        heroes_found = dict()
        width = 66
        height = 46
        mask_num = 0
        for hero, (template, template_mask) in hero_icons:
            mask_num += 1
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED, mask=template_mask)
            if self.debug:
                logger.debug(f"Best match for hero {hero} is {str(res.max())}")

            loc = np.where((res >= threshold) & (res != float('inf')))
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
            hero_found.team = self.get_team_color(hero_found, width, height, img_rgb)
        return heroes_found.values()


    def get_team_color(self, kf_item: KillFeedItem, width: int, height: int, img_rbg):
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
        for i in range(0, len(some_list), chunk_count):
            yield some_list[i:i + chunk_count]


    def analyze_killfeed(self):
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
