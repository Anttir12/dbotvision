import unittest
from unittest import mock

import cv2
import ddt

from analyzer import KillFeedAnalyzer
from enums import Hero, Team, Ability

test_cases = (
        ("artillery.png", )
)


@ddt.ddt
class KillFeedAnalyzerTestCase(unittest.TestCase):

    @ddt.data(
        ("artillery.png", [(Team.BLUE, Hero.BASTION, Ability.ARTILLERY, Team.RED, Hero.BASTION)]),
        ("artillery.png", [(Team.BLUE, Hero.BASTION, Ability.ARTILLERY, Team.RED, Hero.BASTION)]),
        ("artillery.png", [(Team.BLUE, Hero.BASTION, Ability.ARTILLERY, Team.RED, Hero.BASTION)]),
        ("artillery.png", [(Team.BLUE, Hero.BASTION, Ability.ARTILLERY, Team.RED, Hero.BASTION)]),
        ("artillery.png", [(Team.BLUE, Hero.BASTION, Ability.ARTILLERY, Team.RED, Hero.BASTION)]),
        ("artillery.png", [(Team.BLUE, Hero.BASTION, Ability.ARTILLERY, Team.RED, Hero.BASTION)]),
        ("artillery.png", [(Team.BLUE, Hero.BASTION, Ability.ARTILLERY, Team.RED, Hero.BASTION)]),
        ("artillery.png", [(Team.BLUE, Hero.BASTION, Ability.ARTILLERY, Team.RED, Hero.BASTION)]),
        ("artillery.png", [(Team.BLUE, Hero.BASTION, Ability.ARTILLERY, Team.RED, Hero.BASTION)]),
        ("artillery.png", [(Team.BLUE, Hero.BASTION, Ability.ARTILLERY, Team.RED, Hero.BASTION)]),
        ("artillery.png", [(Team.BLUE, Hero.BASTION, Ability.ARTILLERY, Team.RED, Hero.BASTION)]),
        )
    @ddt.unpack
    def test_image_analyzing(self, img_name, expected_killfeed):
        kfa = KillFeedAnalyzer(mock.Mock(), thread_count=1)
        img = cv2.imread(f'test_images/{img_name}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        killfeed_lines = kfa.analyze_image(img)
        self.assertEqual(1, len(killfeed_lines))
        self.assertEqual(killfeed_lines[0].hero_left.hero, Hero.BASTION)
        self.assertEqual(killfeed_lines[0].hero_left.team, Team.BLUE)
        self.assertEqual(killfeed_lines[0].hero_right.hero, Hero.BASTION)
        self.assertEqual(killfeed_lines[0].hero_right.team, Team.RED)
        self.assertEqual(killfeed_lines[0].ability, Ability.ARTILLERY)
        print("test")


if __name__ == '__main__':
    unittest.main()
