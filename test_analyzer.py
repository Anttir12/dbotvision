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
        ("artillery.png", [(Team.BLUE, Hero.BASTION, Ability.ARTILLERY, False, Team.RED, Hero.BASTION)]),
        ("boosters.png", [(Team.BLUE, Hero.DVA, Ability.BOOSTERS, False, Team.RED, Hero.TRACER)]),
        ("callmech.png", [(Team.BLUE, Hero.DVA, Ability.CALL_MECH, False, Team.RED, Hero.TRACER)]),
        ("charge.png", [(Team.BLUE, Hero.REINHARDT, Ability.CHARGE, False, Team.RED, Hero.TRACER)]),
        ("concussionmine.png", [(Team.BLUE, Hero.JUNKRAT, Ability.CONCUSSION_MINE, False, Team.RED, Hero.TRACER)]),
        ("dynamite.png", [(Team.BLUE, Hero.ASHE, Ability.DYNAMITE, False, Team.RED, Hero.TRACER)]),
        ("firestrike.png", [(Team.BLUE, Hero.REINHARDT, Ability.FIRE_STRIKE, False, Team.RED, Hero.TRACER)]),
        ("grenade.png", [(Team.BLUE, Hero.BASTION, Ability.GRENADE, False, Team.RED, Hero.TRACER)]),
        ("hook.png", [(Team.BLUE, Hero.ROADHOG, Ability.HOOK, False, Team.RED, Hero.TRACER)]),
        ("javelin.png", [(Team.BLUE, Hero.ORISA, Ability.JAVELIN, False, Team.RED, Hero.TRACER)]),
        ("jumppack.png", [(Team.BLUE, Hero.WINSTON, Ability.JUMP_PACK, False, Team.RED, Hero.TRACER)]),
        ("micromissiles.png", [(Team.BLUE, Hero.DVA, Ability.MICRO_MISSILES, False, Team.RED, Hero.TRACER)]),
        ("moltencore.png", [(Team.BLUE, Hero.TORBJORN, Ability.MOLTEN_CORE, False, Team.RED, Hero.TORBJORN)]),
        ("overclock.png", [(Team.BLUE, Hero.SOJOURN, Ability.OVERCLOCK, True, Team.RED, Hero.TRACER)]),
        ("primalrage.png", [(Team.BLUE, Hero.WINSTON, Ability.PRIMAL_RAGE, False, Team.RED, Hero.TRACER)]),
        ("pulsebomb.png", [(Team.BLUE, Hero.TRACER, Ability.PULSE_BOMB, False, Team.RED, Hero.TRACER)]),
        ("railgun.png", [(Team.BLUE, Hero.SOJOURN, Ability.RAILGUN, True, Team.RED, Hero.TRACER)]),
        ("self-destruct.png", [(Team.BLUE, Hero.DVA, Ability.SELF_DESTRUCT, False, Team.RED, Hero.TRACER)]),
        ("soundwave.png", [(Team.BLUE, Hero.LUCIO, Ability.SOUNDWAVE, False, Team.RED, Hero.TRACER)]),
        ("stormarrow.png", [(Team.BLUE, Hero.HANZO, Ability.STORM_ARROW, False, Team.RED, Hero.TRACER)]),
        ("trap.png", [(Team.BLUE, Hero.JUNKRAT, Ability.TRAP, False, Team.RED, Hero.TRACER)]),
        ("turret.png", [(Team.BLUE, Hero.TORBJORN, Ability.TURRET, False, Team.RED, Hero.TORBJORN)]),
        ("whipshot.png", [(Team.BLUE, Hero.BRIGITTE, Ability.WHIP_SHOT, False, Team.RED, Hero.TRACER)]),
        ("error1.png", []),
        ("criticalhit.png", [(Team.BLUE, Hero.ASHE, None, True, Team.RED, Hero.MERCY)]),
        ("criticalhit1.png", [(Team.BLUE, Hero.KIRIKO, None, True, Team.RED, Hero.LUCIO)]),
        ("criticalhit2.png", [(Team.BLUE, Hero.ASHE, None, True, Team.RED, Hero.SOLDIER)]),
        ("criticalhit3.png", [(Team.BLUE, Hero.SOJOURN, Ability.RAILGUN, True, Team.RED, Hero.JUNKRAT)]),
        ("criticalhit4.png", [(Team.BLUE, Hero.ASHE, None, True, Team.RED, Hero.TRACER)]),
        ("four_kills_with_ult_effect.png", [(Team.BLUE, Hero.HANZO, None, True, Team.RED, Hero.SYMMETRA),
                                            (Team.RED, Hero.BASTION, Ability.ARTILLERY, False, Team.BLUE, Hero.KIRIKO),
                                            (Team.RED, Hero.ZENYATTA, None, True, Team.BLUE, Hero.SOJOURN),
                                            (Team.RED, Hero.BASTION, Ability.ARTILLERY, False, Team.BLUE, Hero.HANZO)]),
        ("sextuple.png", [(Team.BLUE, Hero.ASHE, Ability.BOB, False, Team.RED, Hero.WRECKINGBALL),
                          (Team.BLUE, Hero.ASHE, Ability.BOB, False, Team.RED, Hero.LUCIO),
                          (Team.BLUE, Hero.ASHE, Ability.BOB, False, Team.RED, Hero.JUNKERQUEEN),
                          (Team.BLUE, Hero.ASHE, Ability.BOB, False, Team.RED, Hero.HANZO),
                          (Team.BLUE, Hero.ASHE, Ability.BOB, False, Team.RED, Hero.ANA),
                          (Team.BLUE, Hero.ASHE, None, True, Team.RED, Hero.MERCY)]),
        ("spectate_mode.png", [(Team.BLUE, Hero.HANZO, None, False, Team.RED, Hero.REAPER),
                               (Team.RED, Hero.MOIRA, None, False, Team.BLUE, Hero.ANA)]),
        ("two_kills.png", [(Team.RED, Hero.MOIRA, None, False, Team.BLUE, Hero.HANZO),
                           (Team.BLUE, Hero.SOJOURN, Ability.RAILGUN, True, Team.RED, Hero.ORISA)]),
        )
    @ddt.unpack
    def test_image_analyzing(self, img_name, expected_killfeed):
        kfa = KillFeedAnalyzer(mock.Mock(), thread_count=1)
        img = cv2.imread(f'test_images/{img_name}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        killfeed_lines = kfa.analyze_image(img)
        self.assertEqual(len(expected_killfeed), len(killfeed_lines))
        if img_name == "sextuple.png":
            print("tres")
        for i, kfl in enumerate(killfeed_lines):
            self.assertEqual(kfl.hero_left.team, expected_killfeed[i][0])
            self.assertEqual(kfl.hero_left.hero, expected_killfeed[i][1])
            self.assertEqual(kfl.ability, expected_killfeed[i][2])
            self.assertEqual(kfl.critical_hit, expected_killfeed[i][3])
            self.assertEqual(kfl.hero_right.team, expected_killfeed[i][4])
            self.assertEqual(kfl.hero_right.hero, expected_killfeed[i][5])


if __name__ == '__main__':
    unittest.main()
