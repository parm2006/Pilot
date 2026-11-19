import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from pilot_voice import HandGate, is_open_palm


def open_hand():
    # A splayed palm labeled Left by the model, in metres.
    return np.array([
        (0, .045, 0),
        (-.018, .025, 0), (-.035, .010, 0), (-.050, -.005, 0), (-.065, -.020, 0),
        (-.025, 0, 0), (-.035, -.025, 0), (-.041, -.043, 0), (-.047, -.061, 0),
        (-.005, -.005, 0), (-.007, -.035, 0), (-.008, -.057, 0), (-.009, -.079, 0),
        (.013, 0, 0), (.021, -.028, 0), (.027, -.048, 0), (.033, -.068, 0),
        (.028, .012, 0), (.043, -.011, 0), (.052, -.028, 0), (.061, -.045, 0),
    ])


def detection(points=None, side="Left", score=.99):
    points = open_hand() if points is None else points
    world = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]
    image = [SimpleNamespace(x=.5 + 4 * x, y=.5 + 4 * y) for x, y, _ in points]
    return image, world, [SimpleNamespace(category_name=side, score=score)], 1


def curl(points, base):
    points[base + 1:base + 4] = points[base] + np.array([
        (0, -.020, 0), (0, -.012, -.015), (0, .008, -.020),
    ])


class PalmGestureTests(unittest.TestCase):
    def test_left_and_right_palms_are_accepted(self):
        for mirrored in (False, True):
            points = open_hand()
            if mirrored:
                points[:, 0] *= -1
            side = "Right" if mirrored else "Left"
            with self.subTest(side=side):
                self.assertTrue(is_open_palm(*detection(points, side)))

    def test_in_plane_rotation_and_hand_size_do_not_change_gesture(self):
        for angle in (0, 45, 90, 180, 270):
            radians = np.radians(angle)
            rotation = np.array([
                (np.cos(radians), -np.sin(radians), 0),
                (np.sin(radians), np.cos(radians), 0), (0, 0, 1),
            ])
            for scale in (.6, 1, 1.3):
                with self.subTest(angle=angle, scale=scale):
                    self.assertTrue(is_open_palm(*detection(open_hand() @ rotation.T * scale)))

    def test_back_of_hand_is_rejected_for_either_side(self):
        self.assertFalse(is_open_palm(*detection(side="Right")))
        points = open_hand()
        points[:, 0] *= -1
        self.assertFalse(is_open_palm(*detection(points, "Left")))

    def test_edge_on_palm_is_rejected_but_moderate_tilt_is_allowed(self):
        for degrees, expected in ((25, True), (75, False), (90, False)):
            angle = np.radians(degrees)
            rotation = np.array([
                (np.cos(angle), 0, np.sin(angle)), (0, 1, 0),
                (-np.sin(angle), 0, np.cos(angle)),
            ])
            with self.subTest(degrees=degrees):
                self.assertEqual(is_open_palm(*detection(open_hand() @ rotation.T)), expected)

    def test_each_finger_must_be_extended(self):
        for base in (1, 5, 9, 13, 17):
            points = open_hand()
            curl(points, base)
            with self.subTest(base=base):
                self.assertFalse(is_open_palm(*detection(points)))

    def test_fist_pointing_and_thumbs_up_are_rejected(self):
        for extended in (None, 5, 1):
            points = open_hand()
            for base in (1, 5, 9, 13, 17):
                if base != extended:
                    curl(points, base)
            with self.subTest(extended=extended):
                self.assertFalse(is_open_palm(*detection(points)))

    def test_straight_fingers_together_are_rejected(self):
        points = open_hand()
        for base in (5, 9, 13, 17):
            points[base + 1:base + 4] = points[base] + np.array([
                (0, -.025, 0), (0, -.045, 0), (0, -.065, 0),
            ])
        self.assertFalse(is_open_palm(*detection(points)))

    def test_missing_uncertain_clipped_or_invalid_landmarks_are_rejected(self):
        image, world, side, aspect = detection()
        self.assertFalse(is_open_palm([], [], [], aspect))
        self.assertFalse(is_open_palm(image[:-1], world, side, aspect))
        self.assertFalse(is_open_palm(image, world[:-1], side, aspect))
        self.assertFalse(is_open_palm(*detection(score=.6)))
        self.assertFalse(is_open_palm(*detection(side="Unknown")))
        self.assertFalse(is_open_palm(*detection(score=float("nan"))))
        image[4].x = -0.1
        self.assertFalse(is_open_palm(image, world, side, aspect))
        for value in (0, float("nan"), float("inf")):
            self.assertFalse(is_open_palm(*detection(np.full((21, 3), value))))


class HandGateTests(unittest.TestCase):
    def gate(self, *hands, every=1):
        model = Mock()
        model.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[hand[0] for hand in hands],
            hand_world_landmarks=[hand[1] for hand in hands],
            handedness=[hand[2] for hand in hands],
        )
        return HandGate(landmarker=model, every=every)

    def test_detection_checks_gesture_and_considers_second_hand(self):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        fist = open_hand()
        for base in (1, 5, 9, 13, 17):
            curl(fist, base)
        self.assertFalse(self.gate()._detect(frame))
        self.assertFalse(self.gate(detection(fist))._detect(frame))
        self.assertTrue(self.gate(detection(fist), detection())._detect(frame))

    def test_brief_detection_does_not_mute_and_hold_expires(self):
        gate = self.gate()
        self.assertFalse(gate.update(None, 0, seen=True))
        self.assertFalse(gate.update(None, .1, seen=False))
        self.assertFalse(gate.update(None, .2, seen=True))
        self.assertTrue(gate.update(None, .3, seen=True))
        self.assertTrue(gate.update(None, .5, seen=False))
        self.assertFalse(gate.update(None, 1.01, seen=False))

    def test_only_gesture_checks_refresh_mute(self):
        gate = self.gate(detection())
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        self.assertFalse(gate.update(frame, 0))
        self.assertTrue(gate.update(frame, .1))
        gate.lm.detect_for_video.return_value.handedness[0][0].category_name = "Right"
        self.assertTrue(gate.update(frame, .4))
        self.assertFalse(gate.update(frame, .81))

    def test_frame_sampling_is_preserved(self):
        gate = self.gate(detection(), every=3)
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        for n in range(1, 7):
            self.assertEqual(gate.update(frame, n / 30), n == 6)
        self.assertEqual(gate.lm.detect_for_video.call_count, 2)


if __name__ == "__main__":
    unittest.main()
