import pytest
import numpy as np
import cv2
from src.rules.wrong_side import WrongSideDetector
from src.rules.red_light import RedLightDetector
from src.rules.illegal_parking import IllegalParkingDetector
from src.detection.triple_riding import TripleRidingDetector
from src.detection.seatbelt import SeatbeltDetector
from src.detection.helmet import HelmetDetector
from src.detection.phone_call import PhoneCallDetector

class TestWrongSideDetector:
    def test_allowed_direction_no_violation(self):
        detector = WrongSideDetector(allowed_vector=(0.0, 1.0), min_displacement_px=15.0)
        track = [
            (200.0, 100.0, 0.0),
            (200.0, 150.0, 1.0),
            (200.0, 200.0, 2.0),
            (200.0, 250.0, 3.0),
        ]
        is_violating, conf, meta = detector.check_violation(track)
        assert not is_violating
        assert conf == 0.0

    def test_wrong_way_direction_triggers_violation(self):
        detector = WrongSideDetector(allowed_vector=(0.0, 1.0), min_displacement_px=15.0)
        track = [
            (200.0, 300.0, 0.0),
            (200.0, 250.0, 1.0),
            (200.0, 200.0, 2.0),
            (200.0, 150.0, 3.0),
        ]
        is_violating, conf, meta = detector.check_violation(track)
        assert is_violating
        assert conf >= 0.70
        assert meta["divergence_angle_deg"] >= 120.0

    def test_stationary_jitter_does_not_trigger(self):
        detector = WrongSideDetector(allowed_vector=(0.0, 1.0), min_displacement_px=20.0)
        track = [
            (200.0, 200.0, 0.0),
            (201.0, 200.0, 1.0),
            (200.0, 199.0, 2.0),
            (200.0, 200.0, 3.0),
        ]
        is_violating, conf, meta = detector.check_violation(track)
        assert not is_violating


class TestRedLightDetector:
    def test_crossing_stop_line_on_red(self):
        detector = RedLightDetector(stop_line=((100.0, 400.0), (500.0, 400.0)), signal_state="RED")
        p_prev = (300.0, 380.0)
        p_curr = (300.0, 420.0)
        is_violating, conf, meta = detector.check_violation(track_id=10, p_prev=p_prev, p_curr=p_curr, timestamp_sec=2.5)
        assert is_violating
        assert conf >= 0.90
        assert meta["signal_state"] == "RED"

    def test_crossing_stop_line_on_green_no_violation(self):
        detector = RedLightDetector(stop_line=((100.0, 400.0), (500.0, 400.0)), signal_state="GREEN")
        p_prev = (300.0, 380.0)
        p_curr = (300.0, 420.0)
        is_violating, conf, meta = detector.check_violation(track_id=11, p_prev=p_prev, p_curr=p_curr, timestamp_sec=3.0)
        assert not is_violating

    def test_stopping_before_line_on_red_no_violation(self):
        detector = RedLightDetector(stop_line=((100.0, 400.0), (500.0, 400.0)), signal_state="RED")
        p_prev = (300.0, 360.0)
        p_curr = (300.0, 390.0)
        is_violating, conf, meta = detector.check_violation(track_id=12, p_prev=p_prev, p_curr=p_curr, timestamp_sec=4.0)
        assert not is_violating


class TestIllegalParkingDetector:
    def test_moving_through_no_parking_zone_no_violation(self):
        polygon = [(100.0, 100.0), (400.0, 100.0), (400.0, 400.0), (100.0, 400.0)]
        detector = IllegalParkingDetector(no_parking_zone=polygon, min_duration_sec=5.0)

        t1, c1, m1 = detector.update_and_check(track_id=1, position=(150.0, 150.0), timestamp_sec=0.0)
        t2, c2, m2 = detector.update_and_check(track_id=1, position=(250.0, 250.0), timestamp_sec=2.0)
        t3, c3, m3 = detector.update_and_check(track_id=1, position=(380.0, 380.0), timestamp_sec=6.0)
        assert not t1 and not t2 and not t3

    def test_stationary_inside_no_parking_zone_triggers_violation(self):
        polygon = [(100.0, 100.0), (400.0, 100.0), (400.0, 400.0), (100.0, 400.0)]
        detector = IllegalParkingDetector(no_parking_zone=polygon, min_duration_sec=5.0)

        t1, _, _ = detector.update_and_check(track_id=2, position=(200.0, 200.0), timestamp_sec=1.0)
        assert not t1

        t2, _, m2 = detector.update_and_check(track_id=2, position=(201.0, 200.0), timestamp_sec=4.0)
        assert not t2

        t3, conf, m3 = detector.update_and_check(track_id=2, position=(201.0, 201.0), timestamp_sec=6.5)
        assert t3
        assert conf >= 0.70
        assert m3["dwell_time_sec"] >= 5.0


class TestTripleRidingDetector:
    def test_two_riders_no_violation(self):
        detector = TripleRidingDetector(rider_count_threshold=3, min_consecutive_frames=3)
        moto_box = (100.0, 100.0, 200.0, 250.0)
        riders = [(120.0, 80.0, 160.0, 160.0), (140.0, 90.0, 180.0, 170.0)]

        for t in range(5):
            is_viol, conf, meta = detector.check_motorcycle(
                moto_track_id=5,
                moto_box=moto_box,
                all_person_boxes=riders,
                video_timestamp=float(t)
            )
            assert not is_viol

    def test_three_riders_sustained_triggers_violation(self):
        detector = TripleRidingDetector(rider_count_threshold=3, min_consecutive_frames=3)
        moto_box = (100.0, 100.0, 200.0, 250.0)
        riders = [
            (110.0, 70.0, 140.0, 150.0),
            (135.0, 75.0, 165.0, 155.0),
            (160.0, 80.0, 190.0, 160.0)
        ]

        v1, _, _ = detector.check_motorcycle(6, moto_box, riders, 0.0)
        assert not v1
        v2, _, _ = detector.check_motorcycle(6, moto_box, riders, 0.5)
        assert not v2
        v3, conf, meta = detector.check_motorcycle(6, moto_box, riders, 1.0)
        assert v3
        assert conf >= 0.80
        assert meta["rider_count"] >= 3

    def test_still_image_triple_riding_triggers_instantly(self):
        detector = TripleRidingDetector(rider_count_threshold=3)
        moto_box = (100.0, 100.0, 200.0, 250.0)
        riders = [
            (110.0, 70.0, 140.0, 150.0),
            (135.0, 75.0, 165.0, 155.0),
            (160.0, 80.0, 190.0, 160.0)
        ]
        is_viol, conf, meta = detector.check_motorcycle(7, moto_box, riders, 0.0, is_still_image=True)
        assert is_viol
        assert conf >= 0.85
        assert meta["rider_count"] == 3


class TestSeatbeltDetector:
    def test_non_car_returns_indeterminate(self):
        detector = SeatbeltDetector()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        status, conf, meta = detector.predict_seatbelt(img, vehicle_type="motorcycle")
        assert status == "indeterminate"

    def test_glare_windshield_returns_indeterminate(self):
        detector = SeatbeltDetector()
        glare_img = np.full((120, 120, 3), 250, dtype=np.uint8)
        status, conf, meta = detector.predict_seatbelt(glare_img, vehicle_type="car")
        assert status == "indeterminate"
        assert "glare" in meta["reason"].lower()


class TestPhoneCallDetector:
    def test_no_phone_normal_rider_posture(self):
        detector = PhoneCallDetector()
        # Clean image of rider with hands down at handlebar level (no skin at ear level)
        frame = np.full((200, 100, 3), 40, dtype=np.uint8)
        person_box = (0, 0, 100, 200)
        is_call, conf, meta, roi = detector.check_phone_call(frame, person_box)
        assert not is_call
        assert conf == 0.0

    def test_direct_cell_phone_overlap_triggers(self):
        detector = PhoneCallDetector()
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        person_box = (10, 10, 90, 190)
        # Phone bounding box located at upper head/ear level
        phone_box = [60, 30, 85, 70]
        is_call, conf, meta, roi = detector.check_phone_call(frame, person_box, detected_phone_boxes=[phone_box])
        assert is_call
        assert conf >= 0.90
        assert meta["method"] == "direct_cell_phone_detector"
