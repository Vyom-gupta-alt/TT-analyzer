import numpy as np

from ttvision import sample_match as sm


def test_ball_meets_each_racket_at_contact():
    rallies = sm.plan_match(7)
    assert len(rallies) == 4
    for r in rallies:
        for s in r.shots:
            ball = sm.ball_at(r, s.contact)
            assert ball is not None
            assert np.linalg.norm(ball - s.hit) < 1.0          # cm


def test_players_feet_stay_behind_their_end():
    rallies = sm.plan_match(7)
    for t in np.arange(0, 45, 0.5):
        near, _ = sm.player_pose(rallies, "near", float(t))
        far, _ = sm.player_pose(rallies, "far", float(t))
        feet = [27, 28, 29, 30, 31, 32]                       # arms may reach over the table, feet not
        assert near[feet, 1].max() < -sm.HALF_L
        assert far[feet, 1].min() > sm.HALF_L


def test_serve_bounces_on_both_halves():
    r = sm.plan_match(7)[0]
    tables = [t for t, k in r.events if k == "table"]
    first, second = sm.ball_at(r, tables[0]), sm.ball_at(r, tables[1])
    assert first[1] < 0 < second[1]
