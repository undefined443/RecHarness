from gagc.grpo import compute_group_advantages


# ------------------------------------------------------------------ #
# compute_group_advantages                                             #
# ------------------------------------------------------------------ #


def test_advantages_zero_mean_unit_variance():
    rewards = [1.0, 2.0, 3.0, 4.0]
    adv = compute_group_advantages(rewards)
    mu = sum(adv) / len(adv)
    variance = sum(a ** 2 for a in adv) / len(adv)
    assert abs(mu) < 1e-6, "advantages should be zero-mean"
    assert abs(variance - 1.0) < 1e-6, "advantages should be unit-variance"


def test_advantages_degenerate_all_same():
    rewards = [0.5, 0.5, 0.5]
    adv = compute_group_advantages(rewards)
    assert all(abs(a) < 1e-6 for a in adv), "identical rewards -> zero advantages"


def test_advantages_preserves_length():
    rewards = [0.1, 0.9, 0.5, 0.3]
    adv = compute_group_advantages(rewards)
    assert len(adv) == len(rewards)


def test_advantages_empty():
    assert compute_group_advantages([]) == []
