from mars.agents.uncertainty import compute_uncertainty


def test_high_confidence_and_consistency_gives_low_uncertainty():
    result = compute_uncertainty(confidence=0.95, consistency=0.95)
    assert result.value < 0.1


def test_low_confidence_and_consistency_gives_high_uncertainty():
    result = compute_uncertainty(confidence=0.1, consistency=0.1)
    assert result.value > 0.8


def test_dominant_factor_attribution_picks_larger_term():
    result = compute_uncertainty(confidence=0.2, consistency=0.9)
    assert result.dominant_factor == "confidence"
    assert result.breakdown["confidence_term"] > result.breakdown["consistency_term"]


def test_uncertainty_value_bounded_zero_to_one():
    result = compute_uncertainty(confidence=0.0, consistency=0.0)
    assert 0.0 <= result.value <= 1.0
