def test_boolean_contract_independence():
    # Contract-level assertion: a confirmed fall can exist before an AMR approaches.
    event = {"fallen_onto_ground": True, "amr_approach": False, "signal_level": "low"}
    assert event["fallen_onto_ground"] is True
    assert event["amr_approach"] is False
