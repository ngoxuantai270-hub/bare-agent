from src.events import record_event


def test_explicit_event_list_is_still_supported():
    events = ["start"]
    assert record_event("finish", events) == ["start", "finish"]


def test_fresh_default_after_explicit_list():
    assert record_event("fresh") == ["fresh"]
