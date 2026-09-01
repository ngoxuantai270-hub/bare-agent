from src.events import record_event


def test_requests_do_not_share_default_event_state():
    assert record_event("login") == ["login"]
    assert record_event("logout") == ["logout"]
