from gnosis.supersession import FactFreshness, drop_superseded, slot_key


def test_slot_key_extracted_fact_uses_first_entity() -> None:
    key = slot_key("user:1", "fact", ["Biscuit", "Maria"])
    assert key == ("user:1", "fact", "biscuit")


def test_slot_key_extracted_fact_without_entity_has_no_slot() -> None:
    assert slot_key("user:1", "fact", []) is None
    assert slot_key("user:1", "fact", ["   "]) is None


def test_slot_key_typed_predicate_uses_subject_predicate() -> None:
    assert slot_key("User:1", "Lives_In", []) == ("user:1", "lives_in")


def test_slot_key_conversation_predicates_never_supersede() -> None:
    assert slot_key("user:1", "memory", []) is None
    assert slot_key("user:1", "said_user", []) is None
    assert slot_key("user:1", "said_assistant", []) is None


def test_drop_superseded_keeps_only_newest_in_slot_by_event_date() -> None:
    older = FactFreshness(
        ("u", "fact", "biscuit"),
        event_date="2024-01-01",
        observation_date=None,
        created_at="2026-01-01",
    )
    newer = FactFreshness(
        ("u", "fact", "biscuit"),
        event_date="2024-06-01",
        observation_date=None,
        created_at="2025-01-01",
    )
    kept, dropped = drop_superseded([older, newer], lambda item: item)
    assert kept == [newer]
    assert dropped == 1


def test_drop_superseded_uses_observation_date_without_event_dates() -> None:
    # observation_date (conversation_date) used as recency when no event_date
    older = FactFreshness(
        ("u", "lives_in"),
        event_date=None,
        observation_date="2023-05-01",
        created_at="2026-01-01T00:00:00Z",
    )
    newer = FactFreshness(
        ("u", "lives_in"),
        event_date=None,
        observation_date="2024-03-01",
        created_at="2026-01-01T00:00:00Z",
    )
    kept, dropped = drop_superseded([newer, older], lambda item: item)
    assert kept == [newer]
    assert dropped == 1


def test_drop_superseded_falls_back_to_created_at_without_dates() -> None:
    older = FactFreshness(
        ("u", "lives_in"),
        event_date=None,
        observation_date=None,
        created_at="2023-05-07T00:00:00Z",
    )
    newer = FactFreshness(
        ("u", "lives_in"),
        event_date=None,
        observation_date=None,
        created_at="2026-06-28T00:00:00Z",
    )
    kept, dropped = drop_superseded([newer, older], lambda item: item)
    assert kept == [newer]
    assert dropped == 1


def test_drop_superseded_keeps_both_on_tie() -> None:
    first = FactFreshness(
        ("u", "fact", "biscuit"),
        event_date="2024-01-01",
        observation_date=None,
        created_at=None,
    )
    second = FactFreshness(
        ("u", "fact", "biscuit"),
        event_date="2024-01-01",
        observation_date=None,
        created_at=None,
    )
    kept, dropped = drop_superseded([first, second], lambda item: item)
    assert kept == [first, second]
    assert dropped == 0


def test_drop_superseded_keeps_incomparable_facts() -> None:
    # One has only event_date, the other only created_at: not comparable.
    dated = FactFreshness(
        ("u", "fact", "biscuit"),
        event_date="2024-01-01",
        observation_date=None,
        created_at=None,
    )
    stamped = FactFreshness(
        ("u", "fact", "biscuit"),
        event_date=None,
        observation_date=None,
        created_at="2026-01-01T00:00:00Z",
    )
    kept, dropped = drop_superseded([dated, stamped], lambda item: item)
    assert kept == [dated, stamped]
    assert dropped == 0


def test_drop_superseded_keeps_different_slots() -> None:
    biscuit = FactFreshness(
        ("u", "fact", "biscuit"),
        event_date="2024-01-01",
        observation_date=None,
        created_at=None,
    )
    rex = FactFreshness(
        ("u", "fact", "rex"),
        event_date="2024-06-01",
        observation_date=None,
        created_at=None,
    )
    kept, dropped = drop_superseded([biscuit, rex], lambda item: item)
    assert kept == [biscuit, rex]
    assert dropped == 0


def test_drop_superseded_keeps_facts_without_slot_keys() -> None:
    plain = FactFreshness(
        None, event_date="2024-01-01", observation_date=None, created_at=None
    )
    other = FactFreshness(
        None, event_date="2024-06-01", observation_date=None, created_at=None
    )
    kept, dropped = drop_superseded([plain, other], lambda item: item)
    assert kept == [plain, other]
    assert dropped == 0


def test_drop_superseded_collapses_chain_to_single_newest() -> None:
    slot = ("u", "fact", "biscuit")
    a = FactFreshness(
        slot, event_date="2024-01-01", observation_date=None, created_at=None
    )
    b = FactFreshness(
        slot, event_date="2024-03-01", observation_date=None, created_at=None
    )
    c = FactFreshness(
        slot, event_date="2024-06-01", observation_date=None, created_at=None
    )
    kept, dropped = drop_superseded([a, b, c], lambda item: item)
    assert kept == [c]
    assert dropped == 2
