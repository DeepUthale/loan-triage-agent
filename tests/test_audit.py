import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from loan_triage.domain.enums import ActorType
from loan_triage.storage.audit import GENESIS_HASH, AuditLog
from loan_triage.storage.db import Database


@pytest.fixture
def database(tmp_path: Path) -> Database:
    return Database(tmp_path / "audit.sqlite")


@pytest.fixture
def audit(database: Database) -> AuditLog:
    return AuditLog(database)


def write(audit: AuditLog, count: int = 3, case_id: str = "LN-1") -> None:
    for index in range(count):
        audit.record(
            case_id=case_id,
            actor_type=ActorType.AGENT,
            actor="triage-agent",
            stage="extract",
            action="fields_extracted",
            detail={"index": index},
        )


def tamper(path: Path, statement: str) -> None:
    """Someone with direct database access removes the triggers, then edits history."""
    connection = sqlite3.connect(path)
    connection.executescript(
        "DROP TRIGGER audit_no_update; DROP TRIGGER audit_no_delete;" + statement
    )
    connection.close()


def test_empty_log_verifies(audit: AuditLog) -> None:
    assert audit.verify().ok
    assert audit.head_hash() == GENESIS_HASH


def test_events_chain_from_genesis(audit: AuditLog) -> None:
    write(audit)
    events = audit.for_case("LN-1")
    assert events[0].prev_hash == GENESIS_HASH
    assert events[1].prev_hash == events[0].hash
    report = audit.verify()
    assert report.ok and report.events_checked == 3


def test_triggers_refuse_update_and_delete(audit: AuditLog, database: Database) -> None:
    write(audit, 1)
    with database.read() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE audit_events SET actor = 'someone-else'")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM audit_events")


def test_edited_detail_is_detected(audit: AuditLog, database: Database) -> None:
    write(audit)
    tamper(database.path, "UPDATE audit_events SET detail = '{\"index\":99}' WHERE seq = 2;")
    report = audit.verify()
    assert (report.ok, report.first_bad_seq) == (False, 2)


def test_edited_actor_is_detected(audit: AuditLog, database: Database) -> None:
    write(audit)
    tamper(database.path, "UPDATE audit_events SET actor = 'nobody' WHERE seq = 1;")
    assert audit.verify().first_bad_seq == 1


def test_deleted_event_is_detected(audit: AuditLog, database: Database) -> None:
    write(audit)
    tamper(database.path, "DELETE FROM audit_events WHERE seq = 2;")
    report = audit.verify()
    assert report.ok is False
    assert "sequence" in (report.problem or "")


def test_whole_case_erasure_is_detected(audit: AuditLog, database: Database) -> None:
    write(audit, 2, "LN-A")
    write(audit, 2, "LN-B")
    tamper(database.path, "DELETE FROM audit_events WHERE case_id = 'LN-A';")
    assert audit.verify().ok is False


def test_rehashing_one_row_still_breaks_the_next_link(audit: AuditLog, database: Database) -> None:
    write(audit)
    tamper(database.path, "UPDATE audit_events SET hash = 'forged' WHERE seq = 1;")
    assert audit.verify().ok is False


def test_full_ssn_is_masked_before_it_is_stored(audit: AuditLog) -> None:
    audit.record(
        case_id="LN-1",
        actor_type=ActorType.AGENT,
        actor="a",
        stage="s",
        action="a",
        detail={"note": "SSN 123-45-6789", "nested": [{"value": "987-65-4321"}]},
    )
    detail = audit.for_case("LN-1")[0].detail
    assert detail == {"note": "SSN XXX-XX-6789", "nested": [{"value": "XXX-XX-4321"}]}


def test_concurrent_writers_keep_the_chain_intact(audit: AuditLog) -> None:
    """Parallel document tasks all append, so the read-then-write must be serialized."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: write(audit, 5, f"LN-{i}"), range(8)))
    report = audit.verify()
    assert report.ok and report.events_checked == 40
