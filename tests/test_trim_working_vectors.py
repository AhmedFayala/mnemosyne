"""Working-memory trim removes the trimmed rows' derived vectors (#1164).

``_trim_working_memory()`` runs after every ``remember()`` and
``remember_batch()``. The ``wm_ad`` trigger clears only ``fts_working``, so the
trim itself has to delete the ``vec_working`` rows (keyed by the working
row's rowid) and the ``memory_embeddings`` fallback rows (keyed by id) of
the rows it removes, and leave the vectors of every surviving row alone.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import mnemosyne.core.beam as bm


def _stamp(**delta) -> str:
    return (datetime.now(timezone.utc) - timedelta(**delta)).replace(tzinfo=None).isoformat()


def _vector(seed: int):
    return [0.1 + seed * 0.01] * bm.EMBEDDING_DIM


def _add_row(beam, memory_id, timestamp, *, session_id="trim", pinned=0, consolidated_at=None, seed=0):
    beam.conn.execute(
        "INSERT INTO working_memory(id, content, timestamp, session_id, pinned, consolidated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (memory_id, f"trim witness {memory_id}", timestamp, session_id, pinned, consolidated_at),
    )
    bm._store_working_embedding(beam.conn, memory_id, _vector(seed))
    beam.conn.commit()


def _working_rowids(beam):
    return dict(beam.conn.execute("SELECT id, rowid FROM working_memory"))


def _embedding_ids(beam):
    return {row[0] for row in beam.conn.execute("SELECT memory_id FROM memory_embeddings")}


def _vec_rowids(beam):
    return {row[0] for row in beam.conn.execute("SELECT rowid FROM vec_working")}


@pytest.fixture
def beam(tmp_path):
    memory = bm.BeamMemory(session_id="trim", db_path=tmp_path / "trim.db")
    try:
        yield memory
    finally:
        memory.conn.close()


def test_item_cap_trim_removes_vectors_of_trimmed_rows_only(beam, monkeypatch):
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1)
    _add_row(beam, "older", _stamp(minutes=30), seed=1)
    _add_row(beam, "old", _stamp(minutes=20), seed=2)
    _add_row(beam, "newest", _stamp(minutes=1), seed=3)
    _add_row(beam, "pinned", _stamp(minutes=40), pinned=1, seed=4)
    _add_row(beam, "consolidated", _stamp(minutes=40), consolidated_at=_stamp(minutes=5), seed=5)
    _add_row(beam, "other-session", _stamp(minutes=40), session_id="other", seed=6)
    vec_on = bm._wm_vec_available(beam.conn)
    if vec_on:
        assert _vec_rowids(beam) == set(_working_rowids(beam).values())

    beam._trim_working_memory()

    survivors = _working_rowids(beam)
    assert set(survivors) == {"newest", "pinned", "consolidated", "other-session"}
    assert _embedding_ids(beam) == set(survivors)
    if vec_on:
        assert _vec_rowids(beam) == set(survivors.values())


def test_ttl_trim_removes_vectors_of_expired_rows(beam, monkeypatch):
    monkeypatch.setattr(bm, "WORKING_MEMORY_TTL_HOURS", 24)
    _add_row(beam, "expired", _stamp(hours=25), seed=1)
    _add_row(beam, "fresh", _stamp(minutes=1), seed=2)
    vec_on = bm._wm_vec_available(beam.conn)

    beam._trim_working_memory()

    survivors = _working_rowids(beam)
    assert set(survivors) == {"fresh"}
    assert _embedding_ids(beam) == {"fresh"}
    if vec_on:
        assert _vec_rowids(beam) == {survivors["fresh"]}


def test_trim_keeps_fallback_vector_while_episodic_row_owns_the_id(beam, monkeypatch):
    """``memory_embeddings`` carries only ``memory_id``; an episodic row with
    the same id still owns it after the working row is trimmed (#1002)."""
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1)
    _add_row(beam, "shared", _stamp(minutes=30), seed=1)
    _add_row(beam, "newest", _stamp(minutes=1), seed=2)
    beam.conn.execute(
        "INSERT INTO episodic_memory(id, content, timestamp, session_id) VALUES (?, ?, ?, ?)",
        ("shared", "episodic copy", _stamp(minutes=30), "trim"),
    )
    beam.conn.commit()
    vec_on = bm._wm_vec_available(beam.conn)

    beam._trim_working_memory()

    survivors = _working_rowids(beam)
    assert set(survivors) == {"newest"}
    assert _embedding_ids(beam) == {"shared", "newest"}
    if vec_on:
        assert _vec_rowids(beam) == {survivors["newest"]}


def test_failed_trim_rolls_back_vector_deletes(beam, monkeypatch):
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1)
    _add_row(beam, "old", _stamp(minutes=30), seed=1)
    _add_row(beam, "newest", _stamp(minutes=1), seed=2)
    before_vec = _vec_rowids(beam) if bm._wm_vec_available(beam.conn) else None
    beam.conn.execute(
        "CREATE TRIGGER fail_trim BEFORE DELETE ON working_memory "
        "BEGIN SELECT RAISE(ABORT, 'trim blocked'); END"
    )
    beam.conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="trim blocked"):
        beam._trim_working_memory()

    assert not beam.conn.in_transaction
    assert set(_working_rowids(beam)) == {"old", "newest"}
    assert _embedding_ids(beam) == {"old", "newest"}
    if before_vec is not None:
        assert _vec_rowids(beam) == before_vec


def test_remember_trim_removes_vectors_of_the_row_it_evicts(beam, monkeypatch):
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 2)
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        bm._embeddings, "embed", lambda values: [_vector(len(text)) for text in values]
    )
    first = beam.remember("first trim witness", source="test")
    second = beam.remember("second trim witness, longer", source="test")
    for memory_id, minutes in ((first, 30), (second, 20)):
        beam.conn.execute(
            "UPDATE working_memory SET timestamp = ? WHERE id = ?", (_stamp(minutes=minutes), memory_id)
        )
    beam.conn.commit()
    assert _embedding_ids(beam) == {first, second}

    third = beam.remember("third trim witness, the longest one", source="test")

    survivors = _working_rowids(beam)
    assert set(survivors) == {second, third}
    assert _embedding_ids(beam) == {second, third}
    if bm._wm_vec_available(beam.conn):
        assert _vec_rowids(beam) == set(survivors.values())
