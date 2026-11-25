"""Tests for URN generation."""

import pytest

from src.ingestion.urns import (
    actor_urn,
    chunk_urn,
    document_urn,
    resource_urn,
    role_urn,
    step_urn,
    system_urn,
)


def test_system_urn() -> None:
    assert system_urn("Finance System") == "urn:system:finance_system"


def test_resource_urn() -> None:
    assert resource_urn(
        "finance",
        "table",
        "Customer",
    ) == "urn:resource:finance:table:customer"


def test_role_urn() -> None:
    assert role_urn("Admin") == "urn:role:admin"


def test_actor_urn() -> None:
    assert actor_urn("John") == "urn:actor:john"


def test_step_urn() -> None:
    assert step_urn("trace123", 2) == "urn:step:trace123:2"


def test_document_urn() -> None:
    assert document_urn("Finance Document") == "urn:doc:finance_document"


def test_chunk_urn() -> None:
    assert chunk_urn("finance", 3) == "urn:chunk:finance:3"


def test_empty_urn_component_raises_error() -> None:
    with pytest.raises(ValueError):
        system_urn("   ")


def test_negative_step_index_raises_error() -> None:
    with pytest.raises(ValueError):
        step_urn("trace123", -1)


def test_negative_chunk_index_raises_error() -> None:
    with pytest.raises(ValueError):
        chunk_urn("finance", -1)