"""Utilities for generating consistent entity URNs."""

import re


def _normalize(value: str) -> str:
    """Normalize a name for use in a URN."""
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    normalized = normalized.strip("_")

    if not normalized:
        raise ValueError("URN components must not be empty.")

    return normalized.lower()


def system_urn(system_id: str) -> str:
    return f"urn:system:{_normalize(system_id)}"


def resource_urn(system_id: str, resource_type: str, name: str) -> str:
    return (
        f"urn:resource:{_normalize(system_id)}:"
        f"{_normalize(resource_type)}:{_normalize(name)}"
    )


def role_urn(role_id: str) -> str:
    return f"urn:role:{_normalize(role_id)}"


def actor_urn(actor_id: str) -> str:
    return f"urn:actor:{_normalize(actor_id)}"


def step_urn(trace_id: str, step_index: int) -> str:
    if step_index < 0:
        raise ValueError("Step index cannot be negative.")
    return f"urn:step:{_normalize(trace_id)}:{step_index}"


def document_urn(doc_id: str) -> str:
    return f"urn:doc:{_normalize(doc_id)}"


def chunk_urn(doc_id: str, chunk_index: int) -> str:
    if chunk_index < 0:
        raise ValueError("Chunk index cannot be negative.")
    return f"urn:chunk:{_normalize(doc_id)}:{chunk_index}"