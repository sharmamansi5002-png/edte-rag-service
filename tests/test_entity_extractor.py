"""Tests for entity extraction."""

from src.ingestion.entity_extractor import extract_entities


def test_extracts_systems_resources_and_roles() -> None:
    text = (
        "The Finance System uses the Customer table. "
        "The Admin role can access the Finance System."
    )

    result = extract_entities(text)

    assert result["systems"] == ["Finance System"]
    assert result["resources"] == ["Customer"]
    assert result["roles"] == ["Admin"]
    assert result["actors"] == []

    assert result["resource_details"] == [
        {
            "system": "Finance System",
            "type": "table",
            "name": "Customer",
        }
    ]


def test_empty_text_returns_empty_lists() -> None:
    assert extract_entities("") == {
        "systems": [],
        "resources": [],
        "resource_details": [],
        "roles": [],
        "actors": [],
    }


def test_extracts_resource_type_and_system() -> None:
    text = (
        "The Inventory System stores information "
        "in the Product table."
    )

    result = extract_entities(text)

    assert result["resource_details"] == [
        {
            "system": "Inventory System",
            "type": "table",
            "name": "Product",
        }
    ]