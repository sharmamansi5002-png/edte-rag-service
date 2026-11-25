"""Extract systems, resources, roles, and actors from document text."""

import re
from typing import Any


def extract_entities(text: str) -> dict[str, Any]:
    """Extract systems, resources, roles, and actors."""

    patterns = {
        "systems": re.compile(
            r"\b((?:[A-Z][A-Za-z0-9_-]*[ \t]+)*?System)\b"
        ),
        "resources": re.compile(
            r"\b([A-Z][A-Za-z0-9_-]*)\s+(table|resource)\b",
            re.IGNORECASE,
        ),
        "roles": re.compile(
            r"\b((?:[A-Z][A-Za-z0-9_-]*\s+){0,2}"
            r"[A-Z][A-Za-z0-9_-]*)\s+role\b",
            re.IGNORECASE,
        ),
        "actors": re.compile(
            r"\b(Interns?|Employees?|Contractors?|Managers?)\b",
            re.IGNORECASE,
        ),
    }

    system_matches = list(patterns["systems"].finditer(text))

    systems = [
        re.sub(
            r"^The\s+",
            "",
            match.group(1),
            flags=re.IGNORECASE,
        ).strip()
        for match in system_matches
    ]

    unique_systems = sorted(
        {system.casefold(): system for system in systems}.values(),
        key=str.casefold,
    )

    resource_matches = list(patterns["resources"].finditer(text))

    resources = []
    resource_details = []

    for match in resource_matches:
        name = match.group(1)
        resource_type = match.group(2).lower()

        resources.append(name)

        preceding_systems = [
            system_match
            for system_match in system_matches
            if system_match.start() < match.start()
        ]

        if preceding_systems:
            system_name = re.sub(
                r"^The\s+",
                "",
                preceding_systems[-1].group(1),
                flags=re.IGNORECASE,
            ).strip()
        elif len(unique_systems) == 1:
            system_name = unique_systems[0]
        else:
            system_name = ""

        resource_details.append({
            "system": system_name,
            "type": resource_type,
            "name": name,
        })

    roles = [
        re.sub(r"^The\s+", "", role, flags=re.IGNORECASE).strip()
        for role in patterns["roles"].findall(text)
    ]

    actors = patterns["actors"].findall(text)

    return {
        "systems": unique_systems,
        "resources": sorted(set(resources), key=str.casefold),
        "resource_details": resource_details,
        "roles": sorted(set(roles), key=str.casefold),
        "actors": sorted(set(actors), key=str.casefold),
    }