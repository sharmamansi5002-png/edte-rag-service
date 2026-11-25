"""Typed data models for graph documents and entities."""

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """Metadata for an indexed document."""

    urn: str
    title: str
    uri: str


class ChunkRecord(BaseModel):
    """A text chunk and its embedding."""

    urn: str
    document_urn: str
    text: str
    chunk_index: int = Field(ge=0)
    embedding: list[float] = Field(default_factory=list)


class EntityRecord(BaseModel):
    """A system, resource, or role referenced by a chunk."""

    urn: str
    entity_type: str
    name: str


class ChunkEntityLink(BaseModel):
    """A link between a chunk and a referenced entity."""

    chunk_urn: str
    entity_urn: str
    connection_type: str = "REFERENCES"