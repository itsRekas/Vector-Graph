"""RDF triple parsing helpers shared by HTTP and gRPC paths."""

from __future__ import annotations

import re


def parse_rdf_triple(triple_text: str) -> dict | None:
    """Parse an RDF triple text into subject, predicate, object components."""
    pattern = r'<([^>]+)>\s+<([^>]+)>\s+(?:"([^"]+)"|<([^>]+)>)\s*\.'
    match = re.match(pattern, triple_text.strip())
    if not match:
        return None
    subject = match.group(1)
    predicate = match.group(2)
    object_literal = match.group(3)
    object_uri = match.group(4)
    return {
        "subject": subject,
        "predicate": predicate,
        "object": object_literal if object_literal else object_uri,
        "object_type": "literal" if object_literal else "uri",
    }
