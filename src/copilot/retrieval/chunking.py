"""Split Markdown into retrievable sections.

Each section becomes one embedded chunk. We split on a heading prefix and drop
everything before the first heading (title lines, HTML comments), so preamble
never pollutes the index.
"""

from __future__ import annotations


def split_markdown_sections(text: str, heading_prefix: str = "### ") -> list[tuple[str, str]]:
    """Return ``(title, body)`` pairs for each ``heading_prefix`` section."""
    sections: list[tuple[str, str]] = []
    title: str | None = None
    buffer: list[str] = []

    for line in text.splitlines():
        if line.startswith(heading_prefix):
            if title is not None:
                sections.append((title, "\n".join(buffer).strip()))
            title = line[len(heading_prefix):].strip()
            buffer = []
        elif title is not None:
            buffer.append(line)

    if title is not None:
        sections.append((title, "\n".join(buffer).strip()))
    return sections
