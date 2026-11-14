"""The documentation matches the code it describes.

Four failure modes are guarded here, all of them silent in a normal test run: a
CLI flag that exists but is undocumented, an HTTP route with no entry, a page
nobody links to, and a relative link or anchor that points at nothing.

Nothing here needs a GPU, a network, downloaded models or a built frontend, and
`server/app.py` is read as text rather than imported, so the test does not
require FastAPI.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

import dub

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

#: Pages that are deliberately outside the index: the documentation license and
#: the demo-media README (reached from its own folder and from the root README).
UNINDEXED = {"LICENSE.md", "media/README.md"}

#: Upper bound per page, in lines. Generous limits: they exist to stop a page
#: from silently turning into a second manual, not to police formatting.
SIZE_LIMITS = {
    "README.md": 240,
    "docs/getting-started.md": 640,
    "docs/usage.md": 340,
    "docs/reference.md": 480,
    "docs/voices.md": 700,
    "docs/dub-quality.md": 460,
    "docs/performance.md": 240,
    "docs/outputs-and-caching.md": 340,
    "docs/troubleshooting.md": 240,
    "docs/development/guidelines.md": 320,
    "CONTRIBUTING.md": 130,
}

_MARKDOWN_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_FENCE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)
_ROUTE = re.compile(r'@app\.(?:get|post|delete|patch|put)\(\s*"([^"]+)"')
_ROUTE_PARAM = re.compile(r"\{[^}]+\}")
_PARSERS = (
    "build_dub_parser",
    "build_subtitles_parser",
    "build_say_parser",
    "build_voices_parser",
    "build_models_parser",
    "build_anchors_parser",
    "build_info_parser",
    "build_devices_parser",
)


def _pages() -> list[Path]:
    return sorted(DOCS.rglob("*.md"))


def _prose(path: Path) -> str:
    """The page with fenced code removed, so examples cannot look like links."""
    return _FENCE.sub("", path.read_text(encoding="utf-8"))


def _anchor(heading: str) -> str:
    """GitHub's heading anchor: drop markup and punctuation, spaces become dashes."""
    text = re.sub(r"[`*_\[\](){}:,.!?'\"–—]", "", heading.strip().lower())
    return re.sub(r"\s+", "-", text)


def _anchors(path: Path) -> set[str]:
    return {_anchor(m.group(1)) for m in _HEADING.finditer(_prose(path))}


def _flags(parser_name: str) -> set[str]:
    parser = getattr(dub, parser_name)()
    found: set[str] = set()
    stack: list[argparse.ArgumentParser] = [parser]
    while stack:
        current = stack.pop()
        for action in current._actions:  # noqa: SLF001 - argparse exposes no API
            for option in action.option_strings:
                if option.startswith("--") and option != "--help":
                    found.add(option)
            if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
                stack.extend(action.choices.values())
    return found


def test_every_page_is_reachable_from_the_index() -> None:
    index = _prose(DOCS / "README.md")
    missing = [
        page.relative_to(DOCS).as_posix()
        for page in _pages()
        if page.relative_to(DOCS).as_posix() not in UNINDEXED
        and page.relative_to(DOCS).as_posix() not in index
    ]
    assert not missing, (
        "these pages are not linked from docs/README.md: "
        f"{missing}. Add them to the index or delete them."
    )


def test_relative_links_and_anchors_resolve() -> None:
    broken: list[str] = []
    for page in [*sorted(ROOT.glob("*.md")), *_pages()]:
        for match in _MARKDOWN_LINK.finditer(_prose(page)):
            target = match.group(1)
            if "://" in target or target.startswith("mailto:"):
                continue
            if target.startswith("#"):
                if target[1:] not in _anchors(page):
                    broken.append(f"{page.relative_to(ROOT)} -> {target}")
                continue
            path_part, _, anchor = target.partition("#")
            destination = (page.parent / path_part).resolve()
            if not destination.exists():
                broken.append(f"{page.relative_to(ROOT)} -> {target}")
            elif anchor and destination.suffix == ".md" and anchor not in _anchors(destination):
                broken.append(f"{page.relative_to(ROOT)} -> {target} (no such heading)")
    assert not broken, "broken documentation links:\n" + "\n".join(broken)


@pytest.mark.parametrize("parser_name", _PARSERS)
def test_every_cli_flag_is_in_the_reference(parser_name: str) -> None:
    documented = _prose(DOCS / "reference.md")
    missing = sorted(flag for flag in _flags(parser_name) if flag not in documented)
    assert not missing, (
        f"{parser_name} exposes flags missing from docs/reference.md: {missing}"
    )


def test_every_http_route_is_in_the_reference() -> None:
    document = _ROUTE_PARAM.sub("{x}", (DOCS / "reference.md").read_text(encoding="utf-8"))
    routes = [
        _ROUTE_PARAM.sub("{x}", route)
        for route in _ROUTE.findall((ROOT / "server" / "app.py").read_text(encoding="utf-8"))
        if "full_path" not in route  # the SPA catch-all
    ]
    missing = sorted({route for route in routes if route not in document})
    assert not missing, f"routes missing from docs/reference.md: {missing}"


def test_readme_makes_the_app_visible() -> None:
    page = _prose(ROOT / "README.md")
    assert "dub-server" in page, "the README must show how to start the browser app"
    assert "docs/README.md" in page, "the README must link the documentation index"
    assert "Two ways to use it" in page, "the README must present the app as a first-class surface"


def test_pages_stay_within_their_budget() -> None:
    oversized = {
        name: len((ROOT / name).read_text(encoding="utf-8").splitlines())
        for name, limit in SIZE_LIMITS.items()
        if len((ROOT / name).read_text(encoding="utf-8").splitlines()) > limit
    }
    assert not oversized, (
        "pages over their line budget: "
        f"{oversized}. Split the page, or merge the duplicate content elsewhere."
    )
