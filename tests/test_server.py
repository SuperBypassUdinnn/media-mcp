"""Unit tests for `media_frames`'s `image_inline` route.

The stdio contract tests cover the wire shape of the same call; these drive the
tool body directly, so the block construction, the inline cap, and the
degrade-when-missing path are exercised without a server process.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest
from mcp_types import ContentBlock, ImageContent, TextContent

from media import media, server
from tests import fixtures


def _split(result: str | list[ContentBlock]) -> tuple[str, list[ImageContent]]:
    """Split a tool result into its text and its image blocks."""
    if isinstance(result, str):
        return result, []
    text = "\n".join(block.text for block in result if isinstance(block, TextContent))
    return text, [block for block in result if isinstance(block, ImageContent)]


def _png_paths(text: str) -> list[Path]:
    """Return the absolute PNG paths the text printed, in order."""
    paths: list[Path] = []
    for line in text.splitlines():
        candidate = line.strip()
        if candidate.startswith("[") and "]" in candidate:
            candidate = candidate.split("]", 1)[1].strip()
        if candidate.startswith("/") and candidate.endswith(".png"):
            paths.append(Path(candidate))
    return paths


@pytest.fixture
def source(tmp_path, monkeypatch) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    return fixtures.make_motion_patch(tmp_path / "clip.mp4")


def test_image_inline_attaches_the_png_the_tool_wrote(source):
    result = asyncio.run(
        server.media_frames(str(source), at=[0.5, 1.0], image_inline=True)
    )
    text, images = _split(result)
    paths = _png_paths(text)
    assert len(paths) == len(images) == 2
    for image, path in zip(images, paths, strict=True):
        assert image.type == "image" and image.mime_type == "image/png"
        assert base64.b64decode(image.data) == path.read_bytes()


def test_image_inline_is_off_by_default(source):
    result = asyncio.run(server.media_frames(str(source), at=[0.5, 1.0]))
    assert isinstance(result, str)
    assert len(_png_paths(result)) == 2


def test_more_frames_than_the_inline_cap_are_capped_and_reported(source):
    # The cap value is stated in docs/design.md section 5, so pin it here:
    # changing it must force a docs and test decision.
    assert server.INLINE_IMAGE_CAP == 8
    stamps = [round(0.1 * n, 3) for n in range(1, server.INLINE_IMAGE_CAP + 3)]
    result = asyncio.run(
        server.media_frames(
            str(source), at=stamps, count=len(stamps), image_inline=True
        )
    )
    text, images = _split(result)
    assert len(images) == server.INLINE_IMAGE_CAP
    assert (
        f"capped: inline: attached {server.INLINE_IMAGE_CAP} of {len(stamps)} PNGs "
        f"(inline cap {server.INLINE_IMAGE_CAP})" in text
    )
    assert len(_png_paths(text)) == len(stamps)


def test_a_per_tile_label_failure_reaches_the_status_block(source, monkeypatch):
    # A degraded tile is never silent: the renderer's note travels into the
    # response's `skipped:` line with the reason it fell back to unlabelled.
    doomed = 1.0
    real_label_tile = media._label_tile

    def flaky(frame, timestamp, dimensions, family, font, scratch_dir=None):
        if timestamp == doomed:
            raise media.MediaError("label burn simulated failure")
        return real_label_tile(frame, timestamp, dimensions, family, font, scratch_dir)

    monkeypatch.setattr(media, "_label_tile", flaky)
    result = asyncio.run(
        server.media_frames(str(source), at=[0.5, 1.0, 1.5], tile="sheet")
    )

    assert isinstance(result, str)
    assert "skipped:" in result, result
    assert "label" in result.lower() and "simulated failure" in result, result


def test_a_missing_png_degrades_to_its_path_and_says_so(source, monkeypatch):
    real_is_file = Path.is_file

    def hiding_frames(self: Path) -> bool:
        if self.parent.name == "frames" and self.suffix == ".png":
            return False
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", hiding_frames)
    result = asyncio.run(server.media_frames(str(source), at=[0.5], image_inline=True))
    text, images = _split(result)
    assert images == []
    assert len(_png_paths(text)) == 1
    assert "skipped:" in text and "missing" in text
