"""Tests for photo review/sorting: ``media_image_quality`` and
``media_image_sort``.

Images are synthesized with OpenCV (available in the server virtualenv) and
analyzed with the same pure functions the server uses, so the tests exercise
the real blur/brightness/contrast math without touching ffmpeg or the network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import media.server
from media import media as media_mod


def _write_png(
    path: Path, width: int, height: int, color: tuple[int, int, int]
) -> None:
    """Write a solid-color PNG with OpenCV (BGR)."""
    import numpy as np

    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.imwrite(str(path), img)


def _write_blurred_png(path: Path, width: int, height: int) -> None:
    """Write an out-of-focus image: solid color then a Gaussian blur."""
    import numpy as np

    cv2 = __import__("cv2")
    img = np.full((height, width, 3), (120, 140, 160), dtype=np.uint8)
    # Some structure so it is not perfectly flat, then blur it heavily.
    cv2.rectangle(
        img,
        (width // 4, height // 4),
        (3 * width // 4, 3 * height // 4),
        (40, 60, 80),
        -1,
    )
    cv2.GaussianBlur(img, (31, 31), 0, img)
    cv2.imwrite(str(path), img)


def _tier_of(result: str) -> str:
    for line in result.splitlines():
        if line.startswith("Quality tier:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no Quality tier line in:\n{result}")


def test_image_stats_keys_and_ranges(tmp_path):
    path = tmp_path / "sharp.png"
    _write_png(path, 400, 300, (10, 200, 50))
    metrics = media_mod._image_stats(path)
    assert metrics["width"] == 400
    assert metrics["height"] == 300
    assert metrics["resolution"] == 120000
    assert metrics["format"] == "PNG"
    assert metrics["size_bytes"] > 0
    assert metrics["blur_score"] >= 0
    assert 0 <= metrics["brightness"] <= 255
    assert 0 <= metrics["contrast"] <= 255
    assert len(metrics["channels"]) == 3
    assert "brisque_score" in metrics


def test_blurred_image_has_lower_blur_score(tmp_path):
    sharp = tmp_path / "sharp.png"
    blurred = tmp_path / "blurred.png"
    _write_checkerboard(sharp, 300, 300)
    _write_blurred_png(blurred, 300, 300)
    sharp_metrics = media_mod._image_stats(sharp)
    blurred_metrics = media_mod._image_stats(blurred)
    assert sharp_metrics["blur_score"] > blurred_metrics["blur_score"]


def _write_checkerboard(path: Path, width: int, height: int) -> None:
    """High-frequency pattern so the Laplacian blur score is clearly positive."""
    import numpy as np

    cv2 = __import__("cv2")
    tile = 20
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                img[y : y + tile, x : x + tile] = (200, 200, 200)
    cv2.imwrite(str(path), img)


def _write_noisy_png(path: Path, width: int, height: int) -> None:
    """Fine mid-gray noise: high blur score, good brightness and contrast.

    The checkerboard's gray std (~100) is "poor" contrast, so tests that need
    an image classified ``good`` overall use this instead: single-channel
    uniform noise in [70, 190) replicated to BGR has brightness ~130, gray
    std ~35, and large Laplacian variance. Deterministic via a fixed seed.
    """
    import numpy as np

    cv2 = __import__("cv2")
    rng = np.random.default_rng(42)
    gray = rng.uniform(70, 190, size=(height, width)).astype(np.uint8)
    img = np.stack([gray, gray, gray], axis=-1)
    cv2.imwrite(str(path), img)


def _tier_of(result: str) -> str:
    for line in result.splitlines():
        if line.startswith("Quality tier:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no Quality tier line in:\n{result}")


def test_quality_tier_classification():
    # Blur poor drives overall poor regardless of other metrics.
    assert (
        media_mod.quality_tier(
            {"blur_score": 10.0, "brightness": 100.0, "contrast": 50.0}
        )
        == "poor"
    )
    # Brightness extreme drives overall poor.
    assert (
        media_mod.quality_tier(
            {"blur_score": 600.0, "brightness": 240.0, "contrast": 50.0}
        )
        == "poor"
    )
    assert (
        media_mod.quality_tier(
            {"blur_score": 600.0, "brightness": 20.0, "contrast": 50.0}
        )
        == "poor"
    )
    # Contrast extreme drives overall poor.
    assert (
        media_mod.quality_tier(
            {"blur_score": 600.0, "brightness": 120.0, "contrast": 10.0}
        )
        == "poor"
    )
    assert (
        media_mod.quality_tier(
            {"blur_score": 600.0, "brightness": 120.0, "contrast": 90.0}
        )
        == "poor"
    )
    # BRISQUE poor drives overall worst-wins tier.
    assert (
        media_mod.quality_tier(
            {
                "blur_score": 600.0,
                "brightness": 120.0,
                "contrast": 50.0,
                "brisque_score": 60.0,
            }
        )
        == "poor"
    )
    # All metrics good => overall good.
    assert (
        media_mod.quality_tier(
            {"blur_score": 600.0, "brightness": 120.0, "contrast": 50.0}
        )
        == "good"
    )
    # All metrics fair => overall fair.
    assert (
        media_mod.quality_tier(
            {"blur_score": 300.0, "brightness": 120.0, "contrast": 50.0}
        )
        == "fair"
    )
    # Missing brisque_score is excluded from worst-wins, not penalized.
    assert (
        media_mod.quality_tier(
            {
                "blur_score": 600.0,
                "brightness": 120.0,
                "contrast": 50.0,
                "brisque_score": None,
            }
        )
        == "good"
    )
    # BRISQUE good (<25) contributes good tier.
    assert (
        media_mod.quality_tier(
            {
                "blur_score": 600.0,
                "brightness": 120.0,
                "contrast": 50.0,
                "brisque_score": 10.0,
            }
        )
        == "good"
    )
    # BRISQUE fair (25-50) contributes fair tier.
    assert (
        media_mod.quality_tier(
            {
                "blur_score": 600.0,
                "brightness": 120.0,
                "contrast": 50.0,
                "brisque_score": 35.0,
            }
        )
        == "fair"
    )


def test_media_image_quality(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    path = tmp_path / "photo.png"
    _write_png(path, 640, 480, (30, 180, 90))
    result = asyncio.run(media.server.media_image_quality(str(path)))
    assert "# media_image_quality" in result
    assert f"Source: {path}" in result
    assert "Resolution: 640x480" in result
    assert "Quality tier:" in result
    assert "Blur score (Laplacian variance):" in result
    assert "Format: PNG" in result
    assert "skipped:" in result
    # The tier artifact was written to the cache entry.
    quality_json = next(cache.rglob("quality.json"))
    payload = json.loads(quality_json.read_text())
    assert payload["tier"] in {"poor", "fair", "good"}
    assert payload["source"] == str(path)
    assert "brisque_score" in payload["metrics"]
    # BRISQUE scoring may fail on some images (package/numpy compatibility);
    # when it succeeds, the score line appears in the output.
    if payload["metrics"]["brisque_score"] is not None:
        assert "BRISQUE score (lower is better):" in result


def test_media_image_quality_flag_poor(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    path = tmp_path / "blurry.png"
    _write_blurred_png(path, 300, 300)
    # flag_poor=True should append FLAG: POOR when the tier is poor.
    result_flagged = asyncio.run(
        media.server.media_image_quality(str(path), flag_poor=True)
    )
    assert "FLAG: POOR" in result_flagged, result_flagged
    # Default (flag_poor=False) should not append FLAG: POOR.
    result_plain = asyncio.run(media.server.media_image_quality(str(path)))
    assert "FLAG: POOR" not in result_plain, result_plain


def test_brisque_score_degrades_gracefully():
    """_brisque_score returns None or a float when scoring an image."""
    import numpy as np

    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    score = media_mod._brisque_score(img)
    assert score is None or isinstance(score, float)


def test_media_image_sort_keps_best_first(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    sharp = tmp_path / "sharp.png"
    blurred = tmp_path / "blurred.png"
    _write_checkerboard(sharp, 300, 300)
    _write_blurred_png(blurred, 300, 300)
    result = asyncio.run(
        media.server.media_image_sort(
            [str(sharp), str(blurred)], by="blur", order="desc"
        )
    )
    items = [
        line
        for line in result.splitlines()
        if line.strip() and line[0].isdigit() and ". " in line
    ]
    # First kept image should be the sharp one (higher blur score).
    assert str(sharp) in items[0], items
    assert str(blurred) in items[1], items
    assert "Kept: 2 of 2" in result


def test_media_image_sort_filters_by_min_quality(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    sharp = tmp_path / "sharp.png"
    blurred = tmp_path / "blurred.png"
    _write_checkerboard(sharp, 300, 300)
    _write_blurred_png(blurred, 300, 300)
    result = asyncio.run(
        media.server.media_image_sort(
            [str(sharp), str(blurred)], by="quality", order="desc", min_quality="good"
        )
    )
    assert "Kept:" in result
    # The blurred image is poor and should be filtered out by min_quality=good.
    assert str(blurred) not in result, result


def test_media_image_sort_writes_default_artifact(tmp_path, monkeypatch):
    """The ranking is saved under <cache>/sorts/ and named in the response."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    sharp = tmp_path / "sharp.png"
    blurred = tmp_path / "blurred.png"
    _write_checkerboard(sharp, 300, 300)
    _write_blurred_png(blurred, 300, 300)
    result = asyncio.run(
        media.server.media_image_sort([str(sharp), str(blurred)], by="blur")
    )
    artifact_line = next(
        line for line in result.splitlines() if line.startswith("Artifact: ")
    )
    artifact = Path(artifact_line.split(": ", 1)[1])
    assert artifact.exists()
    assert cache in artifact.parents
    payload = json.loads(artifact.read_text())
    assert payload["tool"] == "media_image_sort"
    assert payload["params"]["by"] == "blur"
    assert payload["summary"] == {"total": 2, "kept": 2}
    assert [row["source"] for row in payload["kept"]] == [str(sharp), str(blurred)]
    assert all(row["kept"] is True for row in payload["kept"])
    assert payload["filtered"] == []


def test_media_image_sort_writes_explicit_output(tmp_path, monkeypatch):
    """`output` writes the ranking to that exact path."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    good = tmp_path / "good.png"
    blurred = tmp_path / "blurred.png"
    _write_noisy_png(good, 300, 300)
    _write_blurred_png(blurred, 300, 300)
    output = tmp_path / "reports" / "ranking.json"
    result = asyncio.run(
        media.server.media_image_sort(
            [str(good), str(blurred)],
            by="quality",
            order="desc",
            min_quality="good",
            output=str(output),
        )
    )
    assert f"Artifact: {output}" in result
    assert output.exists()
    payload = json.loads(output.read_text())
    assert payload["summary"] == {"total": 2, "kept": 1}
    assert [row["source"] for row in payload["kept"]] == [str(good)]
    assert [row["source"] for row in payload["filtered"]] == [str(blurred)]
    assert payload["filtered"][0]["kept"] is False
