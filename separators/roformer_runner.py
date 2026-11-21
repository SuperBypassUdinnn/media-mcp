"""Headless Roformer stem separation runner for the media-mcp vocal ladder.

Executed as a subprocess under the Ultimate Vocal Remover venv interpreter
(torch 2.12+cu130 at ``<UVR_ROOT>/.venv``), pointed at by
``MEDIA_SEPARATE_BIN`` (mirroring ``MEDIA_RHYTHM_BIN``). Usage::

    roformer_runner.py <media.wav> <out_dir> [--model <name>] [--stems vocals,instrumental]

Imports the ``roformer_native`` model definitions from the UVR repo root
(``UVR_ROOT`` env var, default
``~/.clone/Github/<hash>/ultimatevocalremovergui``), loads the matching
config yaml from ``models/MDX_Net_Models/model_data/mdx_c_configs/*.yaml``,
runs chunked inference (``chunk_size: 588800``, ``dim_t: 801``), checks free
VRAM before GPU use and falls back to CPU, and writes
``<out_dir>/vocals.wav`` plus ``<out_dir>/instrumental.wav``.

This module is import-safe without torch: all heavy imports (torch,
``roformer_native``) live under ``if __name__ == "__main__":`` or are lazy,
so media/.venv (no torch) can import this module and read its docstring only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# UVR repo root: overridden by UVR_ROOT, otherwise the repository the brief
# pins. Defaults are relative to this file's location so the runner stays
# coherent when moved alongside the repo.
_DEFAULT_UVR_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "ultimatevocalremovergui"
)

# Default checkpoint the brief names as the community-reference vocals extractor.
_DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

# Chunked-inference geometry from the UVR config yaml family.
_CHUNK_SIZE = 588800
_DIM_T = 801


def _uvr_root() -> Path:
    """Resolve the UVR repository root from ``UVR_ROOT`` or the default."""
    return Path(os.environ.get("UVR_ROOT") or _DEFAULT_UVR_ROOT)


def _config_dir(uvr_root: Path) -> Path:
    return uvr_root / "models" / "MDX_Net_Models" / "model_data" / "mdx_c_configs"


def _checkpoint_path(uvr_root: Path, model: str) -> Path:
    """Resolve a checkpoint: bare name looks under UVR checkpoints dir."""
    path = Path(model)
    if path.is_file():
        return path
    candidates = [
        uvr_root / "checkpoints" / model,
        uvr_root / model,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path


def _load_config(config_dir: Path, model: str) -> dict:
    """Load the config yaml matching ``model`` from the config dir."""
    import yaml

    # Strip extension and try `<stem>.yaml`, then `<stem>_config.yaml`.
    stem = Path(model).stem
    for name in (f"{stem}.yaml", f"{stem}_config.yaml", f"{stem}.yml"):
        candidate = config_dir / name
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
    raise FileNotFoundError(f"no mdx_c config yaml for model {model!r} in {config_dir}")


def _separate(
    media_wav: Path,
    out_dir: Path,
    *,
    model: str = _DEFAULT_MODEL,
    stems: tuple[str, ...] = ("vocals", "instrumental"),
) -> None:
    """Run Roformer separation and write the requested stems.

    Heavy imports (torch, roformer_native) are local to this function so the
    module stays import-safe without torch.
    """
    import torch
    import torchaudio
    from roformer_native.bs_roformer import BSRoformer

    uvr_root = _uvr_root()
    config_dir = _config_dir(uvr_root)
    config = _load_config(config_dir, model)
    checkpoint = _checkpoint_path(uvr_root, model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            free_bytes, _total = torch.cuda.mem_get_info()
            if free_bytes < 1024**3:  # below 1 GiB free: fall back to CPU
                device = "cpu"
        except (TypeError, ValueError, RuntimeError):
            device = "cpu"

    model_obj = BSRoformer(**config)
    state = torch.load(checkpoint, map_location="cpu")
    model_obj.load_state_dict(state)
    model_obj.to(device)
    model_obj.eval()

    audio, sample_rate = torchaudio.load(str(media_wav))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        # Chunked inference over the full waveform.
        length = audio.shape[1]
        chunk = _CHUNK_SIZE
        stems_audio: dict[str, torch.Tensor] = {
            name: torch.zeros_like(audio) for name in stems
        }
        for offset in range(0, length, chunk):
            piece = audio[:, offset : offset + chunk]
            piece = piece.to(device)
            separated = model_obj(piece.unsqueeze(0)).squeeze(0)
            # ``separated`` is expected to carry one channel per stem in the
            # order of ``stems``; map them positionally.
            for index, name in enumerate(stems):
                if index < separated.shape[0]:
                    stems_audio[name][:, offset : offset + chunk] += separated[index]

    for name in stems:
        torchaudio.save(
            str(out_dir / f"{name}.wav"), stems_audio[name].cpu(), sample_rate
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roformer_runner.py",
        description="Headless Roformer stem separation for media-mcp.",
    )
    parser.add_argument("media_wav", help="Input media WAV to separate.")
    parser.add_argument(
        "out_dir", help="Directory to write vocals.wav/instrumental.wav into."
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="Checkpoint name or path (default: %(default)s).",
    )
    parser.add_argument(
        "--stems",
        default="vocals,instrumental",
        help="Comma-separated stem list to produce (default: vocals,instrumental).",
    )
    args = parser.parse_args(argv)

    try:
        _separate(
            Path(args.media_wav),
            Path(args.out_dir),
            model=args.model,
            stems=tuple(s.strip() for s in args.stems.split(",") if s.strip()),
        )
    except Exception as err:  # noqa: BLE001 - report every failure via stderr
        print(f"roformer_runner: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
