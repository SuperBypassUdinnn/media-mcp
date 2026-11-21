"""Headless Roformer stem separation runner for the media-mcp vocal ladder.

Executed as a subprocess under the Ultimate Vocal Remover venv interpreter
(torch 2.12+cu130 at ``<UVR_ROOT>/.venv``), pointed at by
``MEDIA_SEPARATE_BIN`` (mirroring ``MEDIA_RHYTHM_BIN``). Usage::

    roformer_runner.py <media.wav> <out_dir> [--model <name>] [--stems vocals,instrumental]

Audio IO uses ``soundfile`` (not torchaudio): ``sf.read`` returns a
(samples, channels) float32 array, converted to a channels-first torch
tensor with ``torch.from_numpy(audio).T.float()``; stems are written back
with ``sf.write``. The input media is decoded to a temp stereo 44100 Hz WAV
with ffmpeg (the same decode block as ``rhythm/onsets.py``) because the
server passes the raw source path, which may be mp4/mkv/mp3 rather than WAV.

The ``roformer_native`` model definitions are imported from the UVR repo
(``UVR_ROOT/lib_v5``). ``UVR_ROOT`` resolution order:
  1. the ``UVR_ROOT`` environment variable,
  2. the first sorted glob hit of
     ``$HOME/.clone/Github/*/ultimatevocalremovergui``,
  3. the legacy default: a sibling of this repository.
A ``FileNotFoundError`` listing every searched location is raised when nothing
resolves.

The model config yaml is loaded from
``models/MDX_Net_Models/model_data/mdx_c_configs/*.yaml`` and the checkpoint
from ``models/MDX_Net_Models/`` (falling back to ``checkpoints/`` and the
repo root). Runs chunked inference (``chunk_size: 588800``), checks free VRAM
before GPU use, falls back to CPU below 1 GiB free and on CUDA out-of-memory,
and writes ``<out_dir>/vocals.wav`` plus ``<out_dir>/instrumental.wav``. When
the checkpoint emits a single vocal stem (``num_stems: 1``), instrumental is
derived as mix minus vocals.

This module is import-safe without torch: all heavy imports (torch,
soundfile, ``roformer_native``) are lazy inside ``_separate`` or under
``if __name__ == "__main__":``, so media/.venv (no torch) can import this
module and read its docstring only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# UVR repo root resolution order:
#   1. UVR_ROOT environment variable
#   2. first sorted glob hit under $HOME/.clone/Github/*/ultimatevocalremovergui
#   3. legacy default: a sibling of this repository
_DEFAULT_UVR_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "ultimatevocalremovergui"
)

# Default checkpoint the brief names as the community-reference vocals extractor.
_DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

# Chunked-inference geometry from the UVR config yaml family.
_CHUNK_SIZE = 588800
_DIM_T = 801


def _uvr_root() -> Path:
    """Resolve the UVR repository root.

    Resolution order: ``UVR_ROOT`` env var → first sorted glob hit of
    ``$HOME/.clone/Github/*/ultimatevocalremovergui`` → the legacy default
    sibling of this repository. Raises ``FileNotFoundError`` listing every
    location searched when nothing resolves.
    """
    searched: list[str] = []

    env_root = os.environ.get("UVR_ROOT")
    if env_root:
        searched.append(f"UVR_ROOT={env_root!r}")
        candidate = Path(env_root)
        if candidate.is_dir():
            return candidate

    for candidate in sorted(
        Path.home().glob(".clone/Github/*/ultimatevocalremovergui")
    ):
        searched.append(f"glob={candidate}")
        if candidate.is_dir():
            return candidate

    searched.append(f"default={_DEFAULT_UVR_ROOT}")
    if _DEFAULT_UVR_ROOT.is_dir():
        return _DEFAULT_UVR_ROOT

    raise FileNotFoundError(
        "UVR repository root could not be resolved; searched locations: "
        + "; ".join(searched)
    )


def _config_dir(uvr_root: Path) -> Path:
    return uvr_root / "models" / "MDX_Net_Models" / "model_data" / "mdx_c_configs"


def _checkpoint_path(uvr_root: Path, model: str) -> Path:
    """Resolve a checkpoint: bare name looks under UVR checkpoints/model dirs."""
    path = Path(model)
    if path.is_file():
        return path
    candidates = [
        uvr_root / "checkpoints" / model,
        uvr_root / "models" / "MDX_Net_Models" / model,
        uvr_root / model,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path


def _load_config(config_dir: Path, model: str) -> dict:
    """Load the config yaml matching ``model`` from the config dir.

    UVR configs use ``!!python/tuple`` tags, which ``SafeLoader`` does not
    construct by default, so a small loader subclass handles them without
    resorting to the unsafe full loader.
    """
    import yaml
    from yaml import SafeLoader

    class _TupleSafeLoader(SafeLoader):
        pass

    _TupleSafeLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: tuple(loader.construct_sequence(node)),
    )

    # Strip extension and try `<stem>.yaml`, then `<stem>_config.yaml`.
    stem = Path(model).stem
    for name in (f"{stem}.yaml", f"{stem}_config.yaml", f"{stem}.yml"):
        candidate = config_dir / name
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return yaml.load(handle, Loader=_TupleSafeLoader)
    raise FileNotFoundError(f"no mdx_c config yaml for model {model!r} in {config_dir}")


def _decode_to_wav(media_wav: Path, tmp_dir: Path) -> Path:
    """Decode `media_wav` to a stereo 44100 Hz PCM WAV via ffmpeg.

    Mirrors the decode block in ``rhythm/onsets.py``: the server passes the
    raw source path (possibly mp4/mkv/mp3), so every input goes through one
    ffmpeg pass into a temp WAV that soundfile can read. Matches the roformer
    config's ``sample_rate: 44100`` and the model's stereo input.
    """
    wav = tmp_dir / "input.wav"
    done = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(media_wav),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0 or not wav.is_file():
        raise RuntimeError(
            "ffmpeg decode of "
            f"{media_wav} failed: "
            f"{(done.stderr or '').strip() or 'no output file'}"
        )
    return wav


def _separate(
    media_wav: Path,
    out_dir: Path,
    *,
    model: str = _DEFAULT_MODEL,
    stems: tuple[str, ...] = ("vocals", "instrumental"),
) -> None:
    """Run Roformer separation and write the requested stems.

    Heavy imports (torch, soundfile, roformer_native) are local to this
    function so the module stays import-safe without torch.
    """
    import soundfile as sf
    import torch

    uvr_root = _uvr_root()
    config_dir = _config_dir(uvr_root)
    config = _load_config(config_dir, model)
    checkpoint = _checkpoint_path(uvr_root, model)

    # roformer_native lives under UVR_ROOT/lib_v5, not on the default path.
    lib_v5 = uvr_root / "lib_v5"
    if lib_v5.is_dir() and str(lib_v5) not in sys.path:
        sys.path.insert(0, str(lib_v5))
    from roformer_native.bs_roformer import BSRoformer

    model_cfg = dict(config.get("model", {}))
    stereo = bool(model_cfg.get("stereo", False))
    num_stems = int(model_cfg.get("num_stems", len(stems)))
    if num_stems > len(stems):
        print(
            "roformer_runner: warning: checkpoint num_stems="
            f"{num_stems} exceeds requested stems {stems}; dropping "
            f"checkpoint output stems at indices "
            f"{list(range(len(stems), num_stems))}",
            file=sys.stderr,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            free_bytes, _total = torch.cuda.mem_get_info()
            if free_bytes < 1024**3:  # below 1 GiB free: fall back to CPU
                device = "cpu"
        except (TypeError, ValueError, RuntimeError):
            device = "cpu"

    model_obj = BSRoformer(**model_cfg)
    state = torch.load(checkpoint, map_location="cpu")
    model_obj.load_state_dict(state)
    model_obj.to(device)
    model_obj.eval()

    # Decode the input media to a stereo 44100 Hz WAV, then load via soundfile.
    # sf.read returns (samples, channels); convert to a channels-first tensor.
    with tempfile.TemporaryDirectory(prefix="roformer-in-") as tmp:
        wav_path = _decode_to_wav(media_wav, Path(tmp))
        audio_np, sample_rate = sf.read(str(wav_path), dtype="float32")
    audio = torch.from_numpy(audio_np.copy()).T.float()  # (channels, samples)
    if stereo and audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif not stereo and audio.shape[0] == 2:
        audio = audio.mean(dim=0, keepdim=True)

    mix = audio
    length = audio.shape[1]
    chunk = _CHUNK_SIZE
    produced: dict[str, torch.Tensor] = {}

    def _run(dev: str) -> None:
        with torch.inference_mode():
            for offset in range(0, length, chunk):
                piece = audio[:, offset : offset + chunk].to(dev)
                separated = model_obj(piece.unsqueeze(0)).squeeze(0)
                # ``separated`` is (num_stems, channels, chunk), one tensor per
                # stem in model output order; map them positionally to ``stems``.
                # Clamp to len(stems) so a checkpoint config whose num_stems
                # exceeds the requested stem list never indexes past ``stems``.
                for index in range(min(num_stems, len(stems), separated.shape[0])):
                    name = stems[index]
                    buf = produced.setdefault(name, torch.zeros_like(audio))
                    buf[:, offset : offset + chunk] += separated[index].cpu()

    try:
        _run(device)
    except torch.cuda.OutOfMemoryError:
        if device != "cuda":
            raise
        # CUDA out of memory: fall back to CPU and retry the full inference.
        torch.cuda.empty_cache()
        model_obj.to("cpu")
        device = "cpu"
        produced = {}
        _run("cpu")

    # Some checkpoints emit a single vocal stem (num_stems: 1): derive
    # instrumental as mix minus vocals instead of leaving silence.
    if "instrumental" in stems and "instrumental" not in produced:
        vocal = produced.get("vocals")
        produced["instrumental"] = (
            (mix - vocal) if vocal is not None else torch.zeros_like(mix)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in stems:
        data = produced.get(name, torch.zeros_like(mix))
        sf.write(
            str(out_dir / f"{name}.wav"),
            data.cpu().T.contiguous().numpy(),
            sample_rate,
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
