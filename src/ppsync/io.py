"""Slide manifest loading and audio ingestion."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torchaudio

from .config import TARGET_SR


def _slug_part(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def song_slug(artist: str, title: str) -> str:
    """
    Filesystem slug identifying a song by artist AND title, e.g.
    ``('Incubus', 'Drive') -> 'incubus_drive'``.

    Every per-song artifact (manifest, embedding cache, benchmark results,
    telemetry logs) is named with this slug so artifacts from different songs
    never collide and are identifiable from the filename alone.
    """
    parts = [p for p in (_slug_part(artist), _slug_part(title)) if p]
    return "_".join(parts) or "unknown_song"


def song_dir(artist: str, title: str, base: str | Path = "data") -> Path:
    """
    Directory holding all of one song's artifacts:
    ``<base>/<artist>/<title>`` — e.g. ``data/incubus/drive``.

    The data tree is one directory per artist, one subdirectory per song;
    manifests, caches, and benchmark results for the song all live there.
    """
    artist_part = _slug_part(artist) or "unknown_artist"
    title_part = _slug_part(title) or "unknown_song"
    return Path(base) / artist_part / title_part


def load_song_meta(json_path: Path) -> dict:
    """
    Read song identity from a manifest JSON without loading slides/audio.

    Returns dict with keys: ``song_id`` (title), ``artist``, ``slug``
    (``song_slug(artist, song_id)``), ``pp_uuid``.  Missing fields default to
    empty strings; the slug falls back to the title alone (or the file stem).
    """
    with open(json_path) as f:
        data = json.load(f)
    song_id = str(data.get("song_id", "") or Path(json_path).stem)
    artist = str(data.get("artist", ""))
    return {
        "song_id": song_id,
        "artist": artist,
        "slug": song_slug(artist, song_id),
        "pp_uuid": str(data.get("pp_uuid", "")),
    }


def load_manifest(json_path: Path) -> tuple[Path, list[dict]]:
    """
    Load a slide annotation manifest.

    Expected JSON format::

        {
          "song_id": "amazing_grace",
          "ref_audio": "amazing_grace.wav",
          "slides": [
            {"slide_id": "intro",   "t_ref": 0.0,  "lyrics": "..."},
            {"slide_id": "verse1",  "t_ref": 12.5, "lyrics": "..."},
            ...
          ]
        }

    The ``t_ref`` value is the TRUE musical timestamp (in seconds, matching the
    reference audio) when the slide should advance.  The stop time for each slide
    is inferred as the start of the next slide; the final slide extends to the
    end of the song (filled in after audio duration is known — set to ``None``
    initially so callers can populate it after loading audio).

    The ``ref_audio`` path is resolved relative to the JSON file location if it
    is not absolute.

    Returns:
        audio_path: resolved path to the reference audio file
        slides:     list of dicts — one per slide in chronological order — with
                    keys: slide_id (str), t_ref (float), t_stop (float | None),
                    lyrics (str)
    """
    with open(json_path) as f:
        data = json.load(f)

    raw_audio = data.get("ref_audio", "")
    audio_path = Path(raw_audio)
    if not audio_path.is_absolute():
        audio_path = json_path.parent / audio_path

    raw_slides = data.get("slides", [])
    if not raw_slides:
        raise ValueError(f"No slides found in {json_path}")

    slides: list[dict] = []
    for i, s in enumerate(raw_slides):
        t_ref = float(s["t_ref"])
        t_stop = float(raw_slides[i + 1]["t_ref"]) if i + 1 < len(raw_slides) else None
        slides.append(
            {
                "slide_id": str(s.get("slide_id", f"slide_{i}")),
                "t_ref": t_ref,
                "t_stop": t_stop,
                "lyrics": s.get("lyrics", ""),
                # Slide's 0-based position in the ProPresenter presentation
                # (repeated sections share one); defaults to chronological
                # order for manifests without ProPresenter metadata.
                "pp_slide_index": int(s.get("pp_slide_index", i)),
            }
        )

    return audio_path, slides


def finalize_slide_stops(slides: list[dict], song_duration: float) -> None:
    """Fill in t_stop for the final slide using the song duration (in-place)."""
    if slides and slides[-1]["t_stop"] is None:
        slides[-1]["t_stop"] = song_duration


def load_audio(audio_path: Path) -> torch.Tensor:
    """
    Load an audio file, downmix to mono, and resample to TARGET_SR.

    Returns:
        1-D float tensor of shape [num_samples]
    """
    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0)
