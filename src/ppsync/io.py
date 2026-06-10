"""Slide manifest loading and audio ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio

from .config import TARGET_SR


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
