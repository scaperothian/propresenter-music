"""Tests for manifest loading and audio ingestion."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_manifest(tmp_path):
    """Write a minimal manifest JSON and a matching silent WAV."""
    import torchaudio

    wav_path = tmp_path / "song.wav"
    # 10 seconds of silence at 24kHz mono
    silent = torch.zeros(1, 24_000 * 10)
    torchaudio.save(str(wav_path), silent, 24_000)

    manifest = {
        "song_id": "test_song",
        "ref_audio": "song.wav",
        "slides": [
            {"slide_id": "intro",  "t_ref": 0.0,  "lyrics": "Hello"},
            {"slide_id": "verse1", "t_ref": 3.0,  "lyrics": "World"},
            {"slide_id": "chorus", "t_ref": 6.0,  "lyrics": "Chorus"},
        ],
    }
    json_path = tmp_path / "song.json"
    json_path.write_text(json.dumps(manifest))
    return json_path, wav_path


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

def test_load_manifest_returns_slides(tmp_manifest):
    from ppsync.io import load_manifest

    json_path, wav_path = tmp_manifest
    audio_path, slides = load_manifest(json_path)

    assert audio_path == wav_path
    assert len(slides) == 3
    assert slides[0]["slide_id"] == "intro"
    assert slides[0]["t_ref"] == 0.0
    assert slides[1]["t_stop"] == pytest.approx(6.0)
    assert slides[-1]["t_stop"] is None  # final slide: not yet filled


def test_load_manifest_stop_inferred(tmp_manifest):
    from ppsync.io import load_manifest

    json_path, _ = tmp_manifest
    _, slides = load_manifest(json_path)

    assert slides[0]["t_stop"] == pytest.approx(3.0)
    assert slides[1]["t_stop"] == pytest.approx(6.0)


def test_finalize_slide_stops(tmp_manifest):
    from ppsync.io import finalize_slide_stops, load_manifest

    json_path, _ = tmp_manifest
    _, slides = load_manifest(json_path)
    finalize_slide_stops(slides, 10.0)
    assert slides[-1]["t_stop"] == pytest.approx(10.0)


def test_load_manifest_missing_slides_raises(tmp_path):
    from ppsync.io import load_manifest

    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"song_id": "x", "ref_audio": "x.wav", "slides": []}))
    with pytest.raises(ValueError, match="No slides"):
        load_manifest(path)


# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------

def test_load_audio_shape_and_rate(tmp_path):
    import torchaudio
    from ppsync.io import load_audio

    wav_path = tmp_path / "tone.wav"
    data = torch.zeros(1, 48_000)  # 1s at 48kHz
    torchaudio.save(str(wav_path), data, 48_000)

    wav = load_audio(wav_path)
    assert wav.dim() == 1
    assert wav.shape[0] == 24_000  # resampled to 24kHz


def test_load_audio_stereo_downmix(tmp_path):
    import torchaudio
    from ppsync.io import load_audio

    wav_path = tmp_path / "stereo.wav"
    data = torch.randn(2, 24_000)
    torchaudio.save(str(wav_path), data, 24_000)

    wav = load_audio(wav_path)
    assert wav.dim() == 1  # mono
