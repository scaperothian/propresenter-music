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
# song_slug / load_song_meta
# ---------------------------------------------------------------------------

def test_song_slug_artist_and_title():
    from ppsync.io import song_slug

    assert song_slug("Incubus", "Drive") == "incubus_drive"
    assert song_slug("Forrest Frank", "Good Day") == "forrest_frank_good_day"


def test_song_slug_normalizes_punctuation():
    from ppsync.io import song_slug

    assert song_slug("AC/DC", "T.N.T.") == "ac_dc_t_n_t"
    assert song_slug(" Sigur Rós ", "Hoppípolla!") == "sigur_r_s_hopp_polla"


def test_song_slug_missing_parts():
    from ppsync.io import song_slug

    assert song_slug("", "Drive") == "drive"      # no artist → title alone
    assert song_slug("", "") == "unknown_song"


def test_song_dir_hierarchy():
    from ppsync.io import song_dir

    assert song_dir("Incubus", "Drive") == Path("data/incubus/drive")
    assert song_dir("Forrest Frank", "Good Day", base="/x") == \
        Path("/x/forrest_frank/good_day")
    assert song_dir("", "") == Path("data/unknown_artist/unknown_song")


def test_load_song_meta(tmp_path):
    from ppsync.io import load_song_meta

    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "song_id": "Drive", "artist": "Incubus",
        "ref_audio": "x.wav", "pp_uuid": "ABC", "slides": [],
    }))
    meta = load_song_meta(path)
    assert meta["song_id"] == "Drive"
    assert meta["artist"] == "Incubus"
    assert meta["slug"] == "incubus_drive"
    assert meta["pp_uuid"] == "ABC"


def test_load_song_meta_defaults(tmp_path):
    from ppsync.io import load_song_meta

    path = tmp_path / "old_manifest.json"
    path.write_text(json.dumps({"ref_audio": "x.wav", "slides": []}))
    meta = load_song_meta(path)
    assert meta["song_id"] == "old_manifest"  # falls back to file stem
    assert meta["artist"] == ""
    assert meta["slug"] == "old_manifest"


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
