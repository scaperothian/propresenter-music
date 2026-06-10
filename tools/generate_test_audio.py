"""Generate synthetic test audio for smoke-testing the pipeline without a real recording.

Creates a 60-second WAV file at 24kHz where each 'slide' region contains a
distinct sine-wave frequency, making the slides trivially distinguishable by
any embedding model.

Usage:
    python tools/generate_test_audio.py --output data/test_song.wav --manifest data/test_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio


SLIDE_FREQ_HZ = [220, 330, 440, 550, 660, 770, 880, 990]  # A3, E4, A4, ...

SLIDE_NAMES = [
    "intro", "verse1", "chorus1", "verse2", "chorus2", "bridge", "chorus3", "outro"
]


def generate(
    output_wav: Path,
    output_manifest: Path,
    slide_duration_sec: float = 7.0,
    gap_sec: float = 0.5,
    sr: int = 24_000,
    amplitude: float = 0.3,
) -> None:
    n_slides = min(len(SLIDE_FREQ_HZ), len(SLIDE_NAMES))
    slides_out = []
    segments: list[np.ndarray] = []
    t_cursor = 0.0

    for i in range(n_slides):
        freq = SLIDE_FREQ_HZ[i]
        n_samples = int(slide_duration_sec * sr)
        t = np.arange(n_samples) / sr
        tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

        slides_out.append({"slide_id": SLIDE_NAMES[i], "t_ref": round(t_cursor, 3), "lyrics": SLIDE_NAMES[i]})
        segments.append(tone)

        gap_samples = int(gap_sec * sr)
        segments.append(np.zeros(gap_samples, dtype=np.float32))
        t_cursor += slide_duration_sec + gap_sec

    audio = np.concatenate(segments)
    wav_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, N]
    torchaudio.save(str(output_wav), wav_tensor, sr)
    print(f"Saved audio  ({len(audio)/sr:.1f}s) → {output_wav}")

    manifest = {
        "song_id": output_wav.stem,
        "ref_audio": output_wav.name,
        "slides": slides_out,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest ({n_slides} slides) → {output_manifest}")


def main():
    p = argparse.ArgumentParser(description="Generate synthetic test audio for ppsync.")
    p.add_argument("--output", default="data/test_song.wav", help="Output WAV path.")
    p.add_argument("--manifest", default="data/test_manifest.json", help="Output manifest path.")
    p.add_argument("--slide-dur", type=float, default=7.0, help="Duration of each slide section (seconds).")
    p.add_argument("--gap", type=float, default=0.5, help="Gap between slide sections (seconds).")
    args = p.parse_args()

    output_wav = Path(args.output)
    output_manifest = Path(args.manifest)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    generate(output_wav, output_manifest, args.slide_dur, args.gap)


if __name__ == "__main__":
    main()
