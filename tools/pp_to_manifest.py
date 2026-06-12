"""Convert a ProPresenter annotation JSON into a ppsync slide manifest.

The dataset files (e.g. studio_drive.json) store slides grouped by section,
where each slide carries a ``"trigger time"`` *array* — a slide can be shown at
several points in the song (a chorus repeats).  The ppsync pipeline, by
contrast, expects a flat, chronologically-ordered list of slide *instances*,
one per trigger event, with a single ``t_ref`` each (see io.load_manifest).

This adapter explodes the grouped timeline into that flat list:

    presentation.groups[].slides[].("trigger time": [t0, t1, ...])
        →  one manifest slide per (slide, ti), sorted by ti

So the two Chorus slides with three trigger times each become six chronological
instances interleaved with the verses, giving a strictly increasing t_ref
sequence the subsequence-DTW + left-to-right HMM can model.

ProPresenter annotations carry the song title but not the artist, so the
artist is supplied on the command line and written into the manifest.  The
default output lands in the per-song data directory
``data/<artist>/<title>/<artist>_<title>_manifest.json`` — every downstream
artifact (cache, benchmark results, logs) shares that directory and slug, so
files from different songs never collide and are identifiable at a glance.

Usage:
    python tools/pp_to_manifest.py studio_drive.json --artist "Incubus"
    # -> data/incubus/drive/incubus_drive_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ppsync.io import song_dir, song_slug


def _slug(text: str) -> str:
    """Short lowercase slug from a group name, e.g. 'Pre-Chorus 2' -> 'prechorus2'."""
    return re.sub(r"[^a-z0-9]+", "", text.lower()) or "slide"


def convert(
    pp_json_path: Path,
    artist: str,
    audio_override: str | None = None,
    title_override: str | None = None,
) -> dict:
    """Build a ppsync manifest dict from a ProPresenter annotation JSON."""
    pres = json.loads(pp_json_path.read_text())["presentation"]

    audio = audio_override or pres["id"]["audio"]

    # Flatten every (trigger_time, group, slide) into one event per trigger
    # time.  pp_idx is the slide's 0-based position in the presentation's
    # flattened groups[].slides[] order — the index ProPresenter's REST API
    # uses to trigger it.  Repeated trigger times (chorus) share one pp_idx.
    events: list[tuple[float, str, str, int]] = []
    pp_idx = 0
    for group in pres["groups"]:
        gname = group.get("name", "slide")
        for slide in group["slides"]:
            text = (slide.get("text", "") or "").strip()
            for t in slide.get("trigger time", []) or []:
                events.append((float(t), gname, text, pp_idx))
            pp_idx += 1

    if not events:
        raise ValueError(f"No trigger times found in {pp_json_path}")

    # Chronological order is the timeline order — groups are not stored sorted.
    events.sort(key=lambda e: e[0])

    slides = []
    for i, (t, gname, text, idx) in enumerate(events):
        slides.append(
            {
                "slide_id": f"{i:02d}_{_slug(gname)}",
                "t_ref": round(t, 3),
                "lyrics": text,
                "pp_slide_index": idx,
            }
        )

    return {
        "song_id": title_override or pres["id"].get("name", pp_json_path.stem),
        "artist": artist,
        "ref_audio": audio,
        "pp_uuid": pres["id"].get("uuid", ""),
        "slides": slides,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="ProPresenter JSON -> ppsync manifest.")
    p.add_argument("pp_json", help="ProPresenter annotation JSON (e.g. studio_drive.json).")
    p.add_argument(
        "--artist", required=True,
        help="Song artist (ProPresenter annotations don't store it) — becomes "
             "part of every downstream filename, e.g. --artist Incubus.",
    )
    p.add_argument(
        "--title", default=None,
        help="Override song title (default: presentation name).",
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="Output manifest path (default: "
             "<data-dir>/<artist>/<title>/<artist>_<title>_manifest.json).",
    )
    p.add_argument(
        "--data-dir", default="data",
        help="Base directory for the per-song data tree (default: data).",
    )
    p.add_argument(
        "--audio", default=None,
        help="Override reference audio path (default: presentation.id.audio).",
    )
    args = p.parse_args(argv)

    pp_path = Path(args.pp_json)
    manifest = convert(pp_path, artist=args.artist,
                       audio_override=args.audio, title_override=args.title)
    slug = song_slug(manifest["artist"], manifest["song_id"])
    if args.output:
        out = Path(args.output)
    else:
        out_dir = song_dir(manifest["artist"], manifest["song_id"], args.data_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{slug}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"Song: {manifest['artist']} — {manifest['song_id']}  [slug: {slug}]")
    print(f"Wrote {out}  ({len(manifest['slides'])} slide instances)")
    for s in manifest["slides"]:
        first_line = s["lyrics"].splitlines()[0] if s["lyrics"] else ""
        print(f"  {s['slide_id']:16s} {s['t_ref']:7.2f}s  {first_line[:42]}")


if __name__ == "__main__":
    main()
