"""Closed-loop ProPresenter trigger test.

Steps through every distinct ProPresenter slide referenced by a ppsync
manifest, triggering each via propresenter-client's go_to_slide() and reading
back /v1/presentation/slide_index to verify the commanded slide is actually
showing.  Restores the initially active slide at the end.

This exercises the exact call path ppsync's TriggerScheduler uses live.

Usage:
    python tools/pp_trigger_test.py data/studio_manifest.json \
        [--host localhost] [--port 1025] [--settle 0.6]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from propresenter_client.main import ProPresenterController


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop ProPresenter trigger test.")
    p.add_argument("manifest", help="ppsync manifest with pp_slide_index/pp_uuid.")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1025)
    p.add_argument("--settle", type=float, default=0.6,
                   help="Seconds to wait before reading back the slide index "
                        "(the API lags the trigger slightly).")
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    pp_uuid = manifest.get("pp_uuid", "")
    slides = manifest["slides"]

    pro = ProPresenterController(host=args.host, port=args.port)
    if pro.get_status() is None:
        sys.exit(f"Cannot reach ProPresenter at {args.host}:{args.port}")

    active = pro.get_active_presentation_uuid()
    print(f"Active presentation: {active}")
    if pp_uuid and active != pp_uuid:
        print(f"Activating {pp_uuid} (manifest presentation)…")
        if not pro.activate_presentation(pp_uuid):
            sys.exit("Could not activate the manifest's presentation.")
        time.sleep(1.0)

    initial = pro.get_slide_index()
    print(f"Initial slide index: {initial}\n")

    # Every distinct pp slide, in manifest order, labelled by first instance.
    seen: dict[int, str] = {}
    for s in slides:
        seen.setdefault(int(s["pp_slide_index"]), s["slide_id"])

    failures = 0
    for pp_idx, label in seen.items():
        pro.go_to_slide(pp_idx + 1)   # 1-indexed client API
        time.sleep(args.settle)
        shown = pro.get_slide_index()
        ok = shown == pp_idx
        failures += 0 if ok else 1
        print(f"  go_to_slide({pp_idx + 1:2d})  [{label:16s}]  "
              f"→ showing index {shown}  {'OK' if ok else '*** MISMATCH ***'}")

    if initial is not None:
        pro.go_to_slide(initial + 1)
        print(f"\nRestored initial slide index {initial}.")

    print(f"\n{len(seen) - failures}/{len(seen)} slides verified"
          + ("" if failures == 0 else f"  ({failures} FAILED)"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
