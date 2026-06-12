"""
bench/probe_ceiling.py — Find the real unified-memory ceiling on this machine.

The model ladder never OOM'd (a 123B model used only ~69GB of 128GB), so the
true breaking point is above any model we ran. This probe finds it directly:
it allocates GPU memory in chunks until Metal refuses, and reports the wall.

SAFETY: a hard cap (default 125GB) stops the probe before it can push the
machine into a hard hang. Run it on an otherwise-idle machine (NOT while a
benchmark is running) and with someone watching.

Usage:
    uv run python bench/probe_ceiling.py
    uv run python bench/probe_ceiling.py --chunk-gb 4 --cap-gb 125
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

RESULTS_DIR = Path(__file__).parent.parent / "results"


def gb(x_bytes: float) -> float:
    return x_bytes / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-gb", type=float, default=4.0, help="allocation chunk size (GB)")
    ap.add_argument("--cap-gb", type=float, default=125.0, help="hard safety cap (GB active)")
    args = ap.parse_args()

    info = mx.device_info()
    advertised = info.get("max_recommended_working_set_size", 0)
    total = info.get("memory_size", 0)
    max_buf = info.get("max_buffer_length", 0)

    print(f"Device: {info.get('device_name')}")
    print(f"  total unified memory          : {gb(total):.1f} GB")
    print(f"  Metal recommended working set : {gb(advertised):.1f} GB  <- advertised ceiling")
    print(f"  max single buffer             : {gb(max_buf):.1f} GB")
    print(f"  probe chunk size              : {args.chunk_gb} GB")
    print(f"  hard safety cap               : {args.cap_gb} GB")
    print(f"\nAllocating until Metal refuses (or cap)...\n")

    # float16 = 2 bytes/elem
    elems_per_chunk = int(args.chunk_gb * 1e9 / 2)
    held = []  # keep references so buffers aren't freed mid-probe
    failed_at = None
    fail_reason = None

    try:
        while True:
            active_gb = gb(mx.get_active_memory())
            if active_gb >= args.cap_gb:
                print(f"  reached safety cap at {active_gb:.1f} GB active — stopping cleanly.")
                break
            try:
                a = mx.random.normal((elems_per_chunk,), dtype=mx.float16)
                mx.eval(a)               # force real allocation
                held.append(a)
                now = gb(mx.get_active_memory())
                print(f"  allocated +{args.chunk_gb:g} GB  ->  {now:6.1f} GB active")
            except Exception as e:        # noqa: BLE001 - we want any allocation failure
                failed_at = gb(mx.get_active_memory())
                fail_reason = f"{type(e).__name__}: {str(e)[:120]}"
                print(f"\n  *** Metal refused allocation at {failed_at:.1f} GB active ***")
                print(f"  reason: {fail_reason}")
                break
    finally:
        held.clear()
        mx.clear_cache()

    ceiling_gb = failed_at if failed_at else gb(mx.get_active_memory())
    result = {
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "device_name": info.get("device_name"),
        "total_memory_gb": round(gb(total), 1),
        "advertised_working_set_gb": round(gb(advertised), 1),
        "max_single_buffer_gb": round(gb(max_buf), 1),
        "empirical_ceiling_gb": round(ceiling_gb, 1) if failed_at else None,
        "hit_safety_cap": failed_at is None,
        "fail_reason": fail_reason,
    }
    out = RESULTS_DIR / "ceiling-probe.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nResult → {out}")
    if failed_at:
        print(f"Empirical ceiling: ~{failed_at:.0f} GB "
              f"({failed_at/gb(total)*100:.0f}% of {gb(total):.0f}GB physical)")
    else:
        print(f"No failure before the {args.cap_gb}GB safety cap — "
              f"ceiling is at/above that (advertised {gb(advertised):.0f}GB working set).")


if __name__ == "__main__":
    main()
