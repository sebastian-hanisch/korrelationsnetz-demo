"""Rechnet das Raster der Messreihen vor (E1 bis E5) und schreibt data/network_grid.json.

    python tools/make_grid.py            # volles Raster, mehrere Prozesse
    python tools/make_grid.py --quick    # kleines Probe-Raster für Tests (nicht in data/network_grid.json)

Das Raster ist deterministisch (SplitMix64, feste Seeds); tests/test_grid.py rechnet eine feste Probe nach und vergleicht sie mit der Datei."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import knz_constants as C  # noqa: E402
import knz_evaluation as ev  # noqa: E402


def _run(task):
    group, args, kwargs = task
    return group, ev.net_cell(*args, **kwargs)


def main() -> None:
    quick = "--quick" in sys.argv
    tasks = [("e1", (t, C.E1_N, C.GRID_P, C.E1_RUNS), {}) for t in ("faktor", "gitter", "zufall", "kette")]                  # die teuren Zellen zuerst
    tasks += [("e2", (t, n, C.GRID_P, C.E2_RUNS), {}) for t in C.E2_TRUTHS for n in C.E2_NS]
    tasks += [("e3", (t, n, C.GRID_P, C.E3_RUNS), {}) for t in C.E2_TRUTHS for n in C.E3_NS]
    tasks += [("e5", ("faktor", n, C.GRID_P, C.E5_RUNS), {"with_path": False}) for n in C.E5_NS]
    if quick:
        tasks = [("e1", ("kette", 200, 12, 1), {"seeds": (7,)}), ("e2", ("gitter", 200, 15, 1), {"seeds": (7,)}), ("e3", ("kette", 20, 12, 1), {"seeds": (7,)}),
                 ("e5", ("faktor", 200, 12, 1), {"seeds": (7,), "with_path": False})]
    t0 = time.time()
    out = {"e1": [], "e2": [], "e3": [], "e5": []}
    with ProcessPoolExecutor(max_workers=10) as pool:
        for group, cell in pool.map(_run, tasks, chunksize=1):
            out[group].append(cell)
    grid = {"meta": {"seeds": list(C.GRID_SEEDS), "seconds": round(time.time() - t0)}, **out, "probe": ev.grid_probe()}
    dest = ROOT / "data" / ("network_grid_quick.json" if quick else "network_grid.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dest} geschrieben: " + ", ".join(f"{len(v)} {k}" for k, v in out.items()) + f", {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
