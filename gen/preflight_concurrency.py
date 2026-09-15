#!/usr/bin/env python3
"""Refuse a launch whose CONCURRENT tool servers would not fit in memory.

Why this exists, and it is not hypothetical. A training run was launched with `max_inflight = 128` and
`patients_per_population = 16`. Both numbers had been checked -- the TRACE arithmetic was right
(M*N = 128 traces per group, 512 patient answers per step, identical generation cost to the preceding
one-patient-per-episode run) -- but
`batch_size` counts TRACES while `max_inflight` counts EPISODES, and M multiplies the second one only.
One episode stopped being one patient and became 16 concurrent patient answers, each of which spawns its
own `cohortworld_env.toolset` process. Observed on the node: 381 live tool servers and MemAvailable down
to 19 GiB of 2015, on a machine shared with another user. The run was killed before it OOMed.

The check is one multiplication. It is here because the multiplication was not made, not because it is
hard: two numbers that were each individually correct were combined by a factor nobody wrote down.

Everything is an argument with a DERIVED default -- available memory is read from /proc/meminfo, not
restated -- except `--per-server-gib`, whose default is a MEASUREMENT recorded in the docstring of
`MEASURED_PER_SERVER_GIB`. The mean is used rather than the median because the tail is what exhausts a
node.
"""
from __future__ import annotations

import argparse
import sys
import tomllib

MEASURED_PER_SERVER_GIB = 1.30
"""A training run at STEADY STATE: 1026 concurrent tool servers holding 1167 GiB --
mean 1.14, median 0.49, p90 2.42, max 2.42. 1.30 is the MEAN rounded up.

USE THE MEAN, NOT THE MEDIAN. The distribution is bimodal: most servers sit at the bare lattice
(0.49 GiB) and a long tail reaches 2.42 as a rollout accumulates the cells it touched. A total is
n x mean, so the median understates it by 2.3x -- which is precisely the mistake below.

THIS NUMBER HAS NOW BEEN WRONG THREE TIMES, each in a different way, which is why it is measured at
STEADY STATE and not derived, sampled early, or reasoned about:
  2.63  measured on a one-patient-per-episode run (120 live servers, 315.6 GiB, median 1.01, max
        6.81) and carried unchanged into the M = 16 launch. 63% LOW: that run cost 4.29. The check told that
        launch it needed 337 GiB when it took 549, and the run had to be killed at 19 GiB available.
  2.20  DERIVED as 4.29 - 2.40 + 0.29 after the granular-fields change. 4.7x HIGH: it assumed the
        non-states footprint was private per process when most of it is shared pages.
  0.60  MEASURED, but at 49 servers during ramp-up and effectively the median. 2.2x LOW: at 1026
        servers the mean is 1.14, and max_inflight 64 took MemAvailable to 161 GiB before the run was
        stopped ahead of its own watchdog.
Measure with >= a few hundred servers alive and use the mean."""


def available_gib() -> float:
    """Read it, do not restate it."""
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    raise RuntimeError("/proc/meminfo has no MemAvailable")


def concurrent_answers(cfg: dict) -> tuple[int, int, int]:
    """(max_inflight, M, concurrent patient answers). M defaults to 1: an env that does not fan out
    answers one patient per episode."""
    conc = (cfg.get("orchestrator") or {}).get("concurrency") or {}
    inflight = int(conc.get("max_inflight") or 0)
    m = 1
    for src in ((cfg.get("orchestrator") or {}).get("train") or {}).get("source") or []:
        task = (((src.get("env") or {}).get("taskset") or {}).get("task")) or {}
        m = max(m, int(task.get("patients_per_population") or 1))
    return inflight, m, inflight * m


def check(cfg: dict, *, avail_gib: float, per_server_gib: float, reserve_gib: float) -> tuple[bool, str]:
    inflight, m, answers = concurrent_answers(cfg)
    if not inflight:
        return True, "no max_inflight set; the orchestrator's own default applies and this check abstains"
    need = answers * per_server_gib
    budget = avail_gib - reserve_gib
    ok = need <= budget
    verdict = "OK" if ok else "REFUSE"
    return ok, (
        f"{verdict}: max_inflight {inflight} x patients_per_population {m} = {answers} concurrent "
        f"patient answers, each spawning a tool server at ~{per_server_gib} GiB = {need:.0f} GiB needed; "
        f"{avail_gib:.0f} GiB available minus {reserve_gib:.0f} GiB reserved for other users "
        f"= {budget:.0f} GiB budget."
        + ("" if ok else f" Set max_inflight to {int(budget // (per_server_gib * m))} or lower, "
                         f"or reduce patients_per_population.")
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("toml", help="the RL config about to be launched")
    ap.add_argument("--per-server-gib", type=float, default=MEASURED_PER_SERVER_GIB)
    ap.add_argument("--reserve-gib", type=float, default=400.0,
                    help="left for other users of a shared node (default 400)")
    ap.add_argument("--available-gib", type=float, default=None,
                    help="override the reading from /proc/meminfo, for tests")
    a = ap.parse_args(argv)
    cfg = tomllib.load(open(a.toml, "rb"))
    avail = a.available_gib if a.available_gib is not None else available_gib()
    ok, msg = check(cfg, avail_gib=avail, per_server_gib=a.per_server_gib, reserve_gib=a.reserve_gib)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
