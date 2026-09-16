#!/usr/bin/env python
"""T2 under the item-rendering spirit: the rendering, the age conflict, and the zoom dial.

The spirit: an item is its target cohort's own released description, age band de-grouped, degraded
only by dials the agent can invert (A: mask an axis; B: zoom, i.e. coarsen a code).

CPU only. Reads the frozen environment and the pooled edge table; edits neither. Reuses
`build/t2_ceiling.py` for every policy and every reward (so the ladder here IS the published ladder)
and an answerable-supply scan.

Four questions, in the order the spirit forces them:

  §1 RENDER   A T2 item under the spirit is the FROM cohort's own released description with the age
              band de-grouped, dials A (mask) and B (zoom) applied, plus the operation clause. The
              renderer here is string surgery on the lattice's OWN `prompt_body`, which is the
              k-anonymous release -- so "carries nothing the release does not" is true by
              construction rather than by review.

  §2 CONFLICT `age_band shift` is 79/118 of the unseen items, and age_band is the one axis the
              spirit always rewrites to `about xx`. Three sketched options are BUILT as three item
              sets and scored on the same ladder:
                  a) non-age only          b) age kept, band-rendered (the shipped mix)
                  c) age capped at one stratum's share
              The decisive column is the best NO-READ policy's accuracy. The shipped set's number
              is 0.7881 on unseen. If a variant does not move it, the spirit did not fix T2's leak.

  §3 ZOOM     T2's answer is a rate DIFFERENCE across one edge, and `coarseness` is part of a
              state's identity -- so a from-cohort rendered at a different granularity is a
              DIFFERENT cohort with a DIFFERENT edge. Whether the direction survives the zoom is
              measured (`--zoom-sample` from-states), not argued.

  §4 DIAL A   Masking an axis of the COHORT is the spirit's dial. Masking the DESTINATION (the
              operation's `code`) is not the same move: it makes several edges consistent with the
              item, and the label is well posed only if they agree. Measured on the same sample.

Run:  .venv/bin/python build/t2_spirit.py --selfcheck
      .venv/bin/python build/t2_spirit.py --out <rundir>
"""
import argparse, collections, json, os, random, statistics, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
import t2_ceiling as T                                     # noqa: E402
from cohortworld_env.lattice import _shared, PANEL         # noqa: E402

EDGES = f"{PANEL}/CohortWorld_pooled_edges.jsonl"
# version <-> outcome granularity is a BIJECTION, asserted in --selfcheck: C1's rate table holds
# exactly the 19 g1 codes, C2's the 183 g2 codes, default's the 1374 g3 codes, with zero overlap.
# So the server's disease clause ("the asked code is in THIS version's table") is not a filter that
# happens to bite -- it is `g == GRAN[version]`, and two thirds of every edge row's delta payload is
# unreadable by construction. That single fact is why the answerable supply below is so thin.
GRAN = {"C1": 1, "C2": 2, "default": 3}
SPLITS = ("train", "seen", "unseen")


def split_of(sid):
    """The item builder's `split_of` with the default empty salt. Copied, not imported: the item
    builder is not a library and importing it runs its module-level panel load. --selfcheck asserts
    it reproduces the split of all 625 shipped T2 items."""
    import hashlib
    h = int(hashlib.md5(f"cwsplit2:{sid}".encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("seen" if h < 90 else "unseen")

BOOT, SEED = T.BOOT, T.SEED
ZOOM_OPS = ("zoom_coarser", "zoom_finer")
# The number the spirit has to beat, measured by `build/t2_ceiling.py` on the shipped unseen split (n=118).
TODAY_NOREAD_UNSEEN = 0.7881


# --------------------------------------------------------------------------- §1 the renderer

def _degroup(band):
    """`65-69` -> `about 67`. The ONLY transformation the spirit allows, and it is lossy in the
    safe direction: `about 67` is vaguer than the band, never sharper."""
    lo, _, hi = str(band).partition("-")
    try:
        return f"about {(int(lo) + int(hi)) // 2}"
    except ValueError:
        return f"about {band}"


def render(state, mask=()):
    """The item's cohort clause: the released `prompt_body`, age de-grouped, masked axes dropped.

    String surgery on the release itself rather than a re-render from `axes`, because a re-render
    is a second renderer that can drift into stating something the release does not. `mask` names
    axes from {age_band, sex, icd, opcs, bnf}; a masked axis leaves NO trace, so the agent cannot
    tell a masked axis from an empty one -- that is what makes `add`/`switch` the inverse of it.
    """
    body = state["prompt_body"]
    head, _, rest = body.partition("\n")
    band = str(state["axes"]["age_band"])
    head = head.replace(f"aged {band}", "" if "age_band" in mask else _degroup(band))
    sex = str(state["axes"]["sex"]).lower()
    if "sex" in mask:
        head = head.replace(f"({sex}; ", "(").replace(f"{sex}; ", "")
    keep = {"diagnoses": "icd", "procedures": "opcs", "prescriptions": "bnf"}
    lines = [l for l in rest.splitlines()
             if not (l.startswith("- ") and keep.get(l[2:].split(":", 1)[0], "") in mask)]
    return "\n".join([" ".join(head.split())] + lines)


def op_clause(axis, op, code, show_code=True):
    """The operation the second cohort differs by. `show_code=False` is dial A applied to the
    DESTINATION rather than to the cohort -- §4 measures whether the label survives it."""
    w = {"icd": "diagnosis", "opcs": "procedure", "bnf": "prescription"}.get(axis, axis)
    if axis == "age_band":
        return ("the second is in a different age band" if not show_code else
                f"the second is {_degroup(code)} instead")
    if axis == "sex":
        return f"the second is {str(code).lower()} and the first is not"
    if op in ZOOM_OPS:
        return (f"the second's {w} history is recorded at a "
                f"{'finer' if op.endswith('finer') else 'coarser'} level of detail")
    verb = {"add": "also has", "switch": "has, in place of one of these,"}.get(op, "differs by")
    return f"the second {verb} {code if show_code else f'a different {w}'}"


# --------------------------------------------------------------------------- §2 item construction

def _lat():
    return {v: _shared(v) for v in ("C1", "C2", "default")}


def answerable_edge(e, lat):
    """The server's filter (`internal.DeltaTaskset.load`) evaluated on an EDGE rather than a built
    item, so an item set can be built that never contains an unservable row. The disease clause is
    per-cell and is applied in `pool_cells`."""
    L = lat.get(e["from"].split(":", 1)[0])
    return not (L is None or e["cross_version"]
                or e["from"] not in L.states or e["to"] not in L.states)


def pool_cells(lat, dis_ok):
    """One pass: every answerable (edge, granularity, horizon, disease) cell, per (axis, op).

    A cell is the unit the item builder samples. Rows are kept as light tuples -- the 421 MB
    edge file does not fit in memory as parsed dicts, and nothing downstream needs the whole row.
    """
    pool = collections.defaultdict(lambda: {"up": [], "dn": [], "ind": []})
    kept = tot = 0
    for line in open(EDGES):
        e = json.loads(line)
        tot += 1
        if not answerable_edge(e, lat):
            continue
        kept += 1
        v = e["from"].split(":", 1)[0]
        meta = (e["from"], e["to"], e["axis"], e["op"], e["code"], e["cost"],
                e["n_band_from"], e["n_band_to"])
        p = pool[(e["axis"], e["op"])]
        for g in (1, 2, 3):
            ok = dis_ok[(v, g)]
            for hi, di, dv, sep in e.get(f"delta_g{g}", ()):
                if di not in ok:
                    continue
                if sep:
                    p["up" if dv > 0 else "dn"].append((meta, g, hi, di, dv, 1))
                elif dv != 0:
                    p["ind"].append((meta, g, hi, di, dv, 0))
    return pool, tot, kept


def _rec(cell, idx):
    (frm, to, ax, op, code, cost, nbf, nbt), g, hi, di, dv, sep = cell
    return {"task": "T2", "axis": ax, "op": op, "code": code, "granularity": g,
            "disease": T.DIS[g][di], "horizon": T.HORIZ[hi], "cost": cost,
            "cross_version": False,
            "answer_direction": ("up" if dv > 0 else "down") if sep else "indistinguishable",
            "answer_delta": round(dv, 6), "_idx": idx,
            "_audit": {"from": frm, "to": to, "separable": sep,
                       "n_band_from": nbf, "n_band_to": nbt}}


def balance_on_size(rows, rng):
    """`gen/build_items.balance_on_size`, reproduced on these rows: equalise the answer within each
    (n_band_from, n_band_to) cell, because the size pair alone predicted the direction 76.2%."""
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        a = r["_audit"]
        cells[(a["n_band_from"], a["n_band_to"])][r["answer_direction"]].append(r)
    out = []
    for byans in cells.values():
        k = min(len(v) for v in byans.values())
        for v in byans.values():
            out += rng.sample(v, k)
    return out


def build(pool, strata, per_split, seed=SEED, ind_frac=0.5):
    """An item set: `per_split` separable items per (axis, op) per split, direction-balanced, plus
    `ind_frac` as many `indistinguishable` controls, then size-balanced exactly as the builder does.

    Sampling is per SPLIT because `split_of` hashes the from-state, so a stratum can be rich in
    train and empty in unseen; drawing globally would hand the shortfall to whichever split lost.
    """
    rng = random.Random(seed)
    out = {s: [] for s in SPLITS}
    idx = collections.Counter()
    for key in strata:
        p = pool.get(key)
        if not p:
            continue
        for sp in SPLITS:
            up = [c for c in p["up"] if split_of(c[0][0]) == sp]
            dn = [c for c in p["dn"] if split_of(c[0][0]) == sp]
            ind = [c for c in p["ind"] if split_of(c[0][0]) == sp]
            k = min(per_split // 2, len(up), len(dn))
            rows = []
            for side in (up, dn):
                for c in rng.sample(side, k):
                    rows.append(_rec(c, idx[sp])); idx[sp] += 1
            for c in rng.sample(ind, min(len(ind), int(2 * k * ind_frac))):
                rows.append(_rec(c, idx[sp])); idx[sp] += 1
            out[sp] += rows
    for sp in SPLITS:
        out[sp] = balance_on_size(out[sp], rng)
        for i, r in enumerate(out[sp]):
            r["_idx"] = i
    return out


def spirit_prior(items):
    """The strongest ZERO-READ policy the POST-SPIRIT prompt admits, cross-fitted.

    The shipped T2 prompt names only (axis, op, code, disease, horizon). The spirit ADDS the from cohort's
    five axes, so the leak channel gets wider, not narrower, and the honest floor has to be refit on
    the wider channel. Key = everything the item states at dial A = 0 masked, coarsened to what a
    prior could plausibly key on (age decade, sex, and how many codes are on each axis).
    """
    def key(d):
        st = _shared(d["_audit"]["from"].split(":", 1)[0]).states[d["_audit"]["from"]]
        ax = st["axes"]
        band = str(ax["age_band"])
        dec = band.split("-", 1)[0][:1] if band[:1].isdigit() else "?"
        return (d["axis"], d["op"], d["code"], dec, str(ax["sex"]),
                len(ax["icd"]), len(ax["opcs"]), len(ax["bnf"]))
    return T.loo_majority(items, key)


def score_set(items, label):
    """The published ladder plus the spirit rung, on one item set."""
    lines, res = T.ladder(items, label=label)
    truth = [d["answer_direction"] for d in items]
    dec = spirit_prior(items)
    rs = [T.score(T.PM[x], t) for x, t in zip(dec, truth)]
    acc = T.boot([1.0 if x == t else 0.0 for x, t in zip(dec, truth)])
    m, lo, hi, k = T.boot(rs)
    res["rung0 SPIRIT prompt LOO"] = {"reward": [m, lo, hi], "n": k, "accuracy": list(acc[:3])}
    lines.append(f"  {'rung0 SPIRIT prompt LOO':<34s} reward {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                 f"   acc {acc[0]:.4f} [{acc[1]:.4f}, {acc[2]:.4f}]   n={k}")
    # The headline: the best no-read policy on THIS set, and the paired gap reading buys over it.
    noread = {"always-hedge": ["indistinguishable"] * len(items),
              "no-read LOO (axis,op)": T.loo_majority(items, lambda i: (i["axis"], i["op"])),
              "no-read AGE RULE only": T.age_only(items),
              "no-read semantic prior": T.semantic(items),
              "no-read LOO (ax,op,hor,gran)": T.loo_majority(
                  items, lambda i: (i["axis"], i["op"], i["horizon"], i["granularity"])),
              "ZERO-READ LOO (axis,op,code)":
                  T.loo_majority(items, lambda i: (i["axis"], i["op"], i["code"])),
              "SPIRIT prompt LOO": dec}
    accs = {n: statistics.fmean(1.0 if x == t else 0.0 for x, t in zip(d, truth))
            for n, d in noread.items()}
    best = max(accs, key=accs.get)
    ba = T.boot([1.0 if x == t else 0.0 for x, t in zip(noread[best], truth)])
    sr = T.sign_reports(items)
    sacc = [1.0 if T.R.t2_decision(r) == t else 0.0 for r, t in zip(sr, truth)]
    gap = T.boot([a - b for a, b in
                  zip(sacc, [1.0 if x == t else 0.0 for x, t in zip(noread[best], truth)])])
    hedge_acc = statistics.fmean(1.0 if t == "indistinguishable" else 0.0 for t in truth)
    res["headline"] = {"n": len(items), "best_noread": best,
                       "best_noread_acc": list(ba[:3]),
                       "hedge_acc": hedge_acc,
                       "excess_over_hedge": ba[0] - hedge_acc,
                       "sign_acc": list(T.boot(sacc)[:3]),
                       "gap_sign_minus_best_noread": list(gap[:3]),
                       "class_balance": res["class_balance"],
                       "separable_frac": statistics.fmean(
                           1.0 if d["_audit"]["separable"] else 0.0 for d in items)}
    lines.append(f"  HEADLINE  best no-read = {best}  acc {ba[0]:.4f} [{ba[1]:.4f}, {ba[2]:.4f}]"
                 f"  = hedge {hedge_acc:.4f} {ba[0] - hedge_acc:+.4f}"
                 f"   sign-minus-floor {gap[0]:+.4f} [{gap[1]:+.4f}, {gap[2]:+.4f}]   n={len(items)}")
    return lines, res


# ------------------------------------------------------- §3/§4 zoom and destination well-posedness

def sample_states(lat, n, seed=SEED):
    """From-states carrying BOTH a same-version zoom edge and a same-version non-zoom edge -- the
    only states where "does the zoom dial move the answer" is even askable."""
    hz, hn = collections.defaultdict(set), collections.defaultdict(set)
    for line in open(EDGES):
        e = json.loads(line, object_pairs_hook=dict)
        if e["cross_version"]:
            continue
        (hz if e["op"] in ZOOM_OPS else hn)[e["from"]].add((e["axis"], e["op"], e["code"], e["to"]))
    both = sorted(set(hz) & set(hn))
    rng = random.Random(seed)
    return set(rng.sample(both, min(n, len(both)))), hz, hn, len(both)


def zoom_and_destination(lat, dis_ok, sample):
    """One pass restricted to the sampled from-states, answering two questions at once.

    §3 zoom: for a state `s`, its same-version zoom neighbour `s'`, and an operation (axis, op,
    code) both carry, does the direction on the SAME (granularity, horizon, disease) cell agree?
    Disagreement means the dial that renders the cohort coarser also moves the answer, i.e. the
    item's question is not the edge's question.

    §4 destination: for `(s, axis, op)` and one cell, do the several destinations `code` can name
    agree on the direction? If not, dial A applied to the operation clause makes the label ill posed.
    """
    rows = collections.defaultdict(list)
    for line in open(EDGES):
        e = json.loads(line)
        if e["from"] not in sample or e["cross_version"] or not answerable_edge(e, lat):
            continue
        v = e["from"].split(":", 1)[0]
        cells = {}
        for g in (1, 2, 3):
            ok = dis_ok[(v, g)]
            for hi, di, dv, sep in e.get(f"delta_g{g}", ()):
                if di in ok:
                    cells[(g, hi, di)] = ("up" if dv > 0 else "down") if sep and dv else \
                                         ("indistinguishable" if dv else None)
        rows[e["from"]].append((e["axis"], e["op"], e["code"], e["to"], cells))

    # ---- §3 zoom: does the direction survive rendering the FROM cohort at another granularity?
    #
    # Three counters, because "agrees 1.0000" on its own is worth nothing:
    #   `all`       every comparable cell -- dominated by indistinguishable == indistinguishable
    #   `sep`       cells where at least ONE side is up/down; this is the informative subset
    #   `control`   the SAME comparison against a RANDOM other from-state's identical (axis, op,
    #               code). If the control also agrees, the comparison is not reading the cohort and
    #               the 1.0000 means nothing.
    z = {"pairs": 0, "matched": 0, "vanish": 0,
         "all_n": 0, "all_agree": 0, "sep_n": 0, "sep_agree": 0, "sep_flip": 0,
         "ctl_n": 0, "ctl_agree": 0, "ctl_sep_n": 0, "ctl_sep_agree": 0}
    keys = sorted(rows)
    rng = random.Random(SEED)
    bykey = collections.defaultdict(list)
    for s_, rs in rows.items():
        for a, o, c, _t, cl in rs:
            if o not in ZOOM_OPS:
                bykey[(a, o, c)].append((s_, cl))

    def compare(cl, other, pre):
        for k, d in cl.items():
            if d is None or k not in other or other[k] is None:
                continue
            z[pre + "_n"] += 1
            z[pre + "_agree"] += (d == other[k])

    for s_ in keys:
        rs = rows[s_]
        zooms = [r for r in rs if r[1] in ZOOM_OPS]
        ops = [r for r in rs if r[1] not in ZOOM_OPS]
        for _, _, _, s2, _ in zooms:
            r2 = rows.get(s2)
            by2 = {(a, o, c): cl for a, o, c, _t, cl in (r2 or []) if o not in ZOOM_OPS}
            for a, o, c, _t, cl in ops:
                z["pairs"] += 1
                other = by2.get((a, o, c))
                if other is None:
                    z["vanish"] += 1
                else:
                    z["matched"] += 1
                    compare(cl, other, "all")
                    for k, d in cl.items():
                        if d is None or k not in other or other[k] is None:
                            continue
                        if d == "indistinguishable" and other[k] == "indistinguishable":
                            continue
                        z["sep_n"] += 1
                        z["sep_agree"] += (d == other[k])
                        z["sep_flip"] += (d != other[k] and "indistinguishable" not in (d, other[k]))
                # NEGATIVE CONTROL on the same (axis, op, code), a random OTHER cohort
                pool_ = [p for p in bykey[(a, o, c)] if p[0] not in (s_, s2)]
                if pool_:
                    _, ocl = rng.choice(pool_)
                    compare(cl, ocl, "ctl")
                    for k, d in cl.items():
                        if d is None or k not in ocl or ocl[k] is None:
                            continue
                        if d == "indistinguishable" and ocl[k] == "indistinguishable":
                            continue
                        z["ctl_sep_n"] += 1
                        z["ctl_sep_agree"] += (d == ocl[k])
    assert z["matched"] + z["vanish"] == z["pairs"]

    # ---- §4 destination: if dial A masked the operation's `code`, several edges fit the item.
    dst = {"groups": 0, "multi": 0, "ambiguous": 0, "sep_multi": 0, "sep_ambiguous": 0}
    per = collections.defaultdict(lambda: collections.defaultdict(set))
    ndest = collections.defaultdict(set)
    for s_, rs in rows.items():
        for a, o, c, _t, cl in rs:
            if o in ZOOM_OPS:
                continue
            ndest[(s_, a, o)].add(c)
            for k, d in cl.items():
                if d is not None:
                    per[(s_, a, o, k)][c].add(d)
    for k, byc in per.items():
        dst["groups"] += 1
        if len(byc) > 1:
            dst["multi"] += 1
            ds = {next(iter(v)) for v in byc.values()}
            if len(ds) > 1:
                dst["ambiguous"] += 1
            # the subset that decides anything: at least one destination is separable, so the
            # item would not be an all-hedge group whatever the agent does.
            if ds - {"indistinguishable"}:
                dst["sep_multi"] += 1
                dst["sep_ambiguous"] += (len(ds) > 1)
    dst["mean_destinations"] = statistics.fmean(len(v) for v in ndest.values()) if ndest else 0.0
    return z, dst


# --------------------------------------------------------------------------- selfcheck

def selfcheck():
    lat = _lat()
    dis_ok = _tables(lat)
    # 0. the bijection the whole supply argument rests on, and the server's own served counts.
    for v, g in GRAN.items():
        assert len(dis_ok[(v, g)]) == len(T.DIS[g]), (v, g)
        assert all(not dis_ok[(v, h)] for h in (1, 2, 3) if h != g), (v, g)
    served, nsplit, bad = collections.Counter(), collections.Counter(), 0
    for sp in SPLITS:
        for line in open(f"{T.ITEMS}/t2_delta_{sp}.jsonl"):
            d = json.loads(line)
            nsplit[sp] += 1
            bad += split_of(d["_audit"]["from"]) != sp
            served[sp] += T.is_served(d)
    assert dict(nsplit) == {"train": 429, "seen": 78, "unseen": 118}, nsplit
    assert dict(served) == {"train": 125, "seen": 31, "unseen": 41}, served
    assert bad == 0, f"{bad} items whose split_of disagrees with the file holding them"
    pool, tot, kept = pool_cells(lat, dis_ok)
    # 1. every row this script builds is one the SERVER would serve.
    st = build(pool, [("bnf", "switch"), ("sex", "switch")], 20, seed=7)
    items = [d for s in SPLITS for d in st[s]]
    assert items, "empty control set"
    assert all(T.is_served(d) for d in items), "built an item the environment cannot answer"
    # 2. the age rule is INERT off its own stratum -- otherwise "the age leak is gone" is vacuous.
    assert set(T.age_only(items)) == {"indistinguishable"}, "age rule is not inert on non-age items"
    # 3. POSITIVE control: a set whose label is a function of (axis, op, code) must be READ OFF by
    #    the zero-read rung. Without this, a low no-read score is not evidence of no leak.
    pos = [dict(d, answer_direction=("up" if hash(str(d["code"])) % 2 else "down"))
           for d in items]
    dec = T.loo_majority(pos, lambda i: (i["axis"], i["op"], i["code"]))
    pacc = statistics.fmean(1.0 if a == b["answer_direction"] else 0.0 for a, b in zip(dec, pos))
    assert pacc > 0.9, f"zero-read rung cannot see a planted leak (acc {pacc:.3f})"
    # 4. NEGATIVE control: labels shuffled -> the same rung must fall to the majority rate.
    rng = random.Random(0)
    lab = [d["answer_direction"] for d in items]
    rng.shuffle(lab)
    neg = [dict(d, answer_direction=l) for d, l in zip(items, lab)]
    dec = T.loo_majority(neg, lambda i: (i["axis"], i["op"], i["code"]))
    nacc = statistics.fmean(1.0 if a == b["answer_direction"] else 0.0 for a, b in zip(dec, neg))
    top = collections.Counter(lab).most_common(1)[0][1] / len(lab)
    assert nacc <= top + 0.05, f"zero-read rung scores {nacc:.3f} on shuffled labels (majority {top:.3f})"
    # 5. the renderer states nothing the release does not, and a masked axis leaves no trace.
    L = _shared("C2")
    s = L.states[sorted(L.states)[0]]
    full, m = render(s), render(s, mask=("icd",))
    assert "aged" not in full and "about" in full, full
    assert str(s["axes"]["age_band"]) not in full, "the age band survived de-grouping"
    assert "diagnoses" not in m and len(m) < len(full), m
    print(f"selfcheck OK  served 125/31/41 re-derived, split_of agrees on 625/625, "
          f"version<->granularity bijection holds; {kept:,}/{tot:,} answerable edges, "
          f"{len(items)} control items all servable, planted-leak acc {pacc:.3f}, "
          f"shuffled acc {nacc:.3f} vs majority {top:.3f}")


def _tables(lat):
    """{(version, granularity): the disease indices readable in that version}. By the bijection
    above this is empty unless g == GRAN[version]; it is computed rather than assumed so the
    assertion in --selfcheck has something to check."""
    return {(v, g): {i for i, d in enumerate(T.DIS[g]) if lat[v] is not None and d in lat[v].didx}
            for v in lat for g in (1, 2, 3)}


# --------------------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", help="artifact path (.txt); a .json sibling is written too")
    ap.add_argument("--per-split", type=int, default=60, help="separable items per stratum per split")
    ap.add_argument("--zoom-sample", type=int, default=4000)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    selfcheck()
    lat = _lat()
    dis_ok = _tables(lat)
    pool, tot, kept = pool_cells(lat, dis_ok)

    L, J = [], {"per_split": a.per_split, "edges_total": tot, "edges_answerable": kept,
                "today_noread_unseen": TODAY_NOREAD_UNSEEN, "boot": BOOT, "seed": SEED,
                }
    P = L.append
    P(f"T2 UNDER THE RENDERING SPIRIT   "
      f"boot {BOOT} seed {SEED}")
    P(f"edge rows {tot:,}; answerable (same-version, both endpoints released) {kept:,} "
      f"({kept/tot:.1%})")
    P("")

    # ---------------- §1 the rendering, made concrete
    P("=== §1 THE RENDERING (one item, all three dial settings) ===")
    Lc = _shared("C2")
    demo = None
    for sid in sorted(Lc.states):
        st = Lc.states[sid]
        if st["axes"]["icd"] and st["axes"]["bnf"]:
            demo = st
            break
    for nm, mk in (("dial A = 0 axes masked", ()), ("dial A = 1 masked (opcs)", ("opcs",)),
                   ("dial A = 2 masked (opcs, bnf)", ("opcs", "bnf"))):
        P(f"  [{nm}]")
        for ln in render(demo, mk).splitlines():
            P(f"    {ln}")
        P(f"    OPERATION: {op_clause('bnf', 'add', '06')}.")
    J["render_example"] = {"state": demo["state_id"], "coarseness": demo.get("coarseness"),
                           "rendered": render(demo)}
    P("")

    # ---------------- §2 supply, then the three variants
    P("=== §2 ANSWERABLE SUPPLY, cells per (axis, op) x split ===")
    hdr = f"  {'(axis, op)':<22}" + "".join(f"{s:>10}" for s in SPLITS) + f"{'balanced/split':>16}"
    P(hdr)
    supply = {}
    for key in sorted(pool, key=lambda k: -sum(len(pool[k][s]) for s in ("up", "dn"))):
        per = {}
        for sp in SPLITS:
            u = sum(1 for c in pool[key]["up"] if split_of(c[0][0]) == sp)
            d = sum(1 for c in pool[key]["dn"] if split_of(c[0][0]) == sp)
            per[sp] = 2 * min(u, d)
        supply[f"{key[0]} {key[1]}"] = per
        P(f"  {key[0] + ' ' + key[1]:<22}" + "".join(f"{per[s]:>10,}" for s in SPLITS)
          + f"{min(per.values()):>16,}")
    J["supply_balanced_per_split"] = supply
    P("")

    NONAGE = [k for k in pool if k[0] != "age_band" and k[1] not in ZOOM_OPS]
    AGE = [("age_band", "shift")]
    variants = {
        "(a) non-age only": NONAGE,
        "(a+) non-age, SUPPLY CEILING (no design cap)": NONAGE,
        "(b) age kept at the shipped share, band-rendered": NONAGE + AGE,
        "(c) age capped at one stratum": NONAGE + AGE,
    }
    P("=== §2 THE THREE OPTIONS, each built and scored on the same ladder ===")
    J["variants"] = {}
    built = {}
    for nm, strata in variants.items():
        if nm.startswith("(b)"):
            # (b) is the shipped MIX: age_band shift at ~2x the other strata, which is what makes it
            # 79/118 of the unseen split.
            s = build(pool, strata, a.per_split)
            extra = build(pool, AGE, a.per_split * 4, seed=SEED + 1)
            s = {k: s[k] + extra[k] for k in s}
            for k in s:
                for i, r in enumerate(s[k]):
                    r["_idx"] = i
        elif nm.startswith("(a+)"):
            # No design cap: every direction-balanced separable cell the environment can serve.
            # This is the CEILING on option (a), so "does (a) starve" is answered by supply and not
            # by a target I chose.
            s = build(pool, strata, 10 ** 9)
        else:
            s = build(pool, strata, a.per_split)
        built[nm] = s
        for sp in SPLITS:
            ln, rs = score_set(s[sp], f"{nm}  [{sp}]")
            L += ln
            J["variants"].setdefault(nm, {})[sp] = rs
        allit = [dict(d, _idx=i) for i, d in enumerate(d2 for sp in SPLITS for d2 in s[sp])]
        ln, rs = score_set(allit, f"{nm}  [POOLED]")
        L += ln
        J["variants"][nm]["pooled"] = rs
    P("")

    # ---------------- §3 / §4
    P("=== §3 DOES THE ZOOM DIAL APPLY? / §4 CAN DIAL A MASK THE DESTINATION? ===")
    zx = collections.Counter()
    for line in open(EDGES):
        e = json.loads(line)
        if e["op"] in ZOOM_OPS:
            zx[e["cross_version"]] += 1
    P(f"  zoom edges in the table: {zx[True] + zx[False]:,}, of which cross-version "
      f"{zx[True]:,} ({zx[True]/(zx[True]+zx[False]):.1%}) -- withheld by legal_moves")
    sample, _hz, _hn, nboth = sample_states(lat, a.zoom_sample)
    z, dst = zoom_and_destination(lat, dis_ok, sample)
    P(f"  from-states carrying both a same-version zoom edge and a non-zoom edge: {nboth:,}; "
      f"sampled {len(sample):,}")
    P(f"  §3 operation pairs across a zoom: {z['pairs']:,} = matched {z['matched']:,} + the same "
      f"(axis, op, code) ABSENT from the zoomed cohort {z['vanish']:,} "
      f"({z['vanish']/max(1, z['pairs']):.1%})")
    ag = z["all_agree"] / max(1, z["all_n"])
    sa = z["sep_agree"] / max(1, z["sep_n"])
    ca = z["ctl_agree"] / max(1, z["ctl_n"])
    cs = z["ctl_sep_agree"] / max(1, z["ctl_sep_n"])
    P(f"     direction agrees: all cells {ag:.4f} (n={z['all_n']:,}); cells where at least one "
      f"side is up/down {sa:.4f} (n={z['sep_n']:,}); up/down FLIPS {z['sep_flip']:,}")
    P(f"     NEGATIVE CONTROL, same (axis, op, code) on a RANDOM other cohort: all {ca:.4f} "
      f"(n={z['ctl_n']:,}); at-least-one-separable {cs:.4f} (n={z['ctl_sep_n']:,})")
    P(f"  §4 (from, axis, op, cell) groups {dst['groups']:,}; mean destinations per (from, axis, op) "
      f"{dst['mean_destinations']:.2f}")
    P(f"     groups with >1 destination {dst['multi']:,} ({dst['multi']/max(1,dst['groups']):.1%}); "
      f"destinations DISAGREE on the direction {dst['ambiguous']:,} "
      f"({dst['ambiguous']/max(1, dst['multi']):.1%})")
    P(f"     of the {dst['sep_multi']:,} multi-destination groups where at least one destination is "
      f"SEPARABLE, {dst['sep_ambiguous']:,} disagree "
      f"({dst['sep_ambiguous']/max(1, dst['sep_multi']):.1%}) -- this is the subset that decides a "
      f"masked-destination item")
    J["zoom"] = dict(z, agree_all=ag, agree_sep=sa, control_agree_all=ca, control_agree_sep=cs,
                     cross_version_zoom_edges=zx[True],
                     same_version_zoom_edges=zx[False], states_with_both=nboth,
                     sampled=len(sample))
    J["destination_mask"] = dst

    out = "\n".join(L)
    print(out)
    if a.out:
        open(a.out, "w").write(out + "\n")
        json.dump(J, open(a.out.replace(".txt", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
