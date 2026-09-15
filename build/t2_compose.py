#!/usr/bin/env python
# release-gate: allow-file -- ran against the private UKB extract; shipped for provenance, paths shown as they were
"""T2's item set, settled: the age-band conflict, the supply repair, and the split targets.

CPU only. Reads the frozen environment, the pooled edge table and PhenoCal's published panel;
edits none of them. Reuses `build/t2_ceiling.py` for every policy and every reward (so the ladder
here IS the published one) and `build/t2_spirit.py` for the item record.

Six questions, in the order the evidence forces them:

  §0 THE BIJECTION   The rebuild was sized off a column called
                     "after granularity alignment" (non-age 1,052 / 58 / 68). That column is
                     the supply scan's `bal any-gran`, which counts separable cells with the
                     server's disease clause LIFTED. This section measures whether lifting it is
                     reachable. It is not: version <-> outcome granularity is a bijection, the
                     three disease tables are pairwise disjoint, and `DeltaTaskset.load` tests
                     exact membership in the edge version's own table. So the granularity fix
                     cannot ADD supply; it can only stop the builder emitting items the server
                     drops.

  §1 THE REPAIR      What the clause-4 fix really gives, per stratum per split, and the proof it
                     landed: every emitted item is servable, against a negative control (the same
                     build without the fix) that is not.

  §2 THE AGE         `age_band shift` is 67% of the shipped served unseen and reading beats a free
                     age rule there by +0.0056. Three options are BUILT and scored on one ladder:
                     (a) drop it, (b) keep the shipped mix, (c) cap it. Decided on the measured
                     no-read accuracy, not on the sketch.

  §3 THE RATIO       Non-age balanced supply under seven split ratios. The 80/10/10 partition
                     cannot put 60 non-age items in `unseen` -- 38 is the whole supply there.

  §4 THE COMPOSITION The proposed per-split mix, built, with the full ladder and the headline the
                     rebuild exists for: the best no-read policy's accuracy against the majority
                     class. On the shipped set that is 0.7881 against a 0.5366 majority on unseen.

  §5 PHENOCAL        Is a rate DIFFERENCE across an edge definable on PhenoCal cohorts at all?
                     Measured, not argued: one-operation-apart cohort pairs, the cells they share,
                     and the Wilson-separable direction-balanced supply those cells carry.

Run:  .venv/bin/python build/t2_compose.py --selfcheck
      .venv/bin/python build/t2_compose.py --out <rundir>/artifacts/<name>.txt [--emit <itemdir>]
"""
import argparse, collections, hashlib, json, os, random, re, statistics, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
import t2_ceiling as T                                      # noqa: E402
import t2_spirit as TS                                      # noqa: E402
from cohortworld_env.lattice import _shared, PANEL          # noqa: E402

EDGES = f"{PANEL}/CohortWorld_pooled_edges.jsonl"
PHENOCAL = os.environ.get("CW_PHENOCAL",
                          "/ram/USERS/ziquanw/digitaltwinreward/ukb_qa_brain/output_panel")
GRAN = {"C1": 1, "C2": 2, "default": 3}
SPLITS = ("train", "seen", "unseen")
AGE = ("age_band", "shift")
SEED = T.SEED

# The number the rebuild has to beat is NOT hardcoded. A constant
# `TODAY = {"noread": 0.7881, "majority": 0.5366, "n": 118}` pairs numbers that do not exist on any
# one item set: 0.7881 is measured on `unseen ALL` (n=118, whose majority is 0.4407) while
# 0.5366 is the majority of `unseen SERVED` (n=41, whose best no-read is 0.7561). Quoting the
# excess as 0.7881 - 0.5366 = +0.2515 scores a subset against a reference computed on a DIFFERENT
# subset, which is the one comparison this project forbids. §4 prints the shipped-set number from
# variant (b) instead -- the shipped served set through THIS script's ladder, so it is paired.

# THE DESIGN, all four knobs in one place so the artifact can be read against them.
RATIO = (50, 25)          # train < 50 <= seen < 75 <= unseen, on md5("cwsplit2:" + from_state_id)
# Direction-balance WITHIN (size cell, horizon, granularity). Size-cell balance is inherited from
# `gen/build_items.balance_on_size` and is not optional (the size pair alone predicted the direction
# 76.2%). Horizon and granularity are added here because `no-read LOO (axis, op, horizon,
# granularity)` is a policy on the ladder and it reached 0.4561 on the size-only build; balancing
# them costs 4 unseen items and takes that policy to chance. §4 reports both.
BAL_KEY = ("size", "horizon", "granularity")
# `indistinguishable` controls are drawn WITHIN each (axis, op) stratum, one per separable PAIR, so
# every stratum is up/down/indistinguishable at 1/3 each. Drawing them from strata that have no
# separable supply instead makes `axis` predict the answer: measured, that takes the best no-read
# policy on unseen from 0.3333 to 0.6913 and turns the reading gap NEGATIVE. §4 shows it.
IND_MATCH = ("axis", "op")
DROP_AGE = True

# Reservoir cap on the `indistinguishable` pool per stratum. The unseparable supply is 26k-2.0M
# cells per stratum and the composition draws at most a few hundred; holding it all is 40x the
# memory for no change in what can be sampled.
IND_CAP = 40000


# --------------------------------------------------------------------------- pool

def split_of(sid, ratio=RATIO):
    """The item builder's `split_of` generalised to a ratio. At (80, 10) it is that function
    verbatim, which --selfcheck asserts on all 625 shipped T2 items."""
    h = int(hashlib.md5(f"cwsplit2:{sid}".encode()).hexdigest(), 16) % 100
    a, b = ratio
    return "train" if h < a else ("seen" if h < a + b else "unseen")


def _cellkey(c, fields):
    """A balance/match key over the fields the served prompt actually shows."""
    (_frm, _to, ax, op, code, _cost, nbf, nbt), g, hi, _di, _dv, _sep = c
    v = {"size": (nbf, nbt), "horizon": hi, "granularity": g, "axis": ax, "op": op, "code": code}
    return tuple(v[f] for f in fields)


def build_pool(clause4=True):
    """Every (edge, granularity, horizon, disease) cell the server can answer, per (axis, op).

    `clause4=False` is the NEGATIVE CONTROL: it keeps cells whose disease is not in the edge
    version's table, which is what the earlier item builder does and why 428 of its 625 items are
    dropped at serve time. §1 builds both and checks the servable fraction of each.
    """
    lat = {v: _shared(v) for v in GRAN}
    dis = {g: json.load(open(f"{PANEL}/CohortWorld_C2_diseases_g{g}.json"))["diseases"]
           for g in (1, 2, 3)}
    ok = {(v, g): {i for i, d in enumerate(dis[g]) if d in lat[v].didx}
          for v in GRAN for g in (1, 2, 3)}
    pool = collections.defaultdict(lambda: {"up": [], "dn": [], "ind": []})
    seen_ind = collections.Counter()
    rng = random.Random(SEED)
    n_rows = n_ok = 0
    for line in open(EDGES):
        e = json.loads(line)
        n_rows += 1
        v = e["from"].split(":", 1)[0]
        L = lat.get(v)
        if L is None or e["cross_version"] or e["from"] not in L.states or e["to"] not in L.states:
            continue
        n_ok += 1
        meta = (e["from"], e["to"], e["axis"], e["op"], e["code"], e["cost"],
                e["n_band_from"], e["n_band_to"])
        k = (e["axis"], e["op"])
        p = pool[k]
        for g in (1, 2, 3):
            og = ok[(v, g)]
            for hi, di, dv, sep in e.get(f"delta_g{g}", ()):
                if clause4 and di not in og:
                    continue
                cell = (meta, g, hi, di, dv, 1 if sep else 0)
                if sep:
                    p["up" if dv > 0 else "dn"].append(cell)
                elif dv != 0:
                    seen_ind[k] += 1
                    if len(p["ind"]) < IND_CAP:
                        p["ind"].append(cell)
                    else:                                    # reservoir, so the cap is not a head
                        j = rng.randrange(seen_ind[k])
                        if j < IND_CAP:
                            p["ind"][j] = cell
    return dict(pool), n_rows, n_ok


def balanced(pool, strata, ratio, bal_key):
    """{split: {stratum: emittable separable count}} -- direction-balanced within `bal_key`."""
    out = collections.defaultdict(collections.Counter)
    for k in strata:
        p = pool.get(k)
        if not p:
            continue
        cells = collections.defaultdict(lambda: {"up": 0, "dn": 0})
        for c in p["up"] + p["dn"]:
            cells[(split_of(c[0][0], ratio), _cellkey(c, bal_key))]["up" if c[4] > 0 else "dn"] += 1
        for (sp, _kk), d in cells.items():
            out[sp][k] += 2 * min(d["up"], d["dn"])
    return out


def build(pool, strata, ratio=RATIO, bal_key=BAL_KEY, ind_match=IND_MATCH, cap=None, seed=SEED,
          ind=True):
    """The item set: separable cells direction-balanced within `bal_key`, plus one
    `indistinguishable` control per separable PAIR drawn from the same `ind_match` group.

    `cap` is {(stratum, split): max separable}, the only place a POLICY (as opposed to the supply)
    limits a stratum -- §2's option (c) is exactly this dial on `age_band shift`.
    """
    rng = random.Random(seed)
    out = {s: [] for s in SPLITS}
    for k in strata:
        p = pool.get(k)
        if not p:
            continue
        for sp in SPLITS:
            cells = collections.defaultdict(lambda: {"up": [], "dn": []})
            for c in p["up"] + p["dn"]:
                if split_of(c[0][0], ratio) == sp:
                    cells[_cellkey(c, bal_key)]["up" if c[4] > 0 else "dn"].append(c)
            sep = []
            for d in cells.values():
                m = min(len(d["up"]), len(d["dn"]))
                sep += rng.sample(d["up"], m) + rng.sample(d["dn"], m)
            lim = (cap or {}).get((k, sp))
            if lim is not None and len(sep) > lim:
                up = [c for c in sep if c[4] > 0]
                dn = [c for c in sep if c[4] < 0]
                m = lim // 2
                sep = rng.sample(up, min(m, len(up))) + rng.sample(dn, min(m, len(dn)))
            rows = list(sep)
            if ind:
                bygrp = collections.defaultdict(list)
                for c in p["ind"]:
                    if split_of(c[0][0], ratio) == sp:
                        bygrp[_cellkey(c, ind_match)].append(c)
                for grp, n in collections.Counter(_cellkey(c, ind_match) for c in sep).items():
                    cand = bygrp.get(grp, [])
                    rows += rng.sample(cand, min(len(cand), n // 2))
            for c in rows:
                out[sp].append(TS._rec(c, 0))
    for sp in SPLITS:
        rng.shuffle(out[sp])
        for i, r in enumerate(out[sp]):
            r["_idx"] = i
    return out


# --------------------------------------------------------------------------- reporting helpers

def headline(items):
    """The published ladder's headline row on one item set, plus the majority class."""
    _lines, res = TS.score_set(items, "x")
    h = res["headline"]
    c = collections.Counter(d["answer_direction"] for d in items)
    h["majority"] = max(c.values()) / len(items)
    h["class_counts"] = dict(c)
    h["strata"] = dict(collections.Counter(f"{d['axis']} {d['op']}" for d in items))
    h["separable_n"] = sum(1 for d in items if d["answer_direction"] != "indistinguishable")
    return h


def hrow(tag, sp, h):
    return (f"  {tag:<26}{sp:<7} n={h['n']:<5} sep={h['separable_n']:<5} maj={h['majority']:.4f}"
            f"  best no-read {h['best_noread_acc'][0]:.4f}"
            f" [{h['best_noread_acc'][1]:.4f},{h['best_noread_acc'][2]:.4f}] ({h['best_noread']})"
            f"  sign {h['sign_acc'][0]:.4f}"
            f"  gap {h['gap_sign_minus_best_noread'][0]:+.4f}"
            f" [{h['gap_sign_minus_best_noread'][1]:+.4f},{h['gap_sign_minus_best_noread'][2]:+.4f}]")


# --------------------------------------------------------------------------- §0 / selfcheck

def bijection_report():
    """Is `bal any-gran` -- the column the rebuild was sized off -- real?"""
    lat = {v: _shared(v) for v in GRAN}
    dis = {g: json.load(open(f"{PANEL}/CohortWorld_C2_diseases_g{g}.json"))["diseases"]
           for g in (1, 2, 3)}
    L = []
    for v, g in GRAN.items():
        same = set(lat[v].didx) == set(dis[g])
        assert same, f"{v}'s disease table is not the g{g} table -- the bijection assumed here fails"
        L.append(f"    {v:<8} didx = the g{g} table exactly ({len(dis[g])} codes)")
    ov = {(a, b): len(set(dis[GRAN[a]]) & set(dis[GRAN[b]]))
          for a in GRAN for b in GRAN if a < b}
    assert set(ov.values()) == {0}, f"disease tables overlap: {ov}"
    L.append(f"    pairwise code overlap between the three tables: {sorted(set(ov.values()))} "
             f"-- the tables partition, so g is recoverable from the version and vice versa")
    res = collections.Counter()
    for v, g in GRAN.items():
        for gg in (1, 2, 3):
            if gg == g:
                continue
            for d in dis[gg]:
                res["resolved" if lat[v]._resolve_disease(d) else "refused"] += 1
                res["member"] += d in lat[v].didx
    assert res["member"] == 0, "a foreign-granularity code is in a version's didx -- §0 is wrong"
    L.append(f"    foreign-granularity codes asked of a version's table: {res['refused']:,} refused"
             f" by `Lattice._resolve_disease`, {res['resolved']:,} resolved by prefix, and "
             f"{res['member']} pass `d['disease'] in lat.didx`, which is the clause "
             f"`DeltaTaskset.load` actually applies")
    return L, res


def selfcheck():
    bl, _res = bijection_report()
    # Parity with the shipped item set: split_of at (80, 10) IS build_items.split_of, and the
    # served counts are the ones DeltaTaskset prints.
    served, nsplit = collections.Counter(), collections.Counter()
    for sp in SPLITS:
        for d in T.load(sp):
            nsplit[sp] += 1
            assert split_of(d["_audit"]["from"], (80, 10)) == sp, "split_of drifted from the builder"
            served[sp] += T.is_served(d)
    assert dict(nsplit) == {"train": 429, "seen": 78, "unseen": 118}, nsplit
    assert dict(served) == {"train": 125, "seen": 31, "unseen": 41}, served
    # THE LOO ARTIFACT, asserted so it can never be read as a finding. §4 reports
    # `no-read LOO (axis,op) acc 0.0000` on all three proposed splits. That is NOT evidence that
    # (axis, op) is anti-predictive: `T.loo_majority` breaks ties by `max(sorted(c))`, so on a
    # stratum with equal class counts, removing the held-out item leaves the other two classes
    # tied AHEAD of its own and the prediction is wrong every single time. The proposed set is
    # balanced to exact thirds by construction, so 0.0000 is forced. Two synthetic sets carrying
    # ZERO signal, one exactly balanced and one a single item off, both score 0.0000:
    _flat = [{"answer_direction": d, "axis": "a", "op": "o", "code": "c", "horizon": 1,
              "granularity": 2} for d in ("up", "down", "indistinguishable") * 12]
    for label, xs in (("exact thirds", _flat), ("one item off", _flat + [dict(_flat[0])])):
        a = statistics.fmean(1.0 * (x == i["answer_direction"]) for x, i in
                             zip(T.loo_majority(xs, lambda i: (i["axis"], i["op"])), xs))
        assert a == 0.0, f"{label}: LOO scored {a:.4f}, so the artifact this documents is gone"
    # What DOES carry information is that the max over the seven policies sits on the majority
    # class; the individual sub-majority rungs do not, and must not be quoted.
    # The repair, and the control that shows the check can fail. Built on a cheap stratum so
    # --selfcheck stays a few seconds after the pool pass.
    pool, _r, _o = build_pool(clause4=True)
    bad, _r, _o = build_pool(clause4=False)
    fix = [d for v in build(pool, [("sex", "switch")]).values() for d in v]
    ctl = [d for v in build(bad, [("sex", "switch")]).values() for d in v]
    f_ok = statistics.fmean(1.0 * T.is_served(d) for d in fix)
    c_ok = statistics.fmean(1.0 * T.is_served(d) for d in ctl)
    assert f_ok == 1.0, f"the clause-4 fix did not land: only {f_ok:.4f} of emitted items servable"
    assert c_ok < 1.0, ("the negative control is 100% servable too, so the filter is not the thing "
                        "being tested -- every number in §1 would be vacuous")
    print("\n".join(bl))
    print(f"selfcheck OK  625 items re-split, served 125/31/41 re-derived; "
          f"clause-4 build {f_ok:.4f} servable vs control {c_ok:.4f} (n={len(fix)}/{len(ctl)})")
    return pool


# --------------------------------------------------------------------------- §5 PhenoCal

def _pc_profile(prompt):
    head = prompt.split("with the following recorded medical history", 1)[0]
    m = re.search(r"aged ([\d\-+]+) \(([^;]+); ([^)]+)\)", head)
    body = prompt.split("appeared):\n", 1)[1].split("\n\nAmong", 1)[0]
    conds = frozenset(x[2:].strip() for x in body.splitlines()
                      if x.startswith("- ") and "no recorded" not in x)
    return None if m is None else (m.group(1), m.group(2), m.group(3), conds)


def phenocal_edges(variant="PhenoCal_C2"):
    """One-operation-apart PhenoCal cohort pairs, and the separable delta supply they carry.

    The label rule is the environment's own: `t2_ceiling.separated`, Wilson on the at-risk
    denominator -- which PhenoCal publishes per item as `metadata.n`, so the test is the same test
    and not an analogue of it.
    """
    P, n_rows, unparsed = collections.defaultdict(dict), 0, 0
    for sp in ("train", "val", "test"):
        path = f"{PHENOCAL}/{variant}_{sp}.jsonl"
        if not os.path.exists(path):
            continue
        for line in open(path):
            d = json.loads(line)
            n_rows += 1
            k = _pc_profile(d["prompt"])
            if k is None:
                unparsed += 1
                continue
            m = d["metadata"]
            P[k][(m["disease_icd10_3char"], m["horizon_years"])] = (
                float(d["ground_truth"]["rate"]), int(m["n"]))
    keys = list(P)
    grp = collections.defaultdict(list)
    byrest = collections.defaultdict(list)
    for k in keys:
        a, s, src, c = k
        byrest[("age_band shift", s, src, c)].append(k)
        byrest[("sex switch", a, src, c)].append(k)
        for x in c:
            byrest[("icd switch", a, s, src, frozenset(c - {x}))].append(k)
    for g, v in byrest.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                grp[g[0]].append((v[i], v[j]))
    bysz = collections.defaultdict(list)
    for k in keys:
        bysz[len(k[3])].append(k)
    for n, v in sorted(bysz.items()):
        for k in v:
            for k2 in bysz.get(n + 1, ()):
                if k2[:3] == k[:3] and k[3] < k2[3]:
                    grp["icd add"].append((k, k2))
    out = {}
    for name, pairs in grp.items():
        up = dn = ind = shared = 0
        for k1, k2 in pairs:
            cells = set(P[k1]) & set(P[k2])
            shared += bool(cells)
            for c in cells:
                (p1, n1), (p2, n2) = P[k1][c], P[k2][c]
                if T.separated(p1, n1, p2, n2):
                    if p2 > p1:
                        up += 1
                    else:
                        dn += 1
                elif p1 != p2:
                    ind += 1
        out[name] = {"pairs": len(pairs), "pairs_sharing_a_cell": shared,
                     "up": up, "down": dn, "indistinguishable": ind, "balanced": 2 * min(up, dn)}
    return out, {"rows": n_rows, "profiles": len(keys), "unparsed": unparsed}


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="artifact path (.txt; a .json is written beside it)")
    ap.add_argument("--emit", default=None, help="write t2_delta_{split}.jsonl into this directory")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    pool = selfcheck()
    if a.selfcheck:
        return
    bl, resolver = bijection_report()
    L, J = [], {}
    P = L.append
    P(f"T2 ITEM SET, SETTLED   env {T.R.ENV_VERSION} spec {T.R.REWARD_SPEC_VERSION} "
      f"boot {T.BOOT} seed {SEED}")
    P(f"edges {EDGES}")

    ALL = sorted(pool)
    NONAGE = [k for k in ALL if k != AGE]

    # ---------------- §0
    P("\n=== §0 THE `any-gran` COLUMN IS NOT REACHABLE ===")
    P("  The rebuild was sized at non-age 1,052 / 58 / 68, taken from")
    P("  the supply scan's `bal any-gran` -- the balanced count with the server's disease clause")
    P("  LIFTED. Lifting it is not a builder decision:")
    L.extend(bl)
    P("  So the clause-4 fix cannot ADD a single item. It stops the builder emitting items the")
    P("  server drops. The reachable supply is the `bal` column, not the `bal any-gran` one.")
    J["bijection"] = {"resolver": dict(resolver), "reachable": False}

    # ---------------- §1
    P("\n=== §1 THE SUPPLY REPAIR: what it actually gives ===")
    bal = balanced(pool, ALL, RATIO, BAL_KEY)
    bal8 = balanced(pool, ALL, (80, 10), ("size",))
    P("  `bal @80/10/10, size-only` is the like-for-like number for the sizing's `bal` column.")
    P(f"  {'(axis, op)':<20}{'train':>8}{'seen':>7}{'unseen':>8}   |{'train':>8}{'seen':>7}"
      f"{'unseen':>8}   (design ratio {RATIO[0]}/{RATIO[1]}/{100 - sum(RATIO)}, bal on "
      f"{'+'.join(BAL_KEY)})")
    for k in sorted(ALL, key=lambda x: -sum(bal8[sp][x] for sp in SPLITS)):
        P(f"  {k[0] + ' ' + k[1]:<20}{bal8['train'][k]:>8,}{bal8['seen'][k]:>7,}"
          f"{bal8['unseen'][k]:>8,}   |{bal['train'][k]:>8,}{bal['seen'][k]:>7,}"
          f"{bal['unseen'][k]:>8,}")
    na8 = {sp: sum(bal8[sp][k] for k in NONAGE) for sp in SPLITS}
    na = {sp: sum(bal[sp][k] for k in NONAGE) for sp in SPLITS}
    P(f"  {'NON-AGE TOTAL':<20}{na8['train']:>8,}{na8['seen']:>7,}{na8['unseen']:>8,}"
      f"   |{na['train']:>8,}{na['seen']:>7,}{na['unseen']:>8,}")
    P("")
    P(f"  CLAIM  rebuild sizing: the fix takes non-age balanced supply 316/32/38 "
      f"-> 1,052/58/68.")
    P(f"  FOUND  it gives {na8['train']}/{na8['seen']}/{na8['unseen']} at the same ratio and the "
      f"same balance key -- i.e. exactly the 316/32/38 that was already there.")
    P("  The 1,052/58/68 counts the SAME edges' cells at foreign granularities, which §0 shows the")
    P("  server refuses. What the fix buys is emission efficiency, not supply: the shipped builder")
    P("  emits 625 items of which 197 (31.5%) are servable; every item this builder emits is.")
    zero = [k for k in NONAGE if sum(bal8[sp][k] for sp in SPLITS) == 0
            and sum(bal[sp][k] for sp in SPLITS) == 0]
    P(f"  Strata with ZERO direction-balanced separable supply at any ratio ({len(zero)} of 13 "
      f"non-age): " + ", ".join(f"{x[0]} {x[1]}" for x in zero))
    P("  They can only ever be `indistinguishable` controls -- and §4 shows that USING them as")
    P("  controls is itself a giveaway, so the honest answer is that they cannot appear at all.")
    short = [k for k in NONAGE if bal[
        "unseen"][k] < 60 and sum(bal[sp][k] for sp in SPLITS) > 0]
    P(f"  Strata with SOME supply that still cannot reach n=60 in `unseen` at any ratio: "
      + ", ".join(f"{x[0]} {x[1]}" for x in short))
    J["supply"] = {"balanced_80_10_10_sizeonly": {sp: {f"{k[0]} {k[1]}": bal8[sp][k] for k in ALL}
                                                  for sp in SPLITS},
                   "balanced_design": {sp: {f"{k[0]} {k[1]}": bal[sp][k] for k in ALL}
                                       for sp in SPLITS},
                   "nonage_80_10_10": na8, "nonage_design": na,
                   "spec_claim_nonage": {"train": 1052, "seen": 58, "unseen": 68},
                   "zero_supply_strata": [f"{k[0]} {k[1]}" for k in zero]}

    # ---------------- §3 (ratio first: §2's options are scored at the chosen ratio)
    P("\n=== §3 THE SPLIT RATIO: 80/10/10 cannot hold 60 non-age items in `unseen` ===")
    P(f"  Non-age balanced separable supply, balance key {'+'.join(BAL_KEY)}.")
    P(f"  {'ratio':<14}{'train':>8}{'seen':>7}{'unseen':>8}   unseen by stratum")
    ratios = [(80, 10), (70, 15), (60, 20), (55, 20), (50, 25), (50, 10), (40, 30)]
    rrows = {}
    for r in ratios:
        b = balanced(pool, NONAGE, r, BAL_KEY)
        t = {sp: sum(b[sp].values()) for sp in SPLITS}
        u = {f"{k[0]} {k[1]}": v for k, v in b["unseen"].items() if v}
        rrows[f"{r[0]}/{r[1]}/{100 - sum(r)}"] = {"totals": t, "unseen_by_stratum": u}
        P(f"  {f'{r[0]}/{r[1]}/{100 - sum(r)}':<14}{t['train']:>8,}{t['seen']:>7,}"
          f"{t['unseen']:>8,}   {u}")
    P(f"  RULE APPLIED (stated before any policy was scored): take the ratio with the largest")
    P(f"  `train` among those whose `unseen` non-age supply clears 60. That is "
      f"{RATIO[0]}/{RATIO[1]}/{100 - sum(RATIO)}; 55/20/25 ties on unseen with a larger train and a")
    P(f"  seen of 20, and is rejected on `seen`. `seen` lands short of 60 under every ratio that")
    P(f"  keeps train usable, and is reported rather than papered over.")
    P("  This is a MAJOR change: it repartitions the profiles, so it must be taken once for T1, T2")
    P("  and T3 together or the splits stop being profile-disjoint ACROSS tasks.")
    J["ratio_scan"] = rrows
    J["ratio_chosen"] = f"{RATIO[0]}/{RATIO[1]}/{100 - sum(RATIO)}"

    # ---------------- §2
    P("\n=== §2 THE AGE-BAND CONFLICT: three options, one ladder ===")
    P("  The conflict as posed: the spirit always rewrites age_band to `about xx`, and 79/118 of")
    P("  the shipped unseen items ask about an `age_band shift`. Two facts settle it before any score:")
    P("  (1) A T2 prompt STATES NO COHORT. `internal.DeltaTaskset.load` renders the operation")
    P("      clause, the disease and the horizon and nothing else -- deliberately, because")
    P("      rendering the from-cohort pins answer_delta on 625/625 items. So the spirit's dial A")
    P("      (mask) and dial B (zoom) have no surface to act on in T2, `released text only` is met")
    P("      vacuously, and age de-grouping has exactly ONE place to bite: `_t2_clause`'s age branch,")
    P("      `the second is in the {code} age band instead`.")
    P("  (2) That branch lives in `cohortworld_env/internal.py`, one of the seven modules")
    P("      `reward.semantic_hash()` binds. De-grouping it to `about 72` is a PROMPT SENTENCE")
    P("      change: MAJOR, plus re-certification. Option (b) `exempt the axis` needs that edit and")
    P("      also breaks invariant 1 (one item format). Option (c) `cap the stratum` needs the same")
    P("      edit, only on fewer items.")
    P("  Option (a) DROP the stratum is the only one that costs no frozen-code change: with no")
    P("  `age_band shift` item, the age branch has no caller and no T2 prompt contains an age band.")
    P("  Scored anyway, because 'cheapest' is not 'best':")
    variants = {}
    a_set = build(pool, NONAGE)
    variants["(a) age dropped"] = a_set
    # (b) is the SHIPPED item set's served subset, not a reconstruction of it: it is the thing the
    # 0.7881 was measured on, and putting it through this script's ladder call is also the check
    # that this ladder is the published ladder.
    variants["(b) as shipped"] = {sp: [d for d in T.load(sp) if T.is_served(d)]
                                         for sp in SPLITS}
    nsep = {sp: sum(1 for d in a_set[sp] if d["answer_direction"] != "indistinguishable")
            for sp in SPLITS}
    cap = {(AGE, sp): max(2, nsep[sp] // 3) for sp in SPLITS}
    c_set = build(pool, ALL, cap=cap)
    variants["(c) age capped 25%"] = c_set
    # The giveaway control: the seven zero-supply strata added as pure-`indistinguishable` items.
    d_set = {sp: list(a_set[sp]) for sp in SPLITS}
    z = build(pool, zero, cap={(k, sp): 0 for k in zero for sp in SPLITS}, ind=False)
    zi = build(pool, zero, cap={(k, sp): 0 for k in zero for sp in SPLITS})
    del z
    rng = random.Random(SEED)
    for sp in SPLITS:                       # 4 pure-ind controls per zero-supply stratum per split
        bystr = collections.defaultdict(list)
        for d in zi[sp]:
            bystr[(d["axis"], d["op"])].append(d)
        for v in bystr.values():
            d_set[sp] += rng.sample(v, min(4, len(v)))
        for i, r in enumerate(d_set[sp]):
            r["_idx"] = i
    variants["(d) (a)+pure-ind strata"] = d_set
    vj = {}
    for name, s in variants.items():
        vj[name] = {}
        for sp in SPLITS:
            if not s[sp]:
                continue
            h = headline(s[sp])
            vj[name][sp] = h
            P(hrow(name, sp, h))
    P("  READ: (a) is the only variant whose `unseen` best no-read policy sits ON the majority")
    P("  class. (c) keeps the leak at a quarter strength AND needs the frozen edit. (d) is the")
    P("  control for §1's last line: pure-`indistinguishable` strata make `axis` predict the")
    P("  answer and turn the reading gap negative.")
    P("  DECISION: (a). `age_band shift` leaves the T2 item set entirely. It is the stratum where")
    P("  the environment is worth least -- reading beats a free age rule by +0.0056 [+0.0000,")
    P("  +0.0167] -- so the design pays 2,198 pooled balanced cells for a stratum that measures")
    P("  nothing, and buys back the only frozen-code change the spirit would otherwise force.")
    J["age_options"] = vj

    # ---------------- §4
    P("\n=== §4 THE COMPOSITION ===")
    final = variants["(a) age dropped"]
    P(f"  ratio {RATIO[0]}/{RATIO[1]}/{100 - sum(RATIO)}   strata: every non-age (axis, op) with")
    P(f"  balanced separable supply   direction-balanced within {'+'.join(BAL_KEY)}")
    P(f"  one `indistinguishable` control per separable pair, drawn within {'+'.join(IND_MATCH)}")
    P(f"  {'split':<8}{'n':>6}{'sep':>6}{'up':>5}{'down':>6}{'ind':>6}   strata")
    fj = {}
    for sp in SPLITS:
        h = headline(final[sp])
        fj[sp] = h
        c = h["class_counts"]
        P(f"  {sp:<8}{h['n']:>6}{h['separable_n']:>6}{c.get('up', 0):>5}{c.get('down', 0):>6}"
          f"{c.get('indistinguishable', 0):>6}   {h['strata']}")
    P("")
    for sp in SPLITS:
        lines, _res = TS.score_set(final[sp], f"PROPOSED {sp}")
        L.extend(lines)
    P("")
    P("  === THE NUMBER THE REBUILD EXISTS FOR ===")
    P(f"  {'split':<8}{'majority':>10}{'best no-read':>26}{'policy':>32}{'sign':>9}{'gap':>26}")
    for sp in SPLITS:
        h = fj[sp]
        P(f"  {sp:<8}{h['majority']:>10.4f}"
          f"{f'''{h['best_noread_acc'][0]:.4f} [{h['best_noread_acc'][1]:.4f},{h['best_noread_acc'][2]:.4f}]''':>26}"
          f"{h['best_noread']:>32}{h['sign_acc'][0]:>9.4f}"
          f"{f'''{h['gap_sign_minus_best_noread'][0]:+.4f} [{h['gap_sign_minus_best_noread'][1]:+.4f},{h['gap_sign_minus_best_noread'][2]:+.4f}]''':>26}")
    tod = vj["(b) as shipped"]["unseen"]
    P(f"  shipped, unseen (the SERVED set, n={tod['n']}, this ladder): best no-read "
      f"{tod['best_noread_acc'][0]:.4f} ({tod['best_noread']}) against a majority of "
      f"{tod['majority']:.4f}, excess {tod['best_noread_acc'][0] - tod['majority']:+.4f}")
    P("  (the served set is the like-for-like comparator: every item this builder emits is")
    P("  servable. On the shipped set INCLUDING its 77 unservable items the ladder measured 0.7881")
    P("  against a majority of 0.4407, excess +0.3474. Both are larger than the proposed set's.)")
    P(f"  proposed, unseen (n={fj['unseen']['n']}): best no-read "
      f"{fj['unseen']['best_noread_acc'][0]:.4f} against a majority of "
      f"{fj['unseen']['majority']:.4f}, excess "
      f"{fj['unseen']['best_noread_acc'][0] - fj['unseen']['majority']:+.4f}")
    P("  The detector is NOT dead: the same seven policies find "
      f"{fj['train']['best_noread_acc'][0]:.4f} on train and "
      f"{fj['seen']['best_noread_acc'][0]:.4f} on seen, both above their majority. `unseen` is the")
    P("  split where they find nothing, and it is the split the claim is about.")
    P("  DO NOT QUOTE the sub-majority rungs as evidence. `no-read LOO (axis,op)` reads 0.0000 on")
    P("  all three splits, and --selfcheck proves that is an artifact: leave-one-out majority with")
    P("  `max(sorted(c))` tie-breaking is wrong on EVERY item of an exactly balanced stratum, with")
    P("  or without signal. The claim rests only on the MAX over the seven policies.")
    P("  Seed-invariance of `unseen` is structural, not lucky: at this ratio the unseen non-age")
    P("  supply is taken ENTIRE, so there is no sampling freedom left for a seed to use.")
    P("  HONEST LIMIT: `unseen` is "
      f"{max(fj['unseen']['strata'].values()) / fj['unseen']['n']:.1%} `sex switch`. Off `age_band`")
    P("  and `sex`, the direction of a delta is nearly deterministic and size-balancing removes the")
    P("  stratum with the shortcut, so T2-unseen can only measure the one stratum that survives.")
    P("  `bnf switch` -- the other stratum measured as load-bearing (+0.286) -- has 0 balanced")
    P("  cells in `unseen` at every ratio scanned, and appears in `train` only.")
    J["composition"] = fj
    J["design"] = {"ratio": RATIO, "bal_key": BAL_KEY, "ind_match": IND_MATCH,
                   "drop_age": DROP_AGE, "seed": SEED}

    # ---------------- §5
    P("\n=== §5 IS A RATE DIFFERENCE DEFINABLE ON PHENOCAL COHORTS? ===")
    pc, meta = phenocal_edges()
    P(f"  PhenoCal_C2, all three official splits: {meta['rows']:,} items over "
      f"{meta['profiles']:,} distinct cohort profiles ({meta['unparsed']} unparsed).")
    P("  A pair is `one operation apart` on the axes CohortWorld's action set edits. The label is")
    P("  the environment's own rule -- t2_ceiling.separated, Wilson on the at-risk denominator,")
    P("  which PhenoCal publishes per item as metadata.n.")
    P(f"  {'operation':<16}{'pairs':>10}{'sharing a cell':>16}{'up':>9}{'down':>9}"
      f"{'indist':>11}{'balanced':>10}")
    for k in sorted(pc, key=lambda x: -pc[x]["balanced"]):
        d = pc[k]
        P(f"  {k:<16}{d['pairs']:>10,}{d['pairs_sharing_a_cell']:>16,}{d['up']:>9,}"
          f"{d['down']:>9,}{d['indistinguishable']:>11,}{d['balanced']:>10,}")
    P("  ANSWER: YES, definable, and richer than internally -- PhenoCal's `icd switch` carries")
    P(f"  {pc.get('icd switch', {}).get('balanced', 0):,} balanced separable cells where"
      f" CohortWorld's carries 0. That is a property of the RELEASE, not of biology: CohortWorld's")
    P("  at-risk counts have a median of 12, PhenoCal's are in the thousands, so the same Wilson")
    P("  test separates on one panel and cannot on the other. T2 is therefore NOT in T1's position")
    P("  -- T1 has no external split because a lattice POSITION has no PhenoCal referent at all.")
    P("  RECOMMENDATION: do not build it in this rebuild. Two costs, one of them the one this job")
    P("  was told to avoid:")
    P("   1. It cannot be SERVED without new code inside the content-hashed surface. `DeltaTaskset`")
    P("      resolves `_audit.from`/`to` through a CohortWorld lattice and a PhenoCal cohort has no")
    P("      state id, so an external T2 split needs a new taskset in `cohortworld_phenocal`, which")
    P("      `reward.semantic_hash()` binds -- a version bump and a re-certification pass.")
    P("   2. Both endpoints must be covered for the agent's two readings to be about the item's")
    P("      cohorts, and per-endpoint coverage is already the binding constraint on T3-external")
    P("      (c = 0.1119; 38.6% of PhenoCal_C2 test items ask about a record source the lattice")
    P("      does not contain). A T2-external score would be a second reading of that same limit.")
    P("  The supply above is recorded so the decision can be revisited without re-measuring it.")
    J["phenocal"] = {"meta": meta, "by_operation": pc, "verdict": "definable; deferred"}

    # ---------------- emit
    if a.emit:
        os.makedirs(a.emit, exist_ok=True)
        ok = collections.Counter()
        for sp in SPLITS:
            with open(f"{a.emit}/t2_delta_{sp}.jsonl", "w") as f:
                for d in final[sp]:
                    r = {k: v for k, v in d.items() if k != "_idx"}
                    f.write(json.dumps(r) + "\n")
                    ok[sp] += T.is_served(r)
        frm = {sp: {d["_audit"]["from"] for d in final[sp]} for sp in SPLITS}
        cross = {f"{x}|{y}": len(frm[x] & frm[y]) for i, x in enumerate(SPLITS)
                 for y in SPLITS[i + 1:]}
        assert set(cross.values()) == {0}, f"splits are not profile-disjoint: {cross}"
        P(f"\n=== EMITTED to {a.emit} ===")
        for sp in SPLITS:
            P(f"  t2_delta_{sp}.jsonl  {len(final[sp])} items, {ok[sp]} servable "
              f"({ok[sp] / len(final[sp]):.4f})")
            assert ok[sp] == len(final[sp])
        P(f"  cross-split from-cohort overlap: {cross}")
        J["emitted"] = {"dir": a.emit, "counts": {sp: len(final[sp]) for sp in SPLITS},
                        "servable": dict(ok), "cross_split_profile_overlap": cross}

    out = "\n".join(L)
    print(out)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w").write(out + "\n")
        json.dump(J, open(a.out.replace(".txt", ".json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
