#!/usr/bin/env python3
"""CohortWorld cohort builder — GP-LINKED ONLY, cohorts built ONCE, item = cohort x horizon x ICD.

Three design choices, each ruling out a measured defect of the alternative:
  1. Cohorts are partitioned ONCE. Forming them SEPARATELY INSIDE each horizon loop makes a "cohort"
     not one object across horizons, so the count can only be reported per-horizon (summing them
     double-counts 5x). Here a horizon only filters which members are still observed.
  2. Only GP-linked subjects are kept. GP-unlinked subjects are 54.4% of the cohort, and for them an
     absent BNF history means UNKNOWN, not "no medication" — two different states rendered
     identically. With linked subjects only, every axis is observed for every person and absence is
     interpretable.
  3. The three-system history is rendered here. PC.aggregate's renderer reads p["grid"] — the 18
     CURATED risk factors — not the ICD/OPCS/BNF history the cohorts are built from; through it, 45.8%
     of prompts said "no recorded conditions", and ~5 genuinely different cohorts shared one prompt
     text.

COHORT DEFINITION (disjoint by construction)
  Two cohorts differ in at least one of: age band, sex, ICD history, OPCS history, BNF history —
  where each history axis carries both the CODE prefix and the AGE-BAND at which it first appeared.
  Disjointness is guaranteed by the greedy release: once a patient is placed, they are removed from
  the pool, so no patient can appear in two cohorts.

ADAPTIVE GRANULARITY (PhenoCal Method 2, reused)
  Start at the finest time binning; any profile shared by >= MIN_N patients is released at that
  fineness; the rest are generalised one level coarser and re-grouped. Time: decade -> 4stage ->
  3stage -> 2stage -> flat. Then two code-axis fallbacks (coarser prefix, then presence-only), which
  three code systems need because their profiles are far rarer than 18 binary flags.
"""
import argparse, json, os, pickle, sys, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clinical"))
import panel_common as PC

# (ICD hist, OPCS hist, BNF hist, outcome ICD chars). Every level is one the classification itself
# defines: ICD chapter/category, OPCS chapter/category/subcategory, BNF chapter/section/paragraph.
# A 2-character ICD or OPCS prefix is not a level -- ICD blocks are ranges like A15-A19, which no
# fixed-width prefix reproduces -- so generalising by character count invented resolutions that mean
# nothing to a clinician.
VERSIONS = {"C1": (1, 1, 2, 1),        # chapter / chapter / chapter        | outcome: chapter
            "C2": (3, 3, 4, 2),        # category / category / section       | outcome: 2-char
            "default": (3, 4, 6, 3)}   # category / subcategory / paragraph  | outcome: category

# C2's OUTCOME stays at 2 characters even though that is not a level ICD-10 defines (its blocks are
# ranges such as A15-A19, which no fixed-width prefix reproduces). It is kept because PhenoCal-C2's
# 70,378 public items are posed at 2 characters, and matching them is what lets a CohortWorld agent
# be scored on the same questions as the PhenoCal leaderboard. The HISTORY axes are unaffected and
# stay on real levels; the approximation is confined to the prediction target, where comparability
# with the existing leaderboard is worth more than hierarchical purity.
TIME_LEVELS = [("decade", [30, 40, 50, 60]), ("4stage", [30, 45, 58]),
               ("3stage", [40, 55]), ("2stage", [45]), ("flat", [])]

ICD_CH = {"A": "infectious disease", "B": "infectious disease", "C": "cancer", "D": "blood/neoplasm",
          "E": "endocrine/metabolic", "F": "mental/behavioural", "G": "nervous system",
          "H": "eye/ear", "I": "circulatory", "J": "respiratory", "K": "digestive",
          "L": "skin", "M": "musculoskeletal", "N": "genitourinary", "O": "pregnancy",
          "P": "perinatal", "Q": "congenital", "S": "injury", "T": "injury/poisoning"}
OPCS_CH = {"A": "nervous system", "B": "endocrine/breast", "C": "eye", "D": "ear",
           "E": "respiratory tract", "F": "mouth", "G": "upper digestive", "H": "lower digestive",
           "J": "liver/pancreas", "K": "heart", "L": "arteries/veins", "M": "urinary",
           "N": "male genital", "P": "lower female genital", "Q": "upper female genital",
           "R": "pregnancy-related", "S": "skin", "T": "soft tissue", "U": "diagnostic imaging",
           "V": "bones/joints of skull-spine", "W": "other bones/joints"}
BNF_CH = {"01": "gastro-intestinal", "02": "cardiovascular", "03": "respiratory",
          "04": "central nervous system", "05": "infections", "06": "endocrine",
          "07": "obstetrics/gynaecology", "08": "malignant disease/immunosuppression",
          "09": "nutrition/blood", "10": "musculoskeletal", "11": "eye", "12": "ENT",
          "13": "skin", "14": "immunological/vaccines", "15": "anaesthesia"}


def sysname(s, code):
    """A readable name for a history code at ANY granularity.

    The chapter tables are keyed by the top level of each system (1-char ICD/OPCS, 2-char BNF), so
    a plain table lookup returns the raw code for everything finer -- which is what C2 and default
    use. Their prompts would come out as "- procedures: W9", a token no reader can interpret, the
    same defect as the "*" rung but affecting every finer level. A finer code is therefore named by
    the chapter it belongs to, with the code kept so the agent can still act on it.
    """
    if code == "*":
        return "any (unspecified)"
    if s == "I":
        chap = ICD_CH.get(code[0], "other")
    elif s == "O":
        chap = OPCS_CH.get(code[0], "other procedure")
    else:
        chap = BNF_CH.get(code[:2], "other medicine")
    top = 1 if s in ("I", "O") else 2
    return chap if len(code) <= top else f"{chap} [{code}]"


def render(demo, grid, edges):
    ageband, sex, _ = demo
    lab = PC.stage_label(edges) if edges else None
    sysfull = {"I": "diagnoses", "O": "procedures", "B": "prescriptions"}
    by = collections.defaultdict(list)
    for (s, code, band) in sorted(grid):
        nm = sysname(s, code)
        # `band` is an INDEX into lab, so the guard must be a range check. A membership test
        # `band in lab` compares the index against the label STRINGS and is always False, which
        # silently drops the age-at-first-occurrence qualifier from every prompt.
        show = lab and isinstance(band, int) and 0 <= band < len(lab)
        by[s].append(f"{nm}" + (f" (first recorded {lab[band]})" if show else ""))
    lines = [f"Consider UK adults aged {ageband} ({'female' if str(sex).upper().startswith('F') else 'male'}; "
             f"linked primary-care and hospital records) with the following recorded history:"]
    for s in ("I", "O", "B"):
        lines.append(f"- {sysfull[s]}: " + ("; ".join(by[s]) if by[s] else "none recorded"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=list(VERSIONS))
    ap.add_argument("--hx", default="output_panel/cohortworld_hx.pkl")
    ap.add_argument("--inter", default="output_panel/panel_intermediate.pkl")
    ap.add_argument("--filterout", default="ZRU")
    ap.add_argument("--horizons", default="1,2,3,5,10")
    ap.add_argument("--out", default="output_panel")
    a = ap.parse_args()
    HOR = [int(x) for x in a.horizons.split(",")]
    FILT = frozenset(a.filterout.upper())
    *gran0, OUT_N = VERSIONS[a.version]
    gran0 = tuple(gran0)

    _, pats = PC.load_intermediate(a.inter)
    hx = pickle.load(open(a.hx, "rb"))
    # GP-LINKED ONLY. Judge by the real data, not the flag alone: 1,527 subjects flagged unlinked do
    # carry prescriptions, so the flag disagrees with gp_scripts. Require the flag AND require that the
    # person is not silently missing from the primary-care extract.
    linked = {p for p, v in pats.items() if v["gp"]}
    pats = {p: v for p, v in pats.items() if p in linked}
    print(f"[cw2] GP-linked subjects: {len(pats):,}", flush=True)

    if OUT_N < 3:
        for pid, mp in pats.items():
            prev = {c[:OUT_N] for c in mp["prevalent"]}
            inc = {}
            for c, yr in mp["inc"].items():
                k = c[:OUT_N]
                if k not in inc or yr < inc[k]:
                    inc[k] = yr
            mp["prevalent"] = prev
            mp["inc"] = {k: v for k, v in inc.items() if k not in prev}

    HX = {p: {"I": hx["icd"].get(p, {}), "O": hx["opcs"].get(p, {}), "B": hx["bnf"].get(p, {})}
          for p in pats}

    # ---------- PHASE 1: partition the population into DISJOINT cohorts, ONCE ----------
    coarser = tuple(max(1, g - 1) for g in gran0)
    CODE_LEVELS = [gran0, coarser, (0, 0, 0)]
    unreleased = set(pats)
    cohorts, meta = {}, {}
    cid = 0
    for gran in CODE_LEVELS:
        for tag, edges in TIME_LEVELS:
            if not unreleased:
                break
            buckets = collections.defaultdict(list)
            for pid in unreleased:
                p = pats[pid]
                demo = (p["ageband"], p["sex"], p["gp"])
                grid = []
                for s, n in (("I", gran[0]), ("O", gran[1]), ("B", gran[2])):
                    for code, age in HX[pid][s].items():
                        grid.append((s, code[:n] if n else "*",
                                     PC.bin_age(age, edges) if edges else ""))
                buckets[(demo, len(edges), gran, frozenset(grid))].append(pid)
            for key, members in buckets.items():
                if len(members) >= PC.MIN_N:
                    cohorts[cid] = members
                    meta[cid] = dict(demo=key[0], edges=edges, gran=gran,
                                     grid=key[3], tag=f"{tag}|icd{gran[0]}-opcs{gran[1]}-bnf{gran[2]}")
                    for pid in members:
                        unreleased.discard(pid)
                    cid += 1
    placed = len(pats) - len(unreleased)
    print(f"[cw2] DISJOINT cohorts: {len(cohorts):,} | placed {placed:,}/{len(pats):,} "
          f"({100*placed/len(pats):.1f}%) | suppressed {len(unreleased):,}", flush=True)
    assert sum(len(m) for m in cohorts.values()) == placed, "cohorts overlap"

    # ---------- PHASE 2: item = cohort x horizon x ICD ----------
    items = []
    for H in HOR:
        n_c = 0
        for c, members in cohorts.items():
            mem = [p for p in members if pats[p]["obs"] >= H]
            if len(mem) < PC.MIN_N:
                continue
            n_c += 1
            prev_ct = collections.Counter(); inc_ct = collections.Counter()
            for pid in mem:
                for x in pats[pid]["prevalent"]:
                    prev_ct[x] += 1
                for x, yr in pats[pid]["inc"].items():
                    if yr <= H:
                        inc_ct[x] += 1
            body = render(meta[c]["demo"], meta[c]["grid"], meta[c]["edges"])
            for cat in set(prev_ct) | set(inc_ct):
                if not cat or cat[0] in FILT:
                    continue
                n_elig = len(mem) - prev_ct[cat]
                ev = inc_ct[cat]
                if n_elig < PC.MIN_N or ev < PC.MIN_EVENTS:
                    continue
                lo, hi = PC.wilson(ev, n_elig)
                dname = PC.icd3_name(cat) if len(cat) == 3 else (ICD_CH.get(cat[0], "other") + f" [{cat}]")
                items.append({
                    "prompt": f"{body}\n\nAmong such people who have NOT previously been diagnosed with "
                              f'this condition, what percentage are diagnosed with "{dname}" '
                              f"(ICD-10 {cat}) for the FIRST time within the next {H} years?\n"
                              f"Answer with a single percentage.",
                    "ground_truth": {"rate": round(ev / n_elig, 4)},
                    "metadata": {"cohort_id": c, "disease_icd10_3char": cat, "disease_name": dname,
                                 "n": n_elig, "n_events": ev, "ci95": [round(lo, 4), round(hi, 4)],
                                 "horizon_years": H, "coarseness": meta[c]["tag"]}})
        print(f"[cw2] H={H:>2}y cohorts used {n_c:,} | items so far {len(items):,}", flush=True)

    out = os.path.join(a.out, f"CohortWorld_v2_{a.version}.jsonl")
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    dis = len({it["metadata"]["disease_icd10_3char"] for it in items})
    used = len({it["metadata"]["cohort_id"] for it in items})
    json.dump({"version": a.version, "gp_linked_only": True,
               "hx_granularity": dict(zip(("icd", "opcs", "bnf"), gran0)), "outcome_icd_chars": OUT_N,
               "subjects": len(pats), "cohorts_disjoint": len(cohorts), "cohorts_used": used,
               "placed_pct": round(100*placed/len(pats), 2),
               "horizons": HOR, "items": len(items), "diseases": dis, "k_min": PC.MIN_N},
              open(os.path.join(a.out, f"CohortWorld_v2_{a.version}_stats.json"), "w"), indent=1)
    print(f"\n[cw2] 写入 {out}\n  不相交人群 {len(cohorts):,} (被用到 {used:,}) | items {len(items):,} | 疾病 {dis}")


if __name__ == "__main__":
    main()
