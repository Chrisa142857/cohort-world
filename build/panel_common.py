"""
Shared utilities for the group-level multi-disease panel (Methods 1 & 2).

- load the extraction intermediate and pre-classify each patient's diseases as
  prevalent / incident / none relative to their index + horizon.
- readable names for risk-factor predictor groups and ICD-10 disease categories.
- coarse-timeline prompt rendering.
- per-(group, disease) rate aggregation with Wilson 95% CI + small-cell suppression.
"""
import math, os, pickle, sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_descriptions import describe, _LOOKUP

# readable names for the 18 risk-factor predictor groups
RF_NAMES = {
    "diabetes": "diabetes", "hypertension": "hypertension", "hyperlipidaemia": "high cholesterol",
    "obesity": "obesity", "smoking": "smoking", "alcohol_excess": "excess alcohol use",
    "depression": "depression", "hearing_loss": "hearing loss", "head_injury": "head injury",
    "stroke_tia": "stroke or transient ischaemic attack", "atrial_fibrillation": "atrial fibrillation",
    "ischaemic_heart": "ischaemic heart disease", "heart_failure": "heart failure",
    "ckd": "chronic kidney disease", "parkinsonism": "parkinsonism", "copd": "chronic obstructive pulmonary disease",
    "hypothyroid": "hypothyroidism", "anxiety": "anxiety",
}

MIN_N = 10        # minimum group members per (group, disease) cell
MIN_EVENTS = 5    # minimum events per cell (UKB small-number rule)

# code-axis generalisation: 18 risk-factor groups -> 6 clinical super-categories
SUPERGROUP = {
    "diabetes": "cardiometabolic", "hypertension": "cardiometabolic",
    "hyperlipidaemia": "cardiometabolic", "obesity": "cardiometabolic",
    "atrial_fibrillation": "cardiometabolic", "ischaemic_heart": "cardiometabolic",
    "heart_failure": "cardiometabolic", "ckd": "cardiometabolic",
    "stroke_tia": "cerebrovascular/neurological", "parkinsonism": "cerebrovascular/neurological",
    "head_injury": "cerebrovascular/neurological",
    "depression": "mental health", "anxiety": "mental health",
    "copd": "respiratory",
    "smoking": "lifestyle", "alcohol_excess": "lifestyle",
    "hypothyroid": "endocrine/sensory", "hearing_loss": "endocrine/sensory",
}
SUPER_NAMES = {s: s for s in set(SUPERGROUP.values())}
# members of each super-category (readable names), for explicit in-prompt definitions
SUPER_MEMBERS = {}
for _g, _s in SUPERGROUP.items():
    SUPER_MEMBERS.setdefault(_s, []).append({
        "diabetes": "diabetes", "hypertension": "hypertension", "hyperlipidaemia": "high cholesterol",
        "obesity": "obesity", "atrial_fibrillation": "atrial fibrillation",
        "ischaemic_heart": "ischaemic heart disease", "heart_failure": "heart failure",
        "ckd": "chronic kidney disease", "stroke_tia": "stroke or transient ischaemic attack", "parkinsonism": "parkinsonism",
        "head_injury": "head injury", "depression": "depression", "anxiety": "anxiety",
        "copd": "chronic obstructive pulmonary disease", "smoking": "smoking", "alcohol_excess": "excess alcohol use",
        "hypothyroid": "hypothyroidism", "hearing_loss": "hearing loss"}.get(_g, _g))


def _grid_named(p, code_mode):
    """Return (grid dict {key->min_age}, name map) at the requested code granularity."""
    if code_mode == "super":
        g = {}
        for grp, a in p["grid"].items():
            s = SUPERGROUP.get(grp, grp)
            if s not in g or a < g[s]:
                g[s] = a
        return g, SUPER_NAMES
    return p["grid"], RF_NAMES


def icd3_name(cat):
    return (_LOOKUP.get("icd10", {}).get(cat) or describe(cat, "icd10"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def age_band(age, width=5):
    lo = (age // width) * width
    return f"{lo}-{lo + width - 1}"


def load_intermediate(path):
    """Returns params + patients. Each patient keeps obs_years (follow-up after index),
    a prevalent set, and inc {disease -> years from index to first diagnosis} so incidence
    can be classified at ANY horizon at aggregation time."""
    d = pickle.load(open(path, "rb"))
    index = datetime.strptime(d["params"]["index_date"], "%Y-%m-%d")
    pats = {}
    for pid, v in d["patients"].items():
        prevalent, inc = set(), {}
        for cat, ds in v["dx"].items():
            dt = datetime.strptime(ds, "%Y-%m-%d")
            if dt <= index:
                prevalent.add(cat)
            else:
                inc[cat] = (dt - index).days / 365.25
        pats[pid] = {"sex": v["sex"], "imd": v["imd"], "age": v["age_at_index"],
                     "ageband": age_band(v["age_at_index"]), "gp": bool(v.get("gp_linked", True)),
                     "grid": v["grid"], "obs": v["obs_years"],
                     "prevalent": prevalent, "inc": inc}
    return d["params"], pats


# ---- coarse-timeline profile keys ----
def stage_label(edges):
    """human labels for a list of age cut points, e.g. [40,55] -> ['before 40','40-55','55+']."""
    labels = []
    prev = None
    for e in edges:
        labels.append(f"before {e}" if prev is None else f"{prev}-{e}")
        prev = e
    labels.append(f"{prev}+" if prev is not None else "any age")
    return labels


def bin_age(age, edges):
    for i, e in enumerate(edges):
        if age < e:
            return i
    return len(edges)


def profile_key(p, edges, include_demo=True, code_mode="specific"):
    """frozenset profile at a given time-binning + code granularity. edges=[] -> flat (no time);
    code_mode='super' -> generalise the 18 groups into 6 super-categories."""
    items, _ = _grid_named(p, code_mode)
    if edges:
        grid = frozenset((g, bin_age(a, edges)) for g, a in items.items())
    else:
        grid = frozenset(items)
    # NB: IMD is a constant placeholder (=3) for every patient in this extract -> dropped.
    import os as _os
    _imd = (p.get("imd"),) if _os.environ.get("STRATIFY_IMD") else ()
    demo = ((p["ageband"], p["sex"], p["gp"]) + _imd) if include_demo else None
    return (demo, len(edges), code_mode, grid)


# ---- rendering ----
def render_history(p, edges, code_mode="specific"):
    items, names = _grid_named(p, code_mode)
    labels = stage_label(edges)
    lines = []
    if edges:
        by_stage = defaultdict(list)
        for g, a in items.items():
            by_stage[bin_age(a, edges)].append(names.get(g, g))
        for i, lab in enumerate(labels):
            conds = sorted(by_stage.get(i, []))
            if conds:
                lines.append(f"- {lab}: {', '.join(conds)}")
        if not lines:
            lines.append("- no recorded conditions")
    else:
        conds = sorted(names.get(g, g) for g in items)
        lines.append(f"- {', '.join(conds) if conds else 'no recorded conditions'}")
    return "\n".join(lines)


def render_item(p, edges, disease_cat, horizon, code_mode="specific"):
    dname = icd3_name(disease_cat)
    src = "primary-care + hospital records" if p["gp"] else "hospital records only"
    clause = ("conditions shown as broad categories, grouped by the age at which they first appeared"
              if code_mode == "super" else
              "conditions grouped by the age at which they first appeared")
    import os as _os
    _dep = f"; Townsend deprivation quintile {p.get('imd')} of 5 (1=least deprived, 5=most deprived)" if _os.environ.get("STRATIFY_IMD") else ""
    demo = f"{p['sex'].lower()}; {src}{_dep}"
    hist = render_history(p, edges, code_mode)
    # explicit, self-contained definitions for any super-category that appears
    legend = ""
    if code_mode == "super":
        present = sorted(_grid_named(p, "super")[0].keys())
        defs = "; ".join(f"\"{s}\" means one or more of: {', '.join(sorted(SUPER_MEMBERS[s]))}"
                         for s in present)
        legend = (f"\nHere, each broad category means having ANY of its component conditions "
                  f"(not all): {defs}.\n")
    prompt = (
        f"Consider UK adults aged {p['ageband']} ({demo}) with the following recorded medical "
        f"history ({clause}):\n{hist}\n{legend}\n"
        f"Among such people who have NOT previously been diagnosed with this condition, what "
        f"percentage are diagnosed with \"{dname}\" (ICD-10 {disease_cat}) for the FIRST time "
        f"within the next {horizon} years?\n"
        f"Answer with a single percentage."
    )
    return prompt, dname


# ---- aggregation ----
def aggregate(groups, pats, horizon, coarseness_tag, edges_of, mode_of=lambda k: "specific",
              filterout=frozenset("ZRU"), med_fn=None, wg_fn=None):
    """groups: key -> list[pid] (members already eligible at this horizon).
    Emit one item per (group, disease) cell passing thresholds, at the given horizon.
    filterout: ICD-10 chapter letters to EXCLUDE as targets (default Z=factors-influencing-health,
    R=symptoms/abnormal-findings, U=special-use -> non-disease/administrative codes)."""
    items = []
    n_cells = n_supp = 0
    for key, members in groups.items():
        if len(members) < MIN_N:
            continue
        rep = pats[members[0]]            # representative for rendering (all share the profile)
        edges = edges_of(key); cmode = mode_of(key)
        meds = med_fn(members) if med_fn else None   # k-anon group-mean mediators (>=10 measured each)
        # tally prevalent / incident-within-horizon per disease across members
        prev_ct = defaultdict(int); inc_ct = defaultdict(int)
        for pid in members:
            mp = pats[pid]
            for c in mp["prevalent"]:
                prev_ct[c] += 1
            for c, yr in mp["inc"].items():
                if yr <= horizon:
                    inc_ct[c] += 1
        total = len(members)
        for cat in set(prev_ct) | set(inc_ct):
            if cat and cat[0] in filterout:      # drop non-disease/administrative chapters (Z/R/U)
                continue
            n_elig = total - prev_ct[cat]        # exclude prevalent from denominator
            events = inc_ct[cat]
            n_cells += 1
            if n_elig < MIN_N or events < MIN_EVENTS:
                n_supp += 1
                continue
            rate = events / n_elig
            lo, hi = wilson(events, n_elig)
            wgm = wg_fn(members, cat, horizon) if wg_fn else None   # within-group stratified mediation (Stage 2b)
            prompt, dname = render_item(rep, edges, cat, horizon, cmode)
            items.append({
                "prompt": prompt,
                "ground_truth": {"rate": round(rate, 4)},
                "metadata": {
                    "disease_icd10_3char": cat, "disease_name": dname,
                    "n": n_elig, "n_events": events,
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "horizon_years": horizon, "coarseness": coarseness_tag(key),
                    "profile_groups": sorted(rep["grid"].keys()),
                    **({"mediators": meds} if meds else {}),
                    **(wgm if wgm else {}),
                },
            })
    return items, n_cells, n_supp
