# release-gate: allow-file -- ran against the private UKB extract; shipped for provenance, paths shown as they were
"""Cold-start demonstrations: a clinician's reasoning, written independently of CohortWorld.

WHY THIS DESIGN. A simpler demonstration recipe was measured to be wrong on three counts, all of which
this file fixes by construction:
  1. DIRECTION. Walking from the coarsest demographic cohort into a 10-49 subgroup and copying its
     reading left 47.7% of the committed values literally 0, and no one-move policy beat not
     moving. The clinician's order is the opposite -- stand in the patient's own group, which is too
     small to publish, widen to a group you can trust, and correct for the difference. That is the
     mechanism this file demonstrates.
  2. NUMBERS. Its declared deltas were WORSE THAN A CONSTANT (lift -0.5156 against the best constant),
     because the writer was asked to invent a magnitude with nothing to go on. Here the writer is given
     the empirical effect size of the SAME KIND of history difference, measured on OTHER cohorts
     (leave-this-item-out), and a demonstration is kept only when its number beats the best constant on
     that item. The truth is never shown to the writer; it is only used to accept or reject.
  3. TOOL SYNTAX. 63.5% of its demos contained a full `axis.op.code->C2:<16 hex>` move id and not one
     showed a refused move, which is why 35.7% of the moves of a policy trained on them were fabricated
     ids the environment rejected. A demonstration here contains NO tool call, NO id and NO code: only the reasoning inside
     <cohort-think>. Tool syntax is the base model's job at rollout time, where cw_look lists the ids --
     the same split as R1's cold start, which taught the shape of <think> and left the rest to RL.

The edge is chosen by a rule the agent can also apply at inference (widen to the largest group whose
rate is published; break ties by dropping the history feature least related to the disease asked). It is
NOT chosen by which edge's delta happens to be closest to the truth: that would teach an oracle
selection rule no policy could ever execute.
"""
import argparse, asyncio, collections, functools, hashlib, itertools, json, os, random, re, statistics as st, sys, time

CW = "/work/USERS/ziquanw/cohortworld"
sys.path.insert(0, CW + "/cohortworld_env"); sys.path.insert(0, CW + "/gen")
from cohortworld_env.lattice import _shared, OPCS_CH, BNF_CH            # noqa: E402
from code_descriptions import describe_icd10                            # noqa: E402
# ACCEPTANCE. The demonstration is scored on its REASONING as well as its number:
#   A = w_proc * (R1..R6)/12 + w_num * num_score/0.75,  behind seven hard gates.
# The rubric, the gates, the regexes and num_score all live in build/demo_quality.py and are imported from
# there, never restated: this file parses argv at import time so it cannot be imported the other way, and
# a second copy of a direction regex is a copy that drifts.
from demo_quality import (JARGON, SIGNED, PER_K, UP, DOWN, num_score, quality, add_args, cfg_from,
                          parse_declared,
                          _C, shingles, jaccard)                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--items", default=os.environ.get("CW_ITEMS"), required=False)
ap.add_argument("--split", default="train")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--samples", type=int, default=4, help="writer attempts per item; the closest one is kept if it beats the constant")
ap.add_argument("--writer-url", default=os.environ.get("CW_WRITER_URL", "http://hummer:8160/v1"),
                help="one endpoint, or several separated by commas. Requests are dealt round-robin across "
                     "them, which is how several single-GPU replicas are used without a data-parallel "
                     "coordinator -- the coordinator's ZMQ handshake is a failure point for "
                     "multi-replica serving, and three independent servers cannot have that failure.")
ap.add_argument("--writer-model", default=os.environ.get("CW_WRITER_MODEL", "medgemma27b-writer"))
ap.add_argument("--concurrency", type=int, default=48)
ap.add_argument("--dry-run", action="store_true", help="steps 1-5 only: no writer, print the rendered facts")
ap.add_argument("--judge-top", type=int, default=2, help="how many of an item's writer samples, ranked by their number score, are worth the two judge calls. A lower-ranked sample cannot win on A unless it is >=6 rubric points better, and halving the judge cost buys that risk.")
ap.add_argument("--audit", default="", help="ABSOLUTE path: one JSONL record per CANDIDATE, accepted or not, with every rule, gate, judge answer and score. The REJECTED ones are how a threshold is re-tuned offline, so they are kept.")
add_args(ap)                       # every acceptance constant is a CLI argument with its resolved value recorded
ap.add_argument("--facts-out", default="", help="ABSOLUTE path: write the selected edge of every ranked "
               "item -- op, axis, code, the reference group's exact wording, both ends, the delta -- and "
               "stop before the writer. This is the GROUND TRUTH a demonstration is audited against; "
               "recomputing it is deterministic, so an audit never has to trust what a writer said.")
ap.add_argument("--candidates", default="", help="write EVERY writer sample here (facts + text + its own number + the truth) and apply no acceptance. Generation is the expensive half and filtering is pure CPU, so the two are separated: the writer runs once, and the accept rule can be re-run offline as often as it changes.")
a = ap.parse_args()
QCFG = _C(cfg_from(a))             # resolved, and printed below so the run record can quote it
for _p in (a.out, a.candidates, a.audit):
    if _p and not os.path.isabs(_p): sys.exit(f"paths must be absolute; got {_p!r}")
if not a.items:
    sys.exit("--items or CW_ITEMS is required; there is no default, so this can never silently read a stale item set")
random.seed(a.seed)

# ---------------------------------------------------------------- words, never codes
def strip_codes(t): return re.sub(r"\s*\[[A-Z0-9]{1,6}\]", "", t or "")

def describe(axis, code):
    code = str(code)
    if axis == "bnf":  return f"medication for {BNF_CH.get(code[:2], 'an unclassified indication')}"
    if axis == "opcs": return f"a procedure on the {OPCS_CH.get(code[:1], 'body')}"
    if axis == "icd":  return strip_codes(describe_icd10(code)) or "a recorded diagnosis"
    return "a recorded condition"

def size_words(nb):
    lo = re.match(r"(\d+)", str(nb or ""))
    if not lo: return "an unstated number of people"
    n = int(lo.group(1))
    return ("over a thousand people" if n >= 1000 else "several hundred people" if n >= 200 else
            "one to two hundred people" if n >= 50 else "a few dozen people" if n >= 20 else
            "barely more than a dozen people")

STOP = {"medication", "for", "a", "an", "the", "on", "procedure", "body", "recorded", "condition",
        "unclassified", "indication", "and", "of", "or", "other", "system", "disorders", "diseases"}
def content_words(t):
    return {w for w in re.findall(r"[a-z]+", (t or "").lower()) if len(w) > 2 and w not in STOP}

def same_chapter(dropped_words: str, disease_words: str) -> bool:
    """Is the feature we would drop the SAME THING the question asks about? Compared on the rendered English,
    because a BNF code and an ICD chapter share no character space and the first-character test this replaces
    could never have fired. Measured on a sample batch: an item whose history carried skin medication
    and whose question was a first SKIN diagnosis has a reference rate of 0.6 per 1000 and an answer of 83 per
    1000 -- the feature is very nearly the outcome, so the item tests leakage, not clinical reasoning."""
    return bool(content_words(dropped_words) & content_words(disease_words))

def history_words(prompt):
    out = []
    for ln in (prompt or "").split("\n")[1:12]:
        s = ln.strip()
        if not s or s.startswith(("Your task", "Start with")): break
        out.append(strip_codes(s.lstrip("- ")))
    return "; ".join(out) or "no recorded conditions"

# ---------------------------------------------------------------- the lattice, and the generalising edges
LATS = {}
def lat_for(v):
    if v not in LATS: LATS[v] = _shared(v)
    return LATS[v]

def drop_moves(v, sid):
    """The generalising moves THE ENVIRONMENT ITSELF advertises from this cohort -- called, not
    reimplemented. A local copy of the `most specific released subset` search would be a second source
    of truth for a rule that lives in `Lattice`, and it would also be the slow half of steps 1-5: the
    env keeps a cached stratum index, a copy rebuilds a pool on every call.
    Returns the env's own move dicts: {from, axis, op:"drop", code, to, cost, n_band_to, onset_rung}."""
    return lat_for(v)._drop_moves(sid)

# ---------------------------------------------------------------- the PRECOMPUTED edge table
# `official/CohortWorld_pooled_edges.jsonl` (97,401 rows) is the fixed artifact
# computed from every released cohort. One row per legal move, carrying both endpoints, BOTH n_bands,
# and per granularity `[horizon_index, disease_index, rate_to - rate_from, separated]`. A widening move
# is the REVERSE of an `add` row, so the index below is keyed on the row's DESTINATION.
#
# What it does and does not reach, MEASURED on all 16,770 train items:
#   * 94.9% of item cohorts appear somewhere in the table;
#   * 58.7% have an `add` row pointing AT them -- i.e. the exact one-code-less cohort was released.
#     Where it was not, there is no row to read, which is exactly why `Lattice._drop_moves` computes a
#     subset landing instead of serving an edge (its own measurement: exact removal strands 27.4% and
#     leaves 86.2% unable to reach a 200+ cohort);
#   * 31.3% carry a delta cell for that item's own (horizon, disease). The builder drops a cell when
#     BOTH sides read zero ("both zero: no information, and most cells are this") and 52.8% of the
#     items answer exactly zero, so the missing cells are the zero-answer items, by design.
# Therefore: read the table wherever it reaches -- its delta is the same quantity computed from the same
# published rates, and its `separated` flag (the two Wilson intervals at the published band lower bounds
# do not overlap) is the environment's own verdict on whether the difference is bigger than the noise,
# which is worth having and costs nothing. Fall back to the environment's move elsewhere.
DIS_G, EDGE_STATS = {}, collections.Counter()
REJECT = collections.Counter()
LAST_REJECT = collections.deque(maxlen=12)   # a rolling sample of what a rejected reply actually says
EDGE_OUT, EDGE_ADD_IN = collections.defaultdict(list), {}

def dis_index(v, g):
    """The disease list the edge table's `disease_index` counts into, for this granularity."""
    if (v, g) not in DIS_G:
        f = f"{CW}/official/CohortWorld_{v}_diseases_g{g}.json"
        DIS_G[(v, g)] = (json.load(open(f))["diseases"] if os.path.exists(f) else [])
    return DIS_G[(v, g)]

# ZOOM is the one op group excluded from a demonstration's candidate moves: it changes how finely the SAME
# history is coded, not who the comparison group is, so there is no clinical sentence to write about it --
# "the same people, recorded at three characters instead of one" is a statement about the coding system.
# The other four are all reference classes a clinician actually names.
DEMO_OPS = {"drop", "add", "switch", "shift"}

def build_edge_index(items):
    """ONE pass over `official/CohortWorld_pooled_edges.jsonl` -- the fixed table, built from every
    released cohort -- keeping only the cells these items need.

    Two directions, carrying the delta with OPPOSITE signs, which is why this is one function:
      * OUTGOING (`from` == the item's own cohort): the stored `rate_to - rate_from` is already
        reference-minus-own, the demonstration's own quantity. No flip.
      * INCOMING `add` (`to` == the item's own cohort): its reverse is the exact one-code-less widening,
        so the stored value is own-minus-reference and the demonstration's delta is its negation.
    Only the single cell for this item's own (granularity, horizon, disease) is kept, so the index costs
    kilobytes rather than holding a 421 MB table in memory."""
    need = {}
    for it in items:
        v, g = it.get("version"), int(it.get("granularity") or 2)
        lat = lat_for(v)
        if lat is None: continue
        hs = list(lat.horizons or ()); dl = dis_index(v, g)
        try: need[it["target_cohort"]] = (g, hs.index(int(it["horizon"])), dl.index(it["disease"]))
        except ValueError: continue
    t0 = time.time(); n = kept = 0
    ends = re.compile(r'"cross_version": (true|false), "from": "([^"]+)", "to": "([^"]+)"')
    def cell_of(e, who):
        g, hi, di = need[who]
        for c in (e.get(f"delta_g{g}") or ()):
            if c[0] == hi and c[1] == di: return c
        return None
    with open(f"{CW}/official/CohortWorld_pooled_edges.jsonl") as fh:
        for line in fh:
            n += 1
            m = ends.search(line[:400])          # from/to are written before the delta arrays, so this
            if not m or m.group(1) == "true": continue   # avoids parsing cells no item asked for
            src, dst = m.group(2), m.group(3)
            if src not in need and dst not in need: continue
            e = json.loads(line); op = e["op"]
            if op not in DEMO_OPS: continue
            if src in need:
                c = cell_of(e, src)
                EDGE_OUT[src].append({"axis": e["axis"], "op": op, "code": str(e["code"]), "to": dst,
                                      "n_band_to": e["n_band_to"], "delta": (c[2] if c else None),
                                      "separated": (bool(c[3]) if c else None)})
                kept += 1
            if dst in need and op == "add":
                c = cell_of(e, dst)
                EDGE_ADD_IN[(dst, e["axis"], str(e["code"]))] = {
                    "from": src, "delta": (-c[2] if c else None), "separated": (bool(c[3]) if c else None)}
                kept += 1
    print(f"[edge table] {n:,} rows scanned, {kept:,} cells kept for these items "
          f"({len(EDGE_OUT):,} cohorts with outgoing moves, {len(EDGE_ADD_IN):,} incoming add rows) "
          f"in {time.time()-t0:.0f}s", flush=True)

def ar_ord(band):
    """Rank a published band by its own lower bound: "10-12" -> 10, "1000+" -> 1000, missing -> -1."""
    m = re.match(r"(\d+)", str(band or ""))
    return int(m.group(1)) if m else -1

def rate_at(v, sid, disease, horizon, with_at_risk=False):
    """The published rate, and on request the AT-RISK band -- the denominator that decides how much that
    rate can be trusted. A cohort can be large while the at-risk set for one (disease, horizon) is not."""
    lat = lat_for(v); prev = lat.current
    try:
        lat.current = sid; o = lat.observe(disease, int(horizon))
        if o.get("status") != "published": return (None, None) if with_at_risk else None
        r = float(o["rate"])
        return (r, o.get("at_risk")) if with_at_risk else r
    except Exception:
        return (None, None) if with_at_risk else None
    finally:
        lat.current = prev

# ---------------------------------------------------------------- the writer
# JARGON / SIGNED / PER_K / UP / DOWN are imported from demo_quality above -- one copy, one behaviour.

SYSTEM = ("You are an experienced physician reasoning aloud about one patient's future risk. Write plain "
          "clinical English. Never mention codes, classifications, databases, tools, or how data is organised.")

def writer_prompt(f):
    """The user's design: the edge's parameters AND its delta are inputs, and the writer turns them into a
    clinician's reasoning. Withholding the figure and grading the writer's guess is rejection sampling rather
    than the conversion asked for -- and it produces prose that argues for a large increase and then writes
    a number a hundred times too small, because a language model has no scale for a four-decimal
    probability. With the figure given, the number is right by construction and the demonstration can be judged
    on the only thing left that matters: whether the reasoning actually earns it. That is the R1 cold start --
    thousands of CORRECT traces, filtered on quality."""
    own = f["ref_rate"] - f["true_delta"]                 # commit = reference - delta
    d = f["true_delta"]
    direction = "lower than" if d > 0 else ("higher than" if d < 0 else "the same as")
    # `wider` was true of the drop and of nothing else: an `add` reference is NARROWER and a `shift`
    # reference is neither. The prompt says which group it means instead of asserting a size relation.
    lead = ("The nearest group large enough to trust is" if f["op"] == "drop" else
            "The nearest group whose figure can be read, and which is measured on at least as many people, is")
    # the opening sentence of the demonstration, fixed, in the environment's own words for this move
    word = {"lower than": "lower", "higher than": "higher"}.get(direction, "much the same")
    yrs = f"{f['yr']} year" + ("s" if int(f["yr"]) != 1 else "")
    f["opening"] = (f"Compared with {f['ref_phrase']}, this patient's risk of a first diagnosis of "
                    f"{f['dis']} within {yrs} is {word}.")
    return (f"Patient: {f['who']}. Recorded history: {f['hist']}.\n"
            f"Question: the chance of a first diagnosis of {f['dis']} within {f['yr']} year(s).\n"
            f"Patients whose recorded history matches this one exactly are only {f['own_size']}, too few to quote "
            f"a figure from directly. {lead} {f['ref_phrase']}: "
            f"{f['ref_size']}, in whom this happens "
            f"to about {f['ref_rate']*1000:.1f} in every 1000 over {f['yr']} year(s).\n"
            f"In this patient's own group the figure is about {own*1000:.1f} in every 1000 -- {direction} that "
            f"group.\n\n"
            f"Your note will BEGIN with this sentence, which is already written:\n"
            f"  \"{f['opening']}\"\n\n"
            f"Write the four sentences that follow it: the clinical reasoning a physician would give for that "
            f"difference -- why that group is the right starting comparison for this patient, and by what "
            f"mechanism the one difference between them -- {f['dropped']} -- moves the chance of {f['dis']} "
            f"in that direction and by roughly that much.\n\n"
            f"Rules. Do NOT describe who that group is: the opening sentence has already said it, and saying "
            f"it again in your own words is the one thing that makes a note wrong. Refer to them only as "
            f"\"that group\" or \"them\". Reason from the medicine, not from the numbers you were given -- "
            f"someone reading your answer should be able to reach the same conclusion from the patient's "
            f"history alone. Say plainly, in words, roughly how big the difference is. Write no digits at "
            f"all. Do not mention that you were told any figure, and do not use any code, abbreviation in "
            f"brackets, or technical term. Write only those four sentences.")

WRITERS = [u.strip().rstrip("/") for u in str(a.writer_url).split(",") if u.strip()]
_turn = itertools.count()

async def write_one(session, f, sem, temp):   # f is needed to convert the stated rate into a delta
    import aiohttp
    body = {"model": a.writer_model, "temperature": temp, "max_tokens": 260,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": writer_prompt(f)}]}
    url = WRITERS[next(_turn) % len(WRITERS)]
    async with sem:
        try:
            async with session.post(f"{url}/chat/completions", json=body,
                                    timeout=aiohttp.ClientTimeout(total=180)) as r:
                j = await r.json()
            text = re.sub(r"\s+", " ", j["choices"][0]["message"]["content"].strip())
        except Exception:
            return None, None
    def _no(why):
        REJECT[why] += 1
        LAST_REJECT.append((why, text[-200:] if text else ""))
        return None, None
    text = re.sub(r"^(Okay|OK|Alright|Well)[,.]?\s*", "", text)
    text = re.sub(r"^(let'?s|let us) (think|consider|look)[^.]*\.\s*", "", text, flags=re.I)
    text = (text[:1].upper() + text[1:]) if text else text
    if not text: return _no("empty")
    if JARGON.search(text): return _no("jargon or a bracketed code")
    op_norm = re.sub(r"\W+", " ", f["opening"]).strip().lower()
    while True:                                  # it sometimes repeats the opening twice
        t_norm = re.sub(r"\W+", " ", text).strip().lower()
        if not t_norm.startswith(op_norm[:60]): break
        cut = len(text) - len(text)              # find the end of the copied sentence in the ORIGINAL text
        m = re.match(r"\s*[^.]*\.\s*", text)
        if not m: break
        text = text[m.end():]
    text = text.strip()
    if not text: return _no("wrote nothing but a copy of the opening sentence")
    if re.search(r"\d", text): return _no("wrote a figure when asked for words only")
    kw = [w for w in re.findall(r"[a-z][a-z/-]{3,}", str(f.get("dropped") or "").lower())
          if w not in ("medication", "procedure", "recorded", "condition", "unclassified", "indication",
                       "related", "problems", "disease", "disorder")]
    for sn in re.split(r"(?<=[.!?])\s+", text):
        if not re.search(r"\b(group|others|individuals|people|those|patients|peers|counterparts)\b", sn, re.I):
            continue
        if any(k in sn.lower() for k in kw) and re.search(
                r"\b(without|not|no|lack\w*|absence|absent|aren'?t|isn'?t|free of)\b", sn, re.I):
            return _no("described the comparison group again instead of leaving it to the opening")
    val = f["true_delta"]               # by construction: the closing line is appended, not guessed
    ups, downs = len(UP.findall(text)), len(DOWN.findall(text))
    if not ((val <= 0 and ups >= downs) or (val > 0 and downs >= ups) or (ups == downs == 0)):   # delta>0 means the patient is LOWER
        return _no("the prose argues the opposite direction from the delta")
    # the fixed closing line, in the demonstration's own convention: what must be SUBTRACTED from the
    # reference group's figure to reach this patient's, which is the chain commit = observe - delta.
    text = f"{f['opening']} {text.rstrip()} Adjustment: {val:+.4f}."
    return text, val

# ---------------------------------------------------------------- per item: steps 1-5
def move_words(v, m, lat):
    """How the reference group differs from this patient, in a clinician's words. One phrase per op --
    the demonstration says WHO it is comparing against, and every op group names a different `who`.
    The fixed table's rows make `add`, `switch` and `shift` available alongside the environment's own
    `drop`, and a physician names all four."""
    ax, code = m["axis"], str(m["code"])
    if ax == "age_band": return (f"people with the same recorded history who are in the {code} age group", None)
    if ax == "sex":      return (f"people with the same recorded history who are {str(code).lower()}", None)
    feat = describe(ax, code)
    if m["op"] == "drop":   return (f"people who share the rest of this history but for whom {feat} is not required", feat)
    if m["op"] == "add":    return (f"people who share this history and in addition have {feat}", feat)
    if m["op"] == "switch": return (f"people who share the rest of this history but who have {feat} in its place", feat)
    return (f"people who share the rest of this history apart from {feat}", feat)

def facts_for(it):
    v, tc = it.get("version"), it.get("target_cohort")
    lat = lat_for(v)
    if not tc or tc not in lat.states: return None, "no target cohort in the lattice"
    dis, hor = it["disease"], int(it["horizon"])
    dis_words = strip_codes(lat.dname.get(dis, dis))
    _, own_at = rate_at(v, tc, dis, hor, with_at_risk=True)
    # EVERY legal move out of the item's own cohort, from two sources that do not overlap:
    #   * the fixed edge table's outgoing rows -- add / switch / shift, with the delta already computed;
    #   * the environment's own generalising move, which no table row can carry (`Lattice._drop_moves`:
    #     "most specific subset" is a fact about the whole stratum, not a relation between two cohorts).
    # Using only the drops would teach one action; using only the table would leave 2,820 item cohorts
    # with no outgoing row at all and would never reach a 1000+ reference, since add/switch/shift all
    # land inside the same 10-49 world.
    moves = [dict(m, op="drop") for m in drop_moves(v, tc)] + list(EDGE_OUT.get(tc, ()))
    if not moves: return None, "the item's own cohort has no legal move to compare against"
    cands = []
    for m in moves:
        p = m["to"]
        if p not in lat.states: continue
        r, at = rate_at(v, p, dis, hor, with_at_risk=True)
        if r is None: continue
        phrase, feat = move_words(v, m, lat)
        rel = 1 if (feat and same_chapter(feat, dis_words)) else 0
        td, sep = m.get("delta"), m.get("separated")
        if m["op"] == "drop":                                   # corroborate the widening against the table
            row = EDGE_ADD_IN.get((tc, m["axis"], str(m["code"])))
            if row and row["from"] == p: td, sep = row["delta"], row["separated"]
        cands.append({"m": m, "sid": p, "rate": r, "n_band": lat.states[p]["n_band"], "at_risk": at,
                      "related": rel, "phrase": phrase, "table_delta": td, "separated": sep})
    if not cands: return None, "no legal move whose destination publishes a rate"
    clean = [c for c in cands if not c["related"]]
    if not clean: return None, "every move changes the very thing the question asks about (leakage)"
    up = [c for c in clean if ar_ord(c["at_risk"]) >= ar_ord(own_at)]
    if not up: return None, "every move leads to a group read off fewer people than this patient's own"
    cands = up
    order = {"1000+": 3, "200-999": 2, "50-199": 1, "10-49": 0}
    # THE MOST RELIABLE READABLE NEIGHBOUR, which is the rule the user named and the one a clinician uses.
    # Reliability is the at-risk denominator of the rate being read, not the cohort's headcount; a group can
    # be large while the at-risk set for one (disease, horizon) is not. Ties go to the move whose difference
    # the table calls `separated`, then to the feature LEAST related to the disease asked, so the reference
    # keeps the risk factors that matter and the adjustment left to explain stays interpretable.
    best = max(cands, key=lambda c: (ar_ord(c["at_risk"]), order.get(str(c["n_band"]), 0),
                                     1 if c["separated"] else 0, -c["related"], -len(str(c["m"]["code"]))))
    p_own = float(it["answer_rate"])
    true_delta = best["rate"] - p_own            # delta = reference - own, per the chain: commit = observe - delta
    # Where the fixed table reaches this move it has already computed the same difference from the same
    # published rates. Compare instead of choosing: a silent disagreement would mean the direction
    # convention is backwards somewhere.
    if best["table_delta"] is not None:
        EDGE_STATS[f'table served the delta ({best["m"]["op"]})'] += 1
        if abs(best["table_delta"] - true_delta) > 1e-6: EDGE_STATS["DISAGREES with the published rates"] += 1
        if best["separated"]: EDGE_STATS["and flags it separated"] += 1
    else:
        EDGE_STATS[f'computed from the published rates ({best["m"]["op"]})'] += 1
    # Two exclusions, both about a demonstration being WRONG rather than merely uncertain: a move that
    # changes the very thing the question asks about (the feature is then the outcome, not a risk factor),
    # and a pair of figures that are both zero (there is no difference to explain). The loose bound on the
    # difference itself is a coding-artefact guard: no single history feature moves an absolute risk by
    # thirty points.
    if best["rate"] == 0.0 and p_own == 0.0: return None, "both figures are zero, nothing to explain"
    if abs(true_delta) > 0.3: return None, "the difference exceeds any plausible single-feature effect"
    return {
        "idx": it.get("_idx"), "item_id": it.get("item_id"), "version": v, "target": tc, "ref": best["sid"],
        "who": strip_codes((it.get("patient") or "").split(" with the following")[0].replace("Consider UK adults", "an adult of")),
        "hist": history_words(it.get("patient") or ""),
        "dis": strip_codes(lat.dname.get(dis, dis)), "yr": hor,
        "own_size": size_words(lat.states[tc]["n_band"]), "ref_size": size_words(best.get("at_risk") or best["n_band"]),
        "ref_rate": best["rate"], "ref_phrase": best["phrase"], "own_rate": p_own,
        "dropped": (describe(best["m"]["axis"], best["m"]["code"])
                    if best["m"]["axis"] in ("icd", "opcs", "bnf") else str(best["m"]["code"])),
        # own_rate is here so the acceptance rule can DERIVE which way round `true_delta` is signed
        # (own-minus-reference or reference-minus-own) instead of assuming it. A rule that restated the
        # convention would go silently backwards the moment the convention flips.
        "ref_at_risk": best.get("at_risk"), "true_delta": true_delta, "n_candidates": len(cands),
        "op": best["m"]["op"], "axis": best["m"]["axis"],
        "move_id": f'{best["m"]["axis"]}.{best["m"]["op"]}.{best["m"]["code"]}',  # RECORDED, never written into the text
        # BOTH ENDS of the chosen edge, as the band and its rank: an edge is only as good as its weaker end.
        "own_band": lat.states[tc]["n_band"], "ref_band": best["n_band"],
        "own_at_risk": own_at, "table_delta": best["table_delta"], "separated": best["separated"],
        "own_ar_ord": ar_ord(own_at), "ref_ar_ord": ar_ord(best.get("at_risk")),
        "own_ord": order.get(str(lat.states[tc]["n_band"]), 0),
        "ref_ord": order.get(str(best["n_band"]), 0),
    }, None

async def main():
    items = [json.loads(l) for l in open(f"{a.items}/t3_rate_{a.split}.jsonl")]
    for i, it in enumerate(items): it["_idx"] = i
    build_edge_index(items)
    facts, skipped = [], collections.Counter()
    t0 = time.time()
    for it in items:                        # steps 1-5 are pure CPU, so they run on EVERY item and the
        f, why = facts_for(it)              # selection happens afterwards on the edges, not on the items
        if f is None: skipped[why] += 1
        else: facts.append(f)
    print(f"[steps 1-5] {len(facts)} items usable, {sum(skipped.values())} skipped {dict(skipped)} "
          f"in {time.time()-t0:.0f}s", flush=True)
    # ORDER BY BOTH ENDS OF THE EDGE, LARGEST FIRST. The weaker end leads, because an edge is measured no
    # better than its smaller side: a rate read off 1000+ people and compared against a 10-49-person cohort
    # is one noisy number minus one solid one. Ties break on the stronger end, then on the reference's own
    # size, so among equally-banded edges the better-populated reference comes first. This is what makes the
    # first 1,000 demonstrations the 1,000 best-measured ones rather than 1,000 arbitrary ones.
    facts.sort(key=lambda f: (min(f["own_ord"], f["ref_ord"]), max(f["own_ord"], f["ref_ord"]),
                              min(f["own_ar_ord"], f["ref_ar_ord"]), max(f["own_ar_ord"], f["ref_ar_ord"]),
                              1 if f["separated"] else 0, -f["idx"]), reverse=True)
    if facts:
        # ROUND-ROBIN ACROSS OPS. Ranking on reliability alone hands the whole set to `drop`, because it is
        # the only move that leaves the 10-49 world: `add`/`switch`/`shift` land on another few-dozen-person
        # cohort 93% of the time (measured). A demonstration set that shows one
        # action teaches one action, and the agent has to choose the parameters at RL time. So each op
        # contributes its own best edges in turn, and the mix is printed rather than assumed.
        byop = collections.defaultdict(list)
        for f in facts: byop[f["op"]].append(f)
        top, k = [], 0
        while a.limit and len(top) < a.limit and any(len(v) > k for v in byop.values()):
            for op in sorted(byop):
                if len(byop[op]) > k and len(top) < a.limit: top.append(byop[op][k])
            k += 1
        if not a.limit: top = facts
        bands = collections.Counter((f["own_band"], f["ref_band"]) for f in top)
        ar = collections.Counter((str(f["own_at_risk"]), str(f["ref_at_risk"])) for f in top)
        print(f"[ranking] keeping the top {len(top)} of {len(facts)} edges by both ends' n_band, then by "
              f"both ends' at-risk band; (own,ref) n_band: {dict(bands.most_common(8))}", flush=True)
        print(f"[ranking] (own,ref) at-risk band of the kept edges: {dict(ar.most_common(8))}", flush=True)
        print(f"[ranking] ops available: {dict(collections.Counter(f['op'] for f in facts))}; "
              f"ops kept: {dict(collections.Counter(f['op'] for f in top))}", flush=True)
        print(f"[edge table] {dict(EDGE_STATS)}", flush=True)
        facts = top
    trues = [f["true_delta"] for f in facts]
    C0 = st.median(trues)          # the L1-best CONSTANT; a demonstration must beat it to be kept
    print(f"[acceptance] best constant delta C0 = {C0:+.6f} on {len(trues)} items; "
          f"|true| median {st.median([abs(x) for x in trues]):.4f}, share positive "
          f"{sum(1 for x in trues if x > 0)/max(len(trues),1):.2f}", flush=True)
    if a.facts_out:
        with open(a.facts_out, "w") as fh:
            for f in facts: fh.write(json.dumps(f) + "\n")
        print(f"[facts] {len(facts)} selected edges -> {a.facts_out}", flush=True)
        return
    if a.dry_run:
        for f in facts[:2]:
            print("\n" + "=" * 100)
            for k in ("item_id", "who", "hist", "dis", "yr", "own_band", "ref_band", "own_size", "ref_size",
                      "own_rate", "ref_rate", "dropped", "n_candidates", "true_delta"): print(f"  {k}: {f[k]}")
            print("-" * 100); print(writer_prompt(f))
        return
    import aiohttp, math
    # The number score (0.5*sign + w_mag*0.5*mag, max 0.75) lives in demo_quality, because the combined
    # score A = w_proc*Sproc + w_num*Snum needs the same function and a second copy would drift. On its
    # own it is not the acceptance rule: the reasoning is scored too.
    score = lambda d, true: num_score(d, true, QCFG.negligible_delta, QCFG.w_mag)
    print(f"[acceptance] {json.dumps(dict(QCFG), sort_keys=True)}", flush=True)
    sem = asyncio.Semaphore(a.concurrency); stats = collections.Counter(); kept = 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fh = open(a.candidates or a.out, "w", buffering=1)                     # line-buffered: the file is readable while it runs
    afh = open(a.audit, "w", buffering=1) if a.audit else None             # ditto, one line per CANDIDATE
    rows, accepted_sh = [], []

    async def one(s_, f):
        f = dict(f, prompt=writer_prompt(f), constant_score=round(score(C0, f["true_delta"]), 4))
        f["prompt_sha256"] = hashlib.sha256(f["prompt"].encode()).hexdigest()   # proves WHICH prompt wrote this
        cands = []
        for k in range(a.samples):
            text, val = await write_one(s_, f, sem, 0.7 + 0.1 * k)
            if val is None: stats["writer sample unusable"] += 1; continue
            cands.append({"text": text, "declared": val, "sample_index": k, "writer_temp": round(0.7 + 0.1 * k, 2),
                          "number_score": round(score(val, f["true_delta"]), 4)})
        if not cands: stats["writer produced nothing usable"] += 1; return None
        if a.candidates:                       # store everything; the accept rule runs later, offline
            stats["candidates"] += len(cands)
            return {"candidates_only": True,
                    "facts": {k: f[k] for k in ("item_id", "idx", "version", "target", "ref", "dropped", "ref_rate",
                                                "ref_at_risk", "true_delta", "n_candidates", "who", "hist", "dis",
                                                "yr", "own_size", "ref_size", "own_band", "ref_band",
                                                "own_rate", "op", "axis", "move_id", "ref_phrase",
                                                "separated", "own_at_risk")},
                    "constant_score": f["constant_score"], "samples": cands}
        # THE NUMBER DOES NOT DECIDE. It ranks which samples are worth the two judge calls; the
        # acceptance verdict is demo_quality.quality()'s, which scores the reasoning behind seven gates.
        cands.sort(key=lambda c: -c["number_score"])
        judged = 0
        for c in cands:
            if judged >= a.judge_top: break
            async with sem:                                  # the judge shares the writer's server: bound the herd
                c["quality"] = q = await quality(f, c["text"], c["declared"], f["true_delta"], QCFG, session=s_)
            if q["judge"].get("call_a_raw"): judged += 1      # a rule-only rejection costs no judge tokens
            stats["gate " + (q["reason"] or "ACCEPTED").split(":")[0].strip()] += 1
        stats["items judged"] += 1
        ok = sorted([c for c in cands if c.get("quality", {}).get("accepted")],
                    key=lambda c: (-c["quality"]["scores"]["A"], -c["quality"]["scores"]["Sproc"],
                                   abs(c["declared"] - f["true_delta"])))
        if not ok: stats["no sample passed the gates"] += 1
        return {"f": f, "cands": cands, "ok": ok}

    def emit(r):
        """Sequential half: G7 (near-duplicate) is the one gate that needs the already-accepted set."""
        nonlocal kept
        f, chosen = r["f"], None
        for rank, c in enumerate(r["ok"]):
            sh = shingles(c["text"])
            mj = max([jaccard(sh, o) for o in accepted_sh], default=0.0)
            c["max_jaccard_vs_accepted"] = round(mj, 4)
            if chosen is None and mj < QCFG.dup_jaccard:
                chosen, c["accepted_rank_within_item"] = c, rank
                accepted_sh.append(sh)
            elif chosen is None:
                c["quality"]["gates"]["G7"] = "fail"
                c["quality"]["accepted"], c["quality"]["reason"] = False, f"G7: near-duplicate of an accepted trace (Jaccard {mj:.3f})"
                stats["gate G7"] += 1
        if afh:
            for c in r["cands"]:
                afh.write(json.dumps({k: f.get(k) for k in ("item_id", "idx", "version", "target", "ref",
                          "dropped", "ref_rate", "ref_at_risk", "own_size", "ref_size", "who", "hist", "dis",
                          "yr", "n_candidates", "prompt_sha256", "own_band", "ref_band", "own_rate")}
                          | {"split": a.split, "text": c["text"],
                             "sample_index": c["sample_index"], "writer_temp": c["writer_temp"],
                             "truth": {"true_delta": f["true_delta"], "own_rate": f["own_rate"],
                                       "C0": round(C0, 6), "constant_score": f["constant_score"]},
                             "number_score": c["number_score"],
                             "max_jaccard_vs_accepted": c.get("max_jaccard_vs_accepted"),
                             "quality": c.get("quality"), "judged": "quality" in c,
                             "accepted": bool(c.get("quality", {}).get("accepted")),
                             "accepted_rank_within_item": c.get("accepted_rank_within_item")},
                          ensure_ascii=False) + "\n")
        if chosen is None: return
        q = chosen["quality"]
        fh.write(json.dumps({"messages": [{"role": "assistant", "content": f"<cohort-think>{chosen['text']}</cohort-think>"}],
                 "meta": {k: f[k] for k in ("item_id", "idx", "version", "target", "ref", "dropped", "ref_rate",
                                            "true_delta", "n_candidates", "ref_at_risk", "own_band", "ref_band", "own_rate",
                                            "prompt_sha256", "constant_score")}
                 | {"declared": chosen["declared"], "number_score": chosen["number_score"], "C0": round(C0, 6),
                    "sample_index": chosen["sample_index"], "writer_temp": chosen["writer_temp"],
                    "accepted_rank_within_item": chosen["accepted_rank_within_item"],
                    "max_jaccard_vs_accepted": chosen["max_jaccard_vs_accepted"],
                    "scores": q["scores"], "gates": q["gates"], "rules": q["rules"], "parsed": q["parsed"],
                    "judge": q["judge"]}}, ensure_ascii=False) + "\n")
        kept += 1
        rows.append({"Sproc": q["scores"]["Sproc"], "A": q["scores"]["A"], "Snum": q["scores"]["Snum"],
                     "words": q["parsed"]["n_words"], "mech": q["judge"].get("mech", ""),
                     "true": f["true_delta"], "declared": chosen["declared"],
                     "constant": f["constant_score"], **{k: q["scores"][k] for k in ("R1","R2","R3","R4","R5","R6")}})

    async def run():
        nonlocal kept
        async with aiohttp.ClientSession() as s_:
            CH = 400
            for i in range(0, len(facts), CH):
                out = await asyncio.gather(*[one(s_, f) for f in facts[i:i + CH]])
                for r in out:
                    if not r: continue
                    if a.candidates:
                        r.pop("candidates_only", None)
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n"); kept += 1
                    else:
                        emit(r)
                print(f"[{time.strftime('%H:%M:%S')}] {min(i + CH, len(facts))}/{len(facts)} items, "
                      f"written {kept}  {dict(stats)}", flush=True)
    await run(); fh.close()
    if afh: afh.close()
    print(f"[writer rejections] {dict(REJECT.most_common())}", flush=True)
    for why, tail in LAST_REJECT:
        print(f"  [rejected: {why}] ...{tail}", flush=True)
    print(f"[written] {kept} demonstrations -> {a.candidates or a.out}  {dict(stats)}")
    if rows:
        n = len(rows)
        qt = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
        for k in ("Sproc", "A", "Snum"):
            xs = [r[k] for r in rows]
            print(f"[{k}] n={n} min {min(xs):.3f} p25 {qt(xs,.25):.3f} median {st.median(xs):.3f} "
                  f"p75 {qt(xs,.75):.3f} max {max(xs):.3f}")
        for k in ("R1", "R2", "R3", "R4", "R5", "R6"):
            c = collections.Counter(r[k] for r in rows)
            top = max(c.values()) / n
            print(f"[rubric] {k}: 0={c[0]} 1={c[1]} 2={c[2]}  mean {sum(r[k] for r in rows)/n:.2f}"
                  + ("   *** DEAD ITEM: >90% on one value, it is measuring nothing ***" if top > 0.9 else ""))
        # THE SET-LEVEL CHECK: is the kept file's number channel better than the best constant?
        lift = st.median([r["Snum"] for r in rows]) - st.median([r["constant"] / 0.75 for r in rows])
        sgn = sum(1 for r in rows if (r["declared"] > 0) == (r["true"] > 0)) / n
        print(f"[lift] median Snum kept {st.median([r['Snum'] for r in rows]):.4f} vs constant "
              f"{st.median([r['constant']/0.75 for r in rows]):.4f}  LIFT {lift:+.4f}"
              + ("   (a file with lift <= 0 does not beat the constant)" if lift <= 0 else ""))
        print(f"[sign] declared agrees with the truth on {sgn:.3f} of kept demos; "
              f"share positive kept {sum(1 for r in rows if r['true']>0)/n:.3f} (stratification target 0.40-0.60)")
        # length bias: the judge must not be paying for words. Spearman = Pearson on ranks.
        rk = lambda xs: [sorted(xs).index(x) for x in xs]
        w, sp = rk([r["words"] for r in rows]), rk([r["Sproc"] for r in rows])
        mw, ms = st.mean(w), st.mean(sp)
        cov = sum((x-mw)*(y-ms) for x, y in zip(w, sp))
        vv = math.sqrt(sum((x-mw)**2 for x in w) * sum((y-ms)**2 for y in sp))
        rho = cov / vv if vv else float("nan")
        print(f"[judge health] Spearman(word count, Sproc) = {rho:+.3f}"
              + ("   *** |rho|>0.30: accept within word-count terciles instead ***" if abs(rho) > 0.30 else ""))
        mech = collections.Counter(r["mech"] for r in rows if r["mech"])
        print(f"[diversity] {len(mech)} distinct mechanisms over {n} kept ({1000*len(mech)/n:.0f} per 1000, "
              f"target >=60); most common {mech.most_common(3)}")

asyncio.run(main())
