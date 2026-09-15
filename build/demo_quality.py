# release-gate: allow-file -- ran against the private UKB extract; shipped for provenance, paths shown as they were
"""Demonstration acceptance: score a demonstration's REASONING, not only its number.

WHY THIS FILE EXISTS.  Accepting a demonstration on `0.5*sign + 0.25*mag` of the declared delta alone
never looks at whether the prose is clinical reasoning, so a vacuous paragraph with a lucky number is
kept and an excellent one with a slightly-off number is thrown away -- backwards for a cold start whose entire purpose is the process.  This module adds the process
channel: six 0/1/2 rubric items (R1..R6), seven hard gates (G1..G7), and one combined score
    A = w_proc * Sproc + w_num * Snum,   Sproc = sum(R)/12,  Snum = num_score(d,true)/0.75.

ONE SOURCE OF TRUTH.  The regexes and `num_score` live HERE and make_cohort_think_demos.py imports them; it
cannot be imported the other way round (it parses argv at import time), and a second copy of a
direction regex is a copy that drifts.

TWO DELIBERATE DEVIATIONS FROM THE SPEC, both visible in the record:
  * G3(b) is specified as "any direction word present anywhere contradicts sign_class(d)".  Taken
    literally that rejects "this is not lower, it is higher", i.e. correct prose that names the
    direction it rules out.  Implemented instead as a NET contradiction per region (final sentence
    first, body as fallback).  The literal verdict is computed anyway and recorded as
    `rules.g3_literal_would_reject`, so the strict variant can be adopted from data rather than taste.
  * R6's "contains >=1 lay clinical noun phrase" is not implemented -- there is no rule for it that is
    not a keyword list, and a keyword list is exactly the proxy R5's spec rejects.  R6 is therefore
    jargon + prompt-echo + truncation only.  # ponytail: add a judge question if R6 ever saturates.

The judge is the same MedGemma-27B that writes.  Self-preference is accepted (we rank one writer's own
samples, we do not compare models) and must be MEASURED, not assumed: see the anchor-set / ablation
harness in the caller's judge_health log.
"""
import argparse, asyncio, json, math, os, random, re, statistics as st

# ---------------------------------------------------------------- rules (imported by make_cohort_think_demos)
JARGON = re.compile(r"\[|\b(bnf|opcs|icd|icd-10|lattice|cohort|state[_ ]id|move[_ ]id|reference class|node|edge|"
                    r"n_band|size_after|cw_[a-z]+|tool|parameter|database|dataset|code)\b", re.I)
PER_K = re.compile(r"(?<![\w.])(\d+\.?\d*)\s*(?:in|per|out of)\s*(?:every\s*)?1[,.]?000\b", re.I)
SIGNED = re.compile(r"[+-]\s?(?:\d+\.?\d*|\.\d+)")
# THE FORMAT REWARD, and the only definition of it. The RL verifier and the demonstration generator must
# agree on what a well-formed <cohort-think> answer is, or the policy is rewarded for a shape the cold
# start never taught. Both import this.
TAG_OPEN, TAG_CLOSE = "<cohort-think>", "</cohort-think>"
_THINK = re.compile(re.escape(TAG_OPEN) + r"(.*?)" + re.escape(TAG_CLOSE), re.S)
_ADJ_TAIL = re.compile(r"Adjustment:\s*([+-]\d+\.\d{4})\.\s*$")

def cohort_think_format(text: str, allow_multiple: bool = False, block: str = "last") -> dict:
    """Does this answer have the shape the demonstrations have? Returns the parts, so a caller can both
    score the format and read the number without parsing twice.

    The five requirements, each the shape of one demonstration row:
      1. exactly one <cohort-think> ... </cohort-think> pair;
      2. the reasoning inside it is prose, not empty;
      3. it ends with `Adjustment: <signed number to four decimals>.`;
      4. that is the ONLY figure in the block -- the reasoning argues from medicine, not from numbers;
      5. no jargon, no bracketed code.
    """
    parts = _THINK.findall(text or "")
    out = {"ok": False, "n_blocks": len(parts), "declared": None, "checks": {}}
    c = out["checks"]
    c["exactly one block"] = len(parts) == 1 or allow_multiple
    if not parts or not c["exactly one block"]:
        return out
    body = (parts[0] if block == "first" else parts[-1]).strip()
    # WHICH block is read is a question about what is being trained, not a detail. Reading the LAST one
    # is right when the whole episode is credited; reading the FIRST is right when only the first turn is
    # -- the number under optimisation must be the number in the tokens that carry the advantage.
    m = _ADJ_TAIL.search(body)
    c["ends with a four-decimal signed Adjustment"] = bool(m)
    reasoning = body[: m.start()].strip() if m else body
    c["the reasoning is prose"] = len(reasoning.split()) >= 20
    c["no other figure in the reasoning"] = not re.search(r"\d", reasoning)
    c["no jargon or bracketed code"] = not bool(JARGON.search(body))
    out["declared"] = float(m.group(1)) if m else None
    out["ok"] = all(c.values())
    return out

def cohort_think_format_reward(text: str, allow_multiple: bool = False) -> float:
    """A graded format score in [0, 1]: the share of the five requirements met. Graded rather than
    pass/fail because a rollout that gets the tags right and the closing line wrong should be pulled
    towards the format, not lumped in with one that emitted no tags at all."""
    r = cohort_think_format(text, allow_multiple=allow_multiple)
    if not r["checks"]: return 0.0
    return sum(1.0 for v in r["checks"].values() if v) / 5.0

# HOW MUCH MUST BE SUBTRACTED -- the sentence the writer is asked for, parsed as the writer actually
# writes it. Requiring an explicit "+" threw away 10,013 of 12,000 samples, every one of which
# had stated the right quantity: "we need to subtract 0.0400", "Subtracting 0.0004 from the comparison
# group's figure", "we would need to subtract 32.6 per 1000". The verb carries the sign -- subtract means
# positive, add means negative -- and the per-1000 phrasing is the writer's own unit, not an error.
DECL = re.compile(r"\b(subtract(?:ing|ed)?|add(?:ing|ed)?)\b[^.]{0,90}?"
                  r"([+-]?\s?(?:\d+\.?\d*|\.\d+))"
                  r"(\s*(?:in|per|out of)\s*(?:every\s*)?1[,.]?000)?", re.I)

def parse_declared(text):
    """The number the demonstration commits to, in absolute rate units, or None.

    Tried in order: the closing `subtract/add` sentence (the one the prompt asks for), then any explicitly
    signed number. Returns the value to SUBTRACT from the reference group's figure, which is the
    demonstration's delta under the environment's own chain, commit = observe - delta."""
    ms = list(DECL.finditer(text or ""))
    if ms:
        m = ms[-1]
        try: v = float(m.group(2).replace(" ", ""))
        except ValueError: v = None
        if v is not None:
            if m.group(3): v /= 1000.0            # "32.6 per 1000"
            if m.group(1).lower().startswith("add"): v = -v
            return v
    nums = SIGNED.findall(text or "")
    if not nums: return None
    try: return float(nums[-1].replace(" ", ""))
    except ValueError: return None

UP = re.compile(r"\b(higher|increase[sd]?|raise[sd]?|elevat|greater|more likely)\b", re.I)
DOWN = re.compile(r"\b(lower|decrease[sd]?|reduce[sd]?|less likely|smaller|protective)\b", re.I)
# "it barely changes" is the third direction the writer is explicitly told it may choose; without it,
# gate G3(c) has no way to tell a correct negligible call from a silent one.
NEGL = re.compile(r"\b(barely|hardly|negligibl\w*|minimal\w*|essentially the same|much the same|"
                  r"little (?:to no |or no )?(?:change|difference)|no meaningful|broadly similar|"
                  r"close to (?:the|that|this)|unchanged|about the same|similar to)\b", re.I)
# borrowed verbatim from gen/demo_defect_census.py; they force R2 down without a judge call
BOTHWAYS = re.compile(r"\b(conversely|alternatively|on the other hand|or (?:conversely|instead)|"
                      r"whereas|although .{0,60}(?:could|might|may)|however,? (?:some|others|many|it)|"
                      r"paradoxically|either (?:raise|increase)|could also (?:lower|reduce|decrease|increase|raise))\b", re.I)
UNIVERSAL = re.compile(r"\b(chronic )?(inflammation|inflammatory (?:process|pathway|state)|immune (?:system|response|"
                       r"activation|modulation)|gut microbiome|systemic (?:effects?|illness|inflammation)|"
                       r"shared (?:risk factors|pathways))\b", re.I)
STOPW = {"the", "and", "for", "with", "that", "this", "was", "has", "are", "her", "his", "not", "any",
         "other", "which", "from", "have", "been", "were", "there", "their", "than", "over", "into",
         "recorded", "medication", "procedure", "condition", "diagnosis", "years", "year", "patient"}


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (t or "").strip()) if s.strip()]


def shingles(t, k=5):
    w = re.findall(r"[a-z0-9]+", (t or "").lower())
    return {tuple(w[i:i + k]) for i in range(max(len(w) - k + 1, 0))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def echo_frac(text, prompt, k=5):
    """Fraction of the trace's own word-5-grams that also occur in the prompt it was written from.
    Baseline on the 25-trace live-writer census: median 0.000, p90 0.018, max 0.042."""
    t, p = shingles(text, k), shingles(prompt, k)
    return (len(t & p) / len(t)) if t else 0.0


def sig_figs(s):
    """Significant figures of a decimal LITERAL as written ('0.6'->1, '13.1'->3, '1.0'->2)."""
    s = str(s).strip().lstrip("+-")
    if "." in s:
        i, fr = s.split(".", 1)
        d = (i.lstrip("0") + fr) if i.strip("0") else fr.lstrip("0")
        return max(len(d), 1)
    return max(len(s.lstrip("0")), 1)


def sign_class(d, s0):
    if abs(d) < s0: return "same"
    return "higher" if d > 0 else "lower"


def num_score(d, true, s0, w_mag):
    """The number channel, shared with make_cohort_think_demos.py: 0.5*sign + w_mag*0.5*mag, max attainable 0.75."""
    if abs(true) < s0:
        sign = 1.0 if abs(d) < s0 else 0.0
    else:
        sign = 1.0 if (d > 0) == (true > 0) and abs(d) >= s0 else 0.0
    mag = math.exp(-abs(d - true) / max(abs(true), s0))
    return 0.5 * sign + w_mag * 0.5 * mag


# ---------------------------------------------------------------- constants, every one a CLI argument
def add_args(ap):
    ap.add_argument("--w-proc", type=float, default=0.60)
    ap.add_argument("--w-num", type=float, default=0.40)
    ap.add_argument("--proc-floor", type=float, default=7.0 / 12.0)
    ap.add_argument("--negligible-delta", type=float, default=0.0026)
    ap.add_argument("--w-mag", type=float, default=0.5)
    ap.add_argument("--d-max", type=float, default=0.5)
    ap.add_argument("--max-words", type=int, default=300)
    ap.add_argument("--echo-warn", type=float, default=0.08)
    ap.add_argument("--echo-reject", type=float, default=0.15)
    ap.add_argument("--dup-jaccard", type=float, default=0.35)
    ap.add_argument("--judge-samples", type=int, default=3)
    ap.add_argument("--judge-temp", type=float, default=0.2)
    ap.add_argument("--judge-temp-suff", type=float, default=0.0)
    ap.add_argument("--judge-url", default=os.environ.get("CW_WRITER_URL", "http://hummer:8160/v1"))
    ap.add_argument("--judge-model", default=os.environ.get("CW_WRITER_MODEL", "medgemma27b-writer"))
    ap.add_argument("--judge-max-tokens", type=int, default=200)
    return ap


def cfg_from(ns):
    """The RESOLVED values, for the run directory's config.yaml."""
    keys = ("w_proc", "w_num", "proc_floor", "negligible_delta", "w_mag", "d_max", "max_words",
            "echo_warn", "echo_reject", "dup_jaccard", "judge_samples", "judge_temp",
            "judge_temp_suff", "judge_url", "judge_model", "judge_max_tokens")
    c = {k: getattr(ns, k) for k in keys}
    if c["w_proc"] < c["w_num"]:
        raise SystemExit(f"--w-proc ({c['w_proc']}) must be >= --w-num ({c['w_num']}): the whole point of "
                         f"this acceptance rule is that the process channel outweighs the number channel")
    return c


class _C(dict):
    __getattr__ = dict.__getitem__


def default_cfg(**over):
    ap = add_args(argparse.ArgumentParser())
    c = cfg_from(ap.parse_args([]))
    c.update(over)
    return _C(c)


# ---------------------------------------------------------------- the judge
JUDGE_SYSTEM = ("You are grading one paragraph of clinical reasoning against a fixed checklist. Answer each "
                "numbered question with yes or no on the evidence in the paragraph alone. Do not reward length, "
                "confidence, or fluent writing. A short paragraph that answers a question is a yes; a long one "
                "that does not is a no. Do not explain. Output one line of JSON and nothing else.")

QUESTIONS = [
    ("1a", "Does the reasoning mention at least two specific things from this patient's recorded history, age band, or sex?"),
    ("1b", "Does it use at least one of them as a stated reason why this patient's risk differs from the wider group's figure, as opposed to only describing the patient?"),
    ("1c", "Could this exact paragraph be pasted unchanged into a note about a different patient who also has {dropped}, without becoming wrong or irrelevant?"),
    ("2a", "Does the reasoning name a specific biological or clinical pathway by which {dropped} would change the chance of {dis}?"),
    ("2b", "Is that pathway specific to this condition, rather than a general appeal to inflammation, immune activation, shared risk factors, or overall poor health?"),
    ("2c", "Is the pathway as stated medically correct? Answer yes, no, or unsure."),
    ("2d", "Does the reasoning commit to one direction, rather than giving reasons pointing both up and down without choosing?"),
    ("3a", "The wider comparison group is people with the same history except that {dropped} is not required. Does the reasoning give a reason this group is an appropriate comparison for this patient that goes beyond it being larger or more reliable?"),
    ("3b", "Does the reasoning state at least one thing this wider group fails to capture about this patient, or one way the comparison could mislead?"),
    ("4a", "Before stating its conclusion, does the reasoning give a reason for the direction it chooses?"),
    ("4b", "Is that reason the pathway from question 2a, rather than a bare assertion that the feature is a known risk factor?"),
    ("5a", "Does the reasoning express uncertainty about how big the difference is, as opposed to only about its direction?"),
    ("5b", "Does it say why that size is uncertain, naming the small size of this patient's own group, how few or how varied the comparable groups are, the length of the time window, or how rare the outcome is?"),
]

SCORING_RULES = (
    'r1: 2 if 1a=yes and 1b=yes and 1c=no; else 1 if 1a=yes and 1c=no; else 0. '
    'r2: 0 if 2a=no or 2c=no or 2d=no; else 2 if 2b=yes; else 1 (2c=unsure counts as yes). '
    'r3: number of yes among 3a and 3b. '
    'r4: 0 if 4a=no; else 2 if 4a=yes and 4b=yes; else 1. '
    'r5: number of yes among 5a and 5b.'
)


def rubric_prompt(f, text, seed):
    qs = list(QUESTIONS)
    random.Random(seed).shuffle(qs)
    body = "\n".join(f"Q{k}. " + v.format(dropped=f["dropped"], dis=f["dis"]) for k, v in qs)
    return (f"Patient: {f['who']}. Recorded history: {f['hist']}.\n"
            f"Question asked: the chance of a first diagnosis of {f['dis']} within {f['yr']} year(s).\n"
            f"Wider comparison group: people with the same history except that {f['dropped']} is not "
            f"required; their figure is {f['ref_rate']*1000:.1f} in every 1000.\n"
            f"Reasoning to grade:\n<<<{text}>>>\n"
            f"{body}\n"
            f"Then set r1..r5 from the answers using these rules: {SCORING_RULES}\n"
            f'Also set "mech" to at most six lowercase words naming the single pathway the reasoning relies '
            f'on, or "none".\n'
            f'Output exactly one line of JSON: {{"r1":0,"r2":0,"r3":0,"r4":0,"r5":0,'
            f'"q":{{"1a":"yes","1b":"no","1c":"no","2a":"yes","2b":"yes","2c":"yes","2d":"yes",'
            f'"3a":"yes","3b":"no","4a":"yes","4b":"yes","5a":"yes","5b":"no"}},"mech":"..."}}')


def sufficiency_prompt(f, text):
    """Prose only: the final sentence is deleted and any OTHER rate the trace states is masked, so the
    judge must recover the direction from the argument and cannot read it off a number."""
    ss = sentences(text)
    body = " ".join(ss[:-1]) if len(ss) > 1 else ss[0] if ss else ""
    ref_s = f"{f['ref_rate']*1000:.1f}"
    body = PER_K.sub(lambda m: m.group(0) if m.group(1) == ref_s else "a figure", body)
    body = SIGNED.sub("a figure", body)
    return (f"A physician wrote the reasoning below about one patient. The reference figure for the wider "
            f"comparison group is {ref_s} in every 1000.\n<<<{body}>>>\n"
            f"Based only on this reasoning, is this patient's own chance higher than, lower than, or "
            f"essentially the same as that figure? Answer with exactly one word: higher, lower, or same.")


async def post_judge(session, cfg, messages, temp, n, max_tokens):
    import aiohttp
    body = {"model": cfg.judge_model, "temperature": temp, "n": n, "max_tokens": max_tokens,
            "messages": messages}
    async with session.post(f"{cfg.judge_url}/chat/completions", json=body,
                            timeout=aiohttp.ClientTimeout(total=300)) as r:
        j = await r.json()
    return [c["message"]["content"].strip() for c in j["choices"]]


def parse_rubric(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m: return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    try:
        rs = {k: int(d[k]) for k in ("r1", "r2", "r3", "r4", "r5")}
    except Exception:
        return None
    if any(v not in (0, 1, 2) for v in rs.values()): return None
    rs["q"] = d.get("q") if isinstance(d.get("q"), dict) else {}
    rs["mech"] = str(d.get("mech", ""))[:80].strip().lower()
    return rs


def parse_word(raw):
    m = re.search(r"\b(higher|lower|same)\b", raw.lower())
    return m.group(1) if m else None


async def default_judge(session, cfg, f, text, seed):
    """-> (rubric_raws, suff_raws). Two calls, n=judge_samples each, one retry on total parse failure."""
    msgs_a = [{"role": "system", "content": JUDGE_SYSTEM},
              {"role": "user", "content": rubric_prompt(f, text, seed)}]
    msgs_b = [{"role": "user", "content": sufficiency_prompt(f, text)}]
    a_raw, b_raw = await asyncio.gather(
        post_judge(session, cfg, msgs_a, cfg.judge_temp, cfg.judge_samples, cfg.judge_max_tokens),
        post_judge(session, cfg, msgs_b, cfg.judge_temp_suff, cfg.judge_samples, 8))
    retries = 0
    if not any(parse_rubric(x) for x in a_raw):
        retries = 1
        a_raw = await post_judge(session, cfg, msgs_a, cfg.judge_temp, cfg.judge_samples, cfg.judge_max_tokens)
    return a_raw, b_raw, retries


# ---------------------------------------------------------------- the one function
async def quality(f, text, declared, true_delta, cfg=None, session=None, judge=None):
    """Score one demonstration.  f = the facts dict from make_cohort_think_demos.facts_for (plus 'prompt' and,
    for gate G6, 'constant_score').  Returns a dict carrying the rubric sub-scores, the combined score,
    the hard-gate verdict and the reason for the first gate that failed.

    Pre-judge gates short-circuit: a trace that fails G1..G3 costs no judge tokens.
    """
    cfg = cfg or default_cfg()
    s0, out = cfg.negligible_delta, {}
    words = re.findall(r"[A-Za-z0-9']+", text or "")
    ss = sentences(text)
    last = ss[-1] if ss else ""
    body = " ".join(ss[:-1]) if len(ss) > 1 else ""
    per_all = PER_K.findall(text or "")
    per_last = PER_K.findall(last)
    sgn_all = SIGNED.findall(text or "")
    sgn_last = SIGNED.findall(last)
    stated = per_last[-1] if per_last else (per_all[-1] if per_all else None)
    own = float(stated) / 1000.0 if stated is not None else None
    form = "per_1000" if per_last else ("signed" if sgn_last else None)
    # WHICH WAY IS `declared` SIGNED?  true_delta may be own - reference or reference - own. Derive it
    # from the facts (both the item's own rate and the truth are there) rather than restating it: a
    # restated convention is a convention that goes stale.
    conv = 1
    if f.get("own_rate") is not None and true_delta:
        conv = 1 if ((f["own_rate"] - f["ref_rate"]) > 0) == (true_delta > 0) else -1
    patient_minus_ref = conv * declared
    sc = sign_class(patient_minus_ref, s0)
    ups_l, dn_l, ups_b, dn_b = len(UP.findall(last)), len(DOWN.findall(last)), len(UP.findall(body)), len(DOWN.findall(body))
    ech = echo_frac(text, f.get("prompt", ""))
    ref_s = f"{f['ref_rate']*1000:.1f}"
    sig_st, sig_ref = (sig_figs(stated) if stated else 0), sig_figs(ref_s)
    out["parsed"] = {"per_1000_value": float(stated) if stated else None, "own_rate": own, "d": declared,
                     "final_number_form": form, "signed_in_final": sgn_last[-1] if sgn_last else None,
                     "sign_class": sc, "patient_minus_ref": patient_minus_ref,
                     "delta_sign_convention": "own_minus_ref" if conv == 1 else "ref_minus_own",
                     "convention_derived": f.get("own_rate") is not None and bool(true_delta),
                     "final_sentence": last, "n_words": len(words), "n_sentences": len(ss),
                     "sig_figs_stated": sig_st, "sig_figs_ref": sig_ref, "per_k_all": per_all}
    out["rules"] = {"jargon_hit": bool(JARGON.search(text or "")), "echo_frac": round(ech, 4),
                    "truncated": not (text or "").rstrip().endswith((".", "?", "!")),
                    "dirword_last": {"up": ups_l, "down": dn_l, "negl": len(NEGL.findall(last))},
                    "dirword_body": {"up": ups_b, "down": dn_b, "negl": len(NEGL.findall(body))},
                    "bothways_hit": bool(BOTHWAYS.search(text or "")), "universal_hit": bool(UNIVERSAL.search(text or ""))}
    gates = {f"G{i}": "pass" for i in range(1, 8)}
    out["gates"], out["scores"], out["judge"] = gates, {}, {}

    def fail(g, why):
        gates[g] = "fail"
        out["accepted"], out["reason"] = False, f"{g}: {why}"
        out["scores"].setdefault("Snum", round(num_score(declared, true_delta, s0, cfg.w_mag) / 0.75, 4))
        return out

    # ---- G1 form / G2 a usable number
    if out["rules"]["jargon_hit"]: return fail("G1", "jargon or a bracketed code")
    if not (per_all or sgn_all): return fail("G2", "states no number anywhere: nothing to accept")
    if form is None: return fail("G1", "no rate per 1000 and no signed number in the final sentence")
    if own is not None and not (0.0 <= own <= 1.0): return fail("G1", f"stated rate out of range ({own})")
    if abs(declared) > cfg.d_max: return fail("G1", f"|d|={abs(declared):.3f} > d_max {cfg.d_max}")
    if out["rules"]["truncated"]: return fail("G1", "trace does not end in terminal punctuation (truncated)")
    if len(words) > cfg.max_words: return fail("G1", f"{len(words)} words > max_words {cfg.max_words}")

    # ---- G3 direction vs number.  Net contradiction per region; the literal 'any word anywhere'
    # verdict is recorded alongside so the strict variant can be chosen from data.
    lit = (sc == "higher" and (dn_l + dn_b) > 0) or (sc == "lower" and (ups_l + ups_b) > 0) or \
          (sc == "same" and (ups_l + ups_b + dn_l + dn_b) > 0 and not (out["rules"]["dirword_last"]["negl"] + out["rules"]["dirword_body"]["negl"]))
    out["rules"]["g3_literal_would_reject"] = bool(lit)
    net = (ups_l - dn_l) if (ups_l or dn_l) else (ups_b - dn_b)
    if not (ups_l or dn_l or ups_b or dn_b) and not (out["rules"]["dirword_last"]["negl"] + out["rules"]["dirword_body"]["negl"]):
        return fail("G3", "no direction word anywhere: the prose does not say which way it went")
    if sc == "higher" and net < 0: return fail("G3", "prose argues down, the number is up")
    if sc == "lower" and net > 0: return fail("G3", "prose argues up, the number is down")
    if sc == "same" and net != 0 and not (out["rules"]["dirword_last"]["negl"] + out["rules"]["dirword_body"]["negl"]):
        return fail("G3", "prose states a direction, the number is negligible")

    # ---- R6 is a pure rule (no judge call)
    if ech >= cfg.echo_reject: r6 = 0
    elif ech >= cfg.echo_warn: r6 = 1
    else: r6 = 2

    # ---- the judge
    seed = f.get("demo_id") or f"{f.get('item_id')}|{f.get('idx')}|{len(text)}"
    if judge is None:
        if session is None: raise ValueError("quality() needs an aiohttp session or an injected judge")
        a_raw, b_raw, retries = await default_judge(session, cfg, f, text, seed)
    else:
        a_raw, b_raw, retries = await judge(f, text, seed)
    parsed = [p for p in (parse_rubric(x) for x in a_raw) if p]
    words_b = [w for w in (parse_word(x) for x in b_raw) if w]
    out["judge"] = {"call_a_raw": a_raw, "call_b_raw": b_raw, "parse_retries": retries,
                    "n_parsed": len(parsed), "judge_model": cfg.judge_model, "judge_temp": cfg.judge_temp,
                    "judge_n": cfg.judge_samples, "question_permutation_seed": str(seed),
                    "q": [p["q"] for p in parsed],
                    "mech": st.mode([p["mech"] for p in parsed]) if parsed else "",
                    "suff_words": words_b}
    if not parsed: return fail("G5", "judge_parse_fail")
    med = lambda k: int(round(st.median([p[k] for p in parsed])))
    R = {"R1": med("r1"), "R2": med("r2"), "R3": med("r3"), "R4": med("r4"), "R5": med("r5"), "R6": r6}

    # R1 cheap pre-screen: no patient fact appears at all -> R1 is 0 whatever the judge said
    facts_w = {w for w in re.findall(r"[a-z]+", f"{f.get('hist','')} {f.get('who','')}".lower())
               if len(w) > 3 and w not in STOPW}
    text_w = set(re.findall(r"[a-z]+", (text or "").lower()))
    out["rules"]["patient_fact_hits"] = len(facts_w & text_w)
    if not (facts_w & text_w): R["R1"] = 0
    # R5 significant-figure rule: a rate stated more precisely than the reference is false precision
    out["rules"]["sigfig_rule_applies"] = form == "per_1000"
    out["rules"]["sigfig_violation"] = bool(form == "per_1000" and stated and sig_st > sig_ref)
    if out["rules"]["sigfig_violation"]: R["R5"] = 0
    # R2 rule floors, no judge call needed
    if out["rules"]["bothways_hit"]: R["R2"] = min(R["R2"], 1)
    if out["rules"]["universal_hit"]: R["R2"] = min(R["R2"], 1)

    # ---- G4 sufficiency
    maj = None
    if words_b:
        cnt = {w: words_b.count(w) for w in set(words_b)}
        top = max(cnt.values())
        winners = [w for w, c in cnt.items() if c == top]
        maj = winners[0] if len(winners) == 1 else None
    out["judge"]["suff_majority"] = maj
    if maj is None: R["R4"] = 0
    elif maj != sc: R["R4"] = 0

    Sproc = sum(R.values()) / 12.0
    raw = num_score(declared, true_delta, s0, cfg.w_mag)
    Snum = raw / 0.75
    A = cfg.w_proc * Sproc + cfg.w_num * Snum
    out["scores"] = dict(R, Sproc=round(Sproc, 4), score_raw=round(raw, 4), Snum=round(Snum, 4), A=round(A, 4))

    if maj is None: return fail("G4", "the judge could not agree what direction the prose implies")
    if maj != sc: return fail("G4", f"prose implies '{maj}', the number says '{sc}'")
    # ---- G5 reasoning floor
    for k in ("R2", "R3", "R4"):
        if R[k] == 0: return fail("G5", f"{k}=0 (mechanism, comparator and direction ARE the task)")
    if Sproc < cfg.proc_floor: return fail("G5", f"Sproc {Sproc:.3f} < floor {cfg.proc_floor:.3f}")
    # ---- G6 number floor
    cs = f.get("constant_score")
    if cs is not None and raw <= cs: return fail("G6", f"number score {raw:.4f} <= constant {cs:.4f}")
    out["accepted"], out["reason"] = True, None       # G7 (dedup) is set by the caller, which alone knows the accepted set
    return out


# ---------------------------------------------------------------- self-check
CLEAN = ("Long-term topical steroid use thins the dermis and blunts local host defence, so a break in this "
         "woman's skin is likelier to be colonised, and that is the pathway from her prescription to a first "
         "recorded skin infection over three years. The wider group shares her age band, her sex and her "
         "diabetes, so the one difference left is that exposure, which is why it is the right comparison "
         "rather than merely the larger one. It does not capture how long she has been on treatment, so the "
         "comparison could understate her. How much higher is uncertain because her own group holds only a "
         "few dozen people and the outcome is rare. I would put her at about 18 in every 1000.")
FIX = {"who": "an adult of 60 to 64, female", "hist": "type 2 diabetes; medication for skin conditions",
       "dis": "a first skin infection", "yr": 3, "ref_rate": 0.0131, "dropped": "medication for skin conditions",
       "prompt": "Patient: an adult of 60 to 64, female. Recorded history: type 2 diabetes.",
       "item_id": "fixture", "idx": 0, "constant_score": 0.10}
GOOD_JUDGE = {"r1": 2, "r2": 2, "r3": 2, "r4": 2, "r5": 2,
              "q": {k: "yes" for k, _ in QUESTIONS}, "mech": "skin barrier thinning"}


def _stub(rub=None, word="higher", n=3):
    raw = json.dumps(rub or GOOD_JUDGE)
    async def j(f, text, seed): return [raw] * n, [word] * n, 0
    return j


def _selfcheck():
    """Every rule must be shown to FIRE and to STAY SILENT.  The echo and duplication thresholds fire on
    nothing in the measured data (max echo 0.042, max pairwise Jaccard 0.0697), which is exactly why a check
    that cannot be made to fire is indistinguishable from one that is always satisfied."""
    cfg = default_cfg()
    run = lambda t, d, true, ff=None, j=None: asyncio.run(
        quality({**FIX, **(ff or {})}, t, d, true, cfg, judge=j or _stub()))

    # -- pure rules, standing alone
    assert sig_figs("0.6") == 1 and sig_figs("13.1") == 3 and sig_figs("1.0") == 2 and sig_figs("18") == 2
    assert sign_class(0.0, 0.0026) == "same" and sign_class(0.01, 0.0026) == "higher" and sign_class(-0.01, 0.0026) == "lower"
    assert abs(num_score(0.05, 0.05, 0.0026, 0.5) - 0.75) < 1e-9        # perfect number = the channel max
    assert echo_frac("a b c d e f", "zz") == 0.0
    assert echo_frac("a b c d e f", "a b c d e f") == 1.0               # the echo rule CAN reach 1.0
    assert jaccard(shingles(CLEAN), shingles(CLEAN)) == 1.0

    # -- the clean trace trips nothing and scores a known value
    r = run(CLEAN, 18 / 1000.0 - 0.0131, 0.006)
    assert r["accepted"] is True and r["reason"] is None, r["reason"]
    assert r["scores"]["R1"] == 2 and r["scores"]["R6"] == 2, r["scores"]
    assert r["rules"]["sigfig_violation"] is False and r["rules"]["truncated"] is False
    assert r["scores"]["Sproc"] == 1.0, r["scores"]
    # the known value: Sproc 1.0, d = +0.0049 vs true 0.006 -> sign 1, mag exp(-0.0011/0.006)
    exp_raw = 0.5 + 0.25 * math.exp(-abs(0.0049 - 0.006) / 0.006)
    assert abs(r["scores"]["score_raw"] - round(exp_raw, 4)) < 2e-4, (r["scores"], exp_raw)
    assert abs(r["scores"]["A"] - round(0.6 * 1.0 + 0.4 * (exp_raw / 0.75), 4)) < 3e-4, r["scores"]
    assert r["scores"]["A"] == 0.9777, r["scores"]["A"]                 # the frozen fixture value

    # -- G1 jargon fires
    assert run("The BNF chapter is unhelpful here, so about 18 in every 1000.", 0.005, 0.006)["reason"].startswith("G1: jargon")
    # -- G2 no rate at all
    assert run("Her risk is higher than the wider group's, I think.", 0.005, 0.006)["reason"].startswith("G2")
    # -- G1 truncation guard fires, and stays silent above
    assert "truncated" in run(CLEAN.rstrip(".") + " and therefore about 18 in every 1000 over", 0.005, 0.006)["reason"]
    # -- G1 rate must be in the FINAL sentence
    assert run("The wider group sits at about 13 in every 1000 and she is higher than that. Higher indeed.",
               0.005, 0.006)["reason"] == "G1: no rate per 1000 and no signed number in the final sentence"
    # ...and the SIGNED form of the same statement passes G1, because the writer prompt may ask for either
    sg = run(CLEAN.replace("about 18 in every 1000.", "a difference of +0.0049 from the wider group."),
             0.0049, 0.006)
    assert sg["accepted"] is True and sg["parsed"]["final_number_form"] == "signed", sg["reason"]
    assert sg["rules"]["sigfig_rule_applies"] is False
    # -- G1 |d| cap
    assert "d_max" in run(CLEAN.replace("18 in every 1000", "900 in every 1000"), 0.9, 0.006)["reason"]
    # -- G3 direction vs number: prose argues up, number is down
    assert run(CLEAN, -0.02, -0.03)["reason"].startswith("G3"), run(CLEAN, -0.02, -0.03)["reason"]
    # -- G3 direction-vs-zero: a direction stated while the number is negligible
    assert run(CLEAN, 0.0001, 0.0001)["reason"].startswith("G3")
    # -- G3 stays silent when the prose says it barely changes and the number agrees
    negl = ("Her steroid cream barely changes the local barrier at this dose, and the wider group already "
            "shares her age, sex and diabetes, so it is essentially the same picture. It does not capture "
            "her treatment duration. The size is uncertain because her own group is only a few dozen people. "
            "I would put her at about 13 in every 1000.")
    assert run(negl, 0.0001, 0.0001, j=_stub(word="same"))["accepted"] is True
    # -- echo rule fires above 0.15 and is silent at 0
    ech = run(CLEAN, 0.0049, 0.006, ff={"prompt": CLEAN})
    assert ech["rules"]["echo_frac"] == 1.0 and ech["scores"]["R6"] == 0, ech["rules"]
    # -- sig-fig rule fires: 18.37 in every 1000 against a reference written 13.1
    sf = run(CLEAN.replace("18 in every 1000", "18.37 in every 1000"), 0.0184 - 0.0131, 0.006)
    assert sf["rules"]["sigfig_violation"] is True and sf["scores"]["R5"] == 0, sf["rules"]
    # -- both-ways / universal-solvent mechanism floors R2 at 1 without a judge call
    bw = run(CLEAN.replace("and that is the pathway", "although it could also lower it, and that is the pathway"),
             0.0049, 0.006)
    assert bw["rules"]["bothways_hit"] is True and bw["scores"]["R2"] == 1, bw["scores"]
    un = run(CLEAN.replace("blunts local host defence", "drives chronic inflammation"), 0.0049, 0.006)
    assert un["rules"]["universal_hit"] is True and un["scores"]["R2"] == 1
    # -- the sign convention is DERIVED, not assumed: the same trace under the flipped convention
    flip = {"own_rate": 0.0131 - 0.0049, "ref_rate": 0.0131}      # own is LOWER, yet true_delta > 0
    fl = run(CLEAN, 0.0049, 0.0049, ff=flip, j=_stub(word="lower"))
    assert fl["parsed"]["delta_sign_convention"] == "ref_minus_own" and fl["parsed"]["sign_class"] == "lower", fl["parsed"]
    assert fl["reason"].startswith("G3"), fl["reason"]            # the prose argues up, the patient is lower
    nf = run(CLEAN, 0.0049, 0.006, ff={"own_rate": 0.0131 + 0.0049})
    assert nf["parsed"]["delta_sign_convention"] == "own_minus_ref" and nf["accepted"] is True, nf["reason"]
    # -- G4 sufficiency: the judge recovers the wrong direction
    assert run(CLEAN, 0.0049, 0.006, j=_stub(word="lower"))["reason"].startswith("G4")
    # -- G5: a zero on the mechanism is a rejection, not a low score
    z = run(CLEAN, 0.0049, 0.006, j=_stub(dict(GOOD_JUDGE, r2=0)))
    assert z["reason"].startswith("G5: R2=0"), z["reason"]
    # -- G5 floor: 6/12 is below 7/12
    lo = run(CLEAN, 0.0049, 0.006, j=_stub(dict(GOOD_JUDGE, r1=1, r2=1, r3=1, r4=1, r5=0)))
    assert lo["scores"]["Sproc"] == 0.5 and lo["reason"].startswith("G5: Sproc"), lo["reason"]
    # ...and 7/12 is exactly the floor, which passes: the boundary is inclusive
    at = run(CLEAN, 0.0049, 0.006, j=_stub(dict(GOOD_JUDGE, r1=1, r2=1, r3=1, r4=1, r5=1)))
    assert abs(at["scores"]["Sproc"] - 7 / 12) < 1e-3 and at["accepted"] is True, at["reason"]
    # -- G6 number floor: the constant beats it
    assert run(CLEAN, 0.0049, 0.006, ff={"constant_score": 0.99})["reason"].startswith("G6")
    # -- judge parse failure is recorded, not silently scored
    async def broken(f, t, s): return ["not json at all"] * 3, ["higher"] * 3, 1
    assert run(CLEAN, 0.0049, 0.006, j=broken)["reason"] == "G5: judge_parse_fail"
    # -- w_proc >= w_num is enforced at startup
    try:
        cfg_from(add_args(argparse.ArgumentParser()).parse_args(["--w-proc", "0.2", "--w-num", "0.8"]))
        raise AssertionError("the weight guard did not fire")
    except SystemExit:
        pass
    print("demo_quality selfcheck: all rules fire AND stay silent; fixture A = 0.9777")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="reasoning-scored acceptance for cohort-think demonstrations")
    add_args(ap)
    ap.add_argument("--selfcheck", action="store_true", help="run the fixture self-check and exit")
    ap.add_argument("--score-jsonl", default="", help="ABSOLUTE path to a jsonl of candidates to score live")
    ap.add_argument("--out", default="", help="ABSOLUTE path for the scored records")
    ns = ap.parse_args()
    for p in (ns.score_jsonl, ns.out):
        if p and not os.path.isabs(p):
            raise SystemExit(f"paths must be absolute, got {p!r}")
    if ns.selfcheck or not ns.score_jsonl:
        _selfcheck()
    else:
        cfg = _C(cfg_from(ns))
        async def _go():
            import aiohttp
            recs = [json.loads(l) for l in open(ns.score_jsonl)]
            async with aiohttp.ClientSession() as s:
                fh = open(ns.out or (ns.score_jsonl + ".scored"), "w", buffering=1)
                for r in recs:
                    q = await quality(r["facts"], r["text"], r["declared"], r["facts"]["true_delta"], cfg, session=s)
                    fh.write(json.dumps(r | {"quality": q}, ensure_ascii=False) + "\n")
                fh.close()
        asyncio.run(_go())
