#!/usr/bin/env python3
"""Reasoning-control `chat_template_kwargs`, dispatched by the checkpoint's DECLARED model_type.

TRAP 1 / PhenoCal BUG 29. PhenoCal's gate was a repo-NAME substring test:

    ctkw = {"enable_thinking": False} if "qwen3" in model_name.lower() else {}

`FreedomIntelligence/HuatuoGPT-3-8B` is model_type=qwen3 (Qwen3ForCausalLM, vocab 151936) and ships
the Qwen3 template, whose `enable_thinking is false` branch pre-fills "<think>\\n\\n</think>\\n\\n".
The name test missed it, thinking stayed ON, the model spent its whole token budget inside an
unterminated <think>, and parse rate was 0.000 with AUROC NaN. It looked like a MODEL failure, then
like a BUDGET failure worth ~27 GPU-hours. The defect was one missing substring.

So: read what the checkpoint DECLARES (config.json `model_type`), and keep the name test only as a
fallback for when the config cannot be read -- loudly, because a silent fallback restores the bug.

Measured on this zoo (templates on disk + a live probe on Qwen3.5-2B): the Qwen3.5
family does NOT ship one polarity. `Qwen3.5-27B/9B/4B` and `Qwen3-32B` gate on
`enable_thinking is false`, i.e. thinking is ON by default and our `false` turns it OFF; `Qwen3.5-2B`
gates on `enable_thinking is true`, i.e. thinking was already OFF and our `false` is a no-op. So on
the pre-fix path the Qwen3.5 SIZE TREND was confounded with reasoning mode -- 2B ran without
thinking while 4B/9B/27B ran with it. Sending the kwarg uniformly removes the confound; it is not
cosmetic.

CohortWorld has never sent these kwargs on the multi-turn eval path at all (they appear only in
gen/direct_ask.py), so every CohortWorld number to date was produced at whatever its template
happened to default to, and no run directory records which that was.

CLI:
    gen/reasoning_kwargs.py                # the dispatch table for the whole zoo
    gen/reasoning_kwargs.py --toml <name>  # the [sampling.extra_body...] block for an eval TOML
    gen/reasoning_kwargs.py --yaml <name>  # one line for a run's config.yaml
    gen/reasoning_kwargs.py --selfcheck
"""
import glob
import json
import os
import sys

GEN = os.path.dirname(os.path.abspath(__file__))
ZOO = os.path.join(GEN, "model_zoo.tsv")
HUB = os.environ.get("CW_HUB", os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")))


def zoo():
    """{name: {repo, parser, extra}} from the one shared map."""
    out = {}
    with open(ZOO) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip() or ln.startswith("#"):
                continue
            cols = (ln.split("\t") + ["-", "-", "-"])[:4]
            out[cols[0]] = {"repo": cols[1], "parser": cols[2],
                            "extra": "" if cols[3] == "-" else cols[3]}
    return out


def snapshot(name):
    """Local hub snapshot dir for a model name. A path is passed through. None if not found.

    TRAP 3. This resolved only through gen/model_zoo.tsv, a SECOND model map that
    predates the frozen zoo. Fourteen zoo models are absent from it -- including `huatuo3_8b`, the
    very model BUG 29 was about. An absent model got snapshot None -> model_type "" -> the repo-NAME
    fallback on a name with no "qwen3" in it -> {} with NO WARNING, because the warning only covered
    an unreadable config.json, not an unknown model. The bug the docstring above describes was
    therefore still firing at the moment that docstring was being read.

    So: resolve through the frozen zoo FIRST (gen/zoo.py -> clinical/model_zoo.py, the single source
    of truth), and fall back to the tsv only for names the zoo does not carry.
    """
    if os.path.isdir(name):
        return name
    try:
        sys.path.insert(0, GEN)
        from zoo import resolve as _zoo_resolve
        row = _zoo_resolve(name)
        if row and row[6]:
            return row[6]
    except Exception as e:
        print(f"[reasoning_kwargs] WARNING: frozen zoo unreadable ({type(e).__name__}: {e})",
              file=sys.stderr, flush=True)
    repo = zoo().get(name, {}).get("repo")
    if not repo:
        return None
    snaps = sorted(glob.glob(os.path.join(HUB, repo, "snapshots", "*")))
    return snaps[0] if snaps else None


def model_type(name):
    """Declared architecture family, lowercased. "" when the config cannot be read.

    config.json is read directly rather than through transformers.AutoConfig: the checkpoints are
    already on local disk, this must work with no network and no HF_TOKEN, and the only field
    wanted is one string.
    """
    snap = snapshot(name)
    if not snap:
        return ""
    try:
        with open(os.path.join(snap, "config.json")) as f:
            return str(json.load(f).get("model_type", "") or "").lower()
    except Exception as e:
        print(f"[reasoning_kwargs] WARNING: cannot read model_type for {name} "
              f"({type(e).__name__}: {e}); falling back to the repo-NAME test, which is exactly "
              f"the BUG 29 condition.", file=sys.stderr, flush=True)
        return ""


def _by_name(text):
    ml = text.lower()
    if "gpt-oss" in ml:
        return {"reasoning_effort": "low"}
    if "qwen3" in ml:
        return {"enable_thinking": False}
    return {}


def resolve(name):
    """(kwargs, model_type, source). source is 'config' or 'name-fallback'.

    Zoo convention, same as PhenoCal's: thinking OFF for every Qwen3-family model, so a branded
    Qwen3 finetune is scored on the same footing as the stock Qwen3 it was tuned from. Turning it on
    for some and off for others confounds the model with its reasoning budget.
    """
    mt = model_type(name)
    if not mt:
        # LOUD. A silent name-fallback is how BUG 29 survived a rewrite meant to kill it.
        print(f"[reasoning_kwargs] WARNING: no declared model_type for {name!r} -- falling back to "
              f"the repo-NAME test, which is exactly the BUG 29 condition. If this model is a "
              f"Qwen3-family finetune under another brand, thinking will be left ON and its parse "
              f"rate will be 0.", file=sys.stderr, flush=True)
        return _by_name(zoo().get(name, {}).get("repo", name)), "", "name-fallback"
    if mt == "gpt_oss":
        return {"reasoning_effort": "low"}, mt, "config"
    if mt.startswith("qwen3"):          # matches qwen3 and qwen3_5
        return {"enable_thinking": False}, mt, "config"
    return {}, mt, "config"


def reasoning_kwargs(name):
    return resolve(name)[0]


def _table():
    z = zoo()
    rows, flagged = [], []
    for name, e in z.items():
        kw, mt, src = resolve(name)
        guess = _by_name(e["repo"])
        mismatch = guess != kw
        rows.append((name, e["repo"].replace("models--", "").replace("--", "/"),
                     mt or "UNREADABLE", json.dumps(kw), src, "MISMATCH" if mismatch else ""))
        if mismatch:
            flagged.append((name, mt, json.dumps(guess), json.dumps(kw)))
    hdr = ("model", "repo", "declared model_type", "chat_template_kwargs", "source", "name-test")
    w = [max(len(hdr[i]), max(len(str(r[i])) for r in rows)) for i in range(6)]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    print("  ".join("-" * w[i] for i in range(6)))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    print()
    if flagged:
        print("BUG-29 PAYOFF -- declared type differs from what the name suggests:")
        for n, mt, g, k in flagged:
            print(f"  {n}: name test says {g}, model_type={mt} says {k}")
    else:
        print("BUG-29 PAYOFF: none -- every zoo model's declared model_type agrees with its name. "
              "The gate is still config-driven, because the zoo grows and the next branded "
              "finetune (HuatuoGPT-3, MedReason-8B, Llama3-Aloe-8B ... all in this hub) will not.")


def _selfcheck():
    assert resolve("qwen3.5-27b")[:2] == ({"enable_thinking": False}, "qwen3_5"), resolve("qwen3.5-27b")
    assert resolve("qwen32b")[:2] == ({"enable_thinking": False}, "qwen3"), resolve("qwen32b")
    assert resolve("qwen14b")[0] == {}, resolve("qwen14b")            # Qwen2.5: no thinking switch
    assert resolve("gemma3-27b")[0] == {}, resolve("gemma3-27b")
    # the BUG-29 case itself, by path: a branded Qwen3 whose NAME says nothing
    huatuo = sorted(glob.glob(os.path.join(HUB, "models--FreedomIntelligence--HuatuoGPT-3-8B",
                                           "snapshots", "*")))
    if huatuo:
        kw, mt, src = resolve(huatuo[0])
        assert (kw, src) == ({"enable_thinking": False}, "config"), (kw, mt, src)
        assert _by_name("FreedomIntelligence/HuatuoGPT-3-8B") == {}, "name test should MISS it"
    # unreadable config -> loud fallback to the name test, not a silent {}
    kw, mt, src = resolve("/nonexistent/qwen3-whatever")
    assert (kw, mt, src) == ({"enable_thinking": False}, "", "name-fallback"), (kw, mt, src)
    print("selfcheck OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        _table()
    elif a[0] == "--selfcheck":
        _selfcheck()
    elif a[0] == "--toml":
        kw, mt, src = resolve(a[1])
        if kw:
            print("[sampling.extra_body.chat_template_kwargs]")
            for k, v in kw.items():
                print(f"{k} = {json.dumps(v)}")
    elif a[0] == "--yaml":
        kw, mt, src = resolve(a[1])
        print(f"chat_template_kwargs: {json.dumps(kw, separators=(',', ':'))}")
        print(f"reasoning_gate_source: {src}{'' if src == 'config' else ' (model_type UNREADABLE)'}")
        print(f"model_type: {mt or 'unreadable'}")
    elif a[0] == "--field":                      # one shell-quotable value, for a driver
        print(json.dumps(resolve(a[1])[0], separators=(",", ":")))
    else:
        sys.exit(__doc__)
