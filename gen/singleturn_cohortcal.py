#!/usr/bin/env python3
"""The single-turn CohortCal leaderboard protocol against any OpenAI-compatible endpoint.

One system prompt (`prompts/sys_estimator.txt`), one user turn per item (the item's own `prompt`),
temperature 0, and a bare percentage parsed from the reply. Writes the record `gen/score_singleturn.py`
scores: `rows` of {idx, true_rate, disease, horizon, n, pred, pred_board_parser, text}.

  gen/singleturn_cohortcal.py --base-url http://host:port/v1 --model NAME --out FILE \
      [--items $CW_PHENOCAL/PhenoCal_C2_test.jsonl] [--conc 64] [--maxtok 48] [--no-thinking]
"""
import argparse, asyncio, hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parsers import parse_pct, parse_last_pct  # noqa: E402


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default=os.path.join(os.environ.get("CW_PHENOCAL", ""), "PhenoCal_C2_test.jsonl"),
                    help="CohortCal item file; default $CW_PHENOCAL/PhenoCal_C2_test.jsonl")
    ap.add_argument("--sys", default=os.path.join(os.path.dirname(HERE), "prompts", "sys_estimator.txt"))
    ap.add_argument("--conc", type=int, default=64); ap.add_argument("--maxtok", type=int, default=48)
    ap.add_argument("--n", type=int, default=0, help="first n items only (smoke)")
    ap.add_argument("--no-thinking", action="store_true", help="pass chat_template_kwargs.enable_thinking=false")
    ap.add_argument("--api-key-var", default="CW_VLLM_KEY")
    a = ap.parse_args()
    import httpx
    items = [json.loads(l) for l in open(a.items)]
    if a.n: items = items[:a.n]
    SYS = open(a.sys).read()
    url = a.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": "Bearer " + os.environ.get(a.api_key_var, "dummy")}
    sem = asyncio.Semaphore(a.conc); outs = [None] * len(items)

    async def one(i, it, client):
        async with sem:
            body = {"model": a.model, "temperature": 0.0, "max_tokens": a.maxtok,
                    "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": it["prompt"]}]}
            if a.no_thinking: body["chat_template_kwargs"] = {"enable_thinking": False}
            for _ in range(3):
                try:
                    r = await client.post(url, json=body, headers=headers, timeout=180)
                    outs[i] = r.json()["choices"][0]["message"]["content"]; return
                except Exception:
                    await asyncio.sleep(2)
            outs[i] = ""
    t0 = time.time()
    async with httpx.AsyncClient() as c:
        await asyncio.gather(*(one(i, it, c) for i, it in enumerate(items)))
    preds = [parse_last_pct(o) for o in outs]; board = [parse_pct(o) for o in outs]
    ok = sum(p is not None for p in preds)
    rec = {"model": a.model, "sys": a.sys, "sys_md5": hashlib.md5(SYS.encode()).hexdigest(),
           "maxtok": a.maxtok, "temperature": 0.0, "parser": "parse_last_pct; parse_pct as pred_board_parser",
           "n": len(items), "parsed": ok, "fmt": ok / max(len(items), 1), "wall_s": round(time.time() - t0, 1),
           "distinct_preds": len({p for p in preds if p is not None}),
           "rows": [{"idx": i, "true_rate": it["ground_truth"]["rate"],
                     "disease": (it.get("metadata") or {}).get("disease_icd10_3char"),
                     "horizon": (it.get("metadata") or {}).get("horizon_years"),
                     "n": (it.get("metadata") or {}).get("n"),
                     "pred": preds[i], "pred_board_parser": board[i], "text": outs[i]}
                    for i, it in enumerate(items)]}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"))
    print(f"wrote {a.out}: n {len(items)} parsed {ok} wall {rec['wall_s']}s")

asyncio.run(main())
