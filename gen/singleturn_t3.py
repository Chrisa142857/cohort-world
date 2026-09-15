#!/usr/bin/env python3
"""The no-environment comparator for the CohortThink cell: one turn, no tools.

The row this fills is the model answering the SAME question the agents answer, from the written
history alone. It has no episode, so it has no matching step and no contrast step to score, and that
absence is the point of the row rather than a gap in it.

Nothing about the question is restated here. The text comes from `internal.cohort_think_question`,
the one the agents receive, cut at the sentence asking for a comparison, since a model with no tools
cannot walk to one. The system prompt is the estimator prompt of the CohortCal board, `prompts/sys_estimator.txt`.

  singleturn_t3.py --host H --port P --model NAME --out FILE [--items PATH] [--conc N]
"""
import argparse, asyncio, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cohortworld_env"))
from cohortworld_env.internal import cohort_think_question
from cohortworld_env import reward as R

SYS = open(os.environ.get("CW_SYS_ESTIMATOR", os.path.join(os.path.dirname(HERE), "prompts", "sys_estimator.txt"))).read().strip()

_CUT = "Reason about which larger group"

def question(it):
    q = cohort_think_question(it["description"], it["disease_words"], it["horizon"], it["sex"])
    return q.split(_CUT)[0].strip()

def disease_words(it):
    """The disease as the agents see it named, read off the environment rather than restated."""
    from cohortworld_env.internal import _disease_name
    return _disease_name(it["version"], it["disease"])

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True); ap.add_argument("--port", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--items", default=os.path.join(os.environ.get("CW_ITEMS", ""), "t3_rate_unseen.jsonl"),
                    help="CohortThink item file; default $CW_ITEMS/t3_rate_unseen.jsonl")
    ap.add_argument("--conc", type=int, default=32); ap.add_argument("--maxtok", type=int, default=32)
    a = ap.parse_args()
    import httpx
    items = [json.loads(l) for l in open(a.items)]
    for it in items:
        it["disease_words"] = disease_words(it)
    url = f"http://{a.host}:{a.port}/v1/chat/completions"
    sem = asyncio.Semaphore(a.conc)
    out = [None] * len(items)

    async def one(i, it, client):
        async with sem:
            body = {"model": a.model, "temperature": 0.0, "max_tokens": a.maxtok,
                    "messages": [{"role": "system", "content": SYS},
                                 {"role": "user", "content": question(it)}],
                    "chat_template_kwargs": {"enable_thinking": False}}
            for _try in range(3):
                try:
                    r = await client.post(url, json=body, timeout=180)
                    txt = r.json()["choices"][0]["message"]["content"]
                    out[i] = {"item_id": it["item_id"], "disease": it["disease"],
                              "horizon": it["horizon"], "version": it["version"],
                              "true_rate": it["answer"]["rate"], "text": txt}
                    return
                except Exception as e:
                    err = str(e)[:120]
            out[i] = {"item_id": it["item_id"], "disease": it["disease"], "horizon": it["horizon"],
                      "version": it["version"], "true_rate": it["answer"]["rate"], "text": "", "error": err}

    async with httpx.AsyncClient() as c:
        await asyncio.gather(*(one(i, it, c) for i, it in enumerate(items)))
    with open(a.out, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} to {a.out}; empty replies {sum(1 for r in out if not r['text'])}")

asyncio.run(main())
