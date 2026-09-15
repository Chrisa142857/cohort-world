#!/usr/bin/env python3
"""Environment calls per item, the "calls" column of Table 4: the number of cw_look, cw_move and
cw_observe results on the trace, before the commit, as mean and standard deviation over items.

  gen/count_calls.py --traces outputs/<run>/.../traces.jsonl
"""
import argparse, collections, json, statistics
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--traces", required=True)
a = ap.parse_args()
calls, by = [], collections.Counter()
for L in open(a.traces):
    for t in (json.loads(L).get("traces") or []):
        msgs = [x["message"] for x in (t.get("nodes") or []) if x.get("message")]
        if not msgs: continue
        c = collections.Counter(m.get("name") or "?" for m in msgs if m.get("role") == "tool")
        calls.append(sum(c.values())); by.update(c)
n = len(calls)
print(f"items {n}  calls per item {statistics.mean(calls):.1f} ± {statistics.pstdev(calls):.1f}  "
      f"(look {by['cw_look']/n:.2f}, move {by['cw_move']/n:.2f}, observe {by['cw_observe']/n:.2f})")
