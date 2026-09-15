#!/usr/bin/env python3
# release-gate: allow-file -- ran against the private UKB extract; shipped for provenance, paths shown as they were
"""Legal candidate codes for the add / switch actions, per version.

Derived from the CODE SYSTEMS as they appear in the source extract, never from the released lattice.
Enumerating the codes that survive in released states would hand the agent the support set, which is
the thing G1 is supposed to test. A code that exists in the records but never reaches k>=10 in any
state stays in the vocabulary on purpose: selecting it is a legal move that returns a miss, and the
miss is informative and costs budget.

No organ-alignment table is needed -- the code system IS the class taxonomy. OPCS chapter letter and
BNF chapter/section are the levels, so `switch` (same-class substitute) is "another code sharing the
prefix one level up", computable from the code alone.
"""
import json, os, pickle, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from OK__build_cohortworld_v2 import VERSIONS, sysname
PANEL = os.environ.get("CW_PANEL", "/work/USERS/ziquanw/cohortworld/official")


def main():
    hx = pickle.load(open(os.environ.get("CW_HX", "/ram/USERS/ziquanw/digitaltwinreward/ukb_qa_brain/output_panel/cohortworld_hx_alldates.pkl"), "rb"))
    full = {"O": set(), "B": set()}
    for sysk, key in (("O", "opcs"), ("B", "bnf")):
        for codes in hx[key].values():
            full[sysk].update(codes)
    print(f"  源记录中的完整码: OPCS {len(full['O']):,} | BNF {len(full['B']):,}")

    for ver, g in VERSIONS.items():
        nO, nB = g[1], g[2]
        vocab = {}
        for sysk, n, ax in (("O", nO, "opcs"), ("B", nB, "bnf")):
            codes = sorted({c[:n] for c in full[sysk] if len(c) >= n})
            vocab[ax] = [{"code": c, "name": sysname(sysk, c),
                          "parent": c[:max(1, n - (2 if sysk == "B" else 1))] if n > 1 else None}
                         for c in codes]
        # released states use a strict SUBSET -- shown here so the "not from the lattice" claim is
        # checkable rather than asserted
        used = {"opcs": set(), "bnf": set()}
        for sp in ("train", "val", "test"):
            for l in open(f"{PANEL}/CohortWorld_{ver}_{sp}_states.jsonl"):
                d = json.loads(l)
                for ax in ("opcs", "bnf"):
                    used[ax].update(c for c, _ in d["axes"][ax] if c != "*")
        out = {"version": ver, "source": "code systems as instantiated in the source extract",
               "granularity": {"opcs_chars": nO, "bnf_chars": nB},
               "switch_class": {"opcs": "codes sharing the first %d chars" % max(1, nO - 1),
                                "bnf": "codes sharing the first %d chars" % max(2, nB - 2)},
               "candidates": vocab,
               "coverage_check": {ax: {"vocabulary": len(vocab[ax]),
                                       "appearing_in_released_states": len(used[ax]),
                                       "unreleased_but_legal": len(vocab[ax]) - len(used[ax])}
                                  for ax in ("opcs", "bnf")}}
        p = f"{PANEL}/CohortWorld_{ver}_vocab.json"
        json.dump(out, open(p, "w"), indent=1)
        cc = out["coverage_check"]
        print(f"  {ver:<8s} OPCS {cc['opcs']['vocabulary']:>5,} 候选 / 已发布中出现 "
              f"{cc['opcs']['appearing_in_released_states']:>5,} | "
              f"BNF {cc['bnf']['vocabulary']:>5,} / {cc['bnf']['appearing_in_released_states']:>5,}")


if __name__ == "__main__":
    main()
