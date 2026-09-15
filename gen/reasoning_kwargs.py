#!/usr/bin/env python3
"""Serving facts and `chat_template_kwargs` for the models behind Tables 3 and 4, from gen/model_zoo.tsv.

gen/model_zoo.tsv is the one source of truth: per model, the Hugging Face repo and revision, the vLLM
tool-call parser, extra serve flags, and the chat_template_kwargs its rows were run with. A name that
is not in the table raises instead of falling back to a guess. The guess this replaces was a
repo-name substring test, and it is how a Qwen3-family checkpoint ends up answering with thinking left
on: its template thinks unless told not to, and under a short token cap the answer never arrives.

  gen/reasoning_kwargs.py                 print the table
  gen/reasoning_kwargs.py --toml NAME     the [sampling.extra_body.chat_template_kwargs] block for an eval TOML
  gen/reasoning_kwargs.py --serve NAME    a `vllm serve` command line for the model
  gen/reasoning_kwargs.py --selfcheck
"""
import json
import os
import sys
import tempfile

ZOO = os.environ.get("CW_MODEL_ZOO",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_zoo.tsv"))
COLS = ("name", "repo", "revision", "parser", "extra", "kwargs")


def zoo(path=ZOO):
    """{name: row} from the table; a malformed row raises with its line number."""
    out = {}
    with open(path) as f:
        for n, ln in enumerate(f, 1):
            ln = ln.rstrip("\n")
            if not ln.strip() or ln.startswith("#"):
                continue
            cols = ln.split("\t")
            if len(cols) != len(COLS):
                raise ValueError(f"{path}:{n}: expected {len(COLS)} tab-separated columns, got {len(cols)}")
            row = dict(zip(COLS, cols))
            if row["name"] in out:
                raise ValueError(f"{path}:{n}: duplicate name {row['name']!r}")
            row["extra"] = "" if row["extra"] == "-" else row["extra"]
            row["kwargs"] = json.loads(row["kwargs"])
            out[row["name"]] = row
    return out


def row(name, path=ZOO):
    z = zoo(path)
    if name not in z:
        raise KeyError(f"{name!r} is not in {path}; add its row (repo, revision, parser, flags, kwargs) "
                       f"rather than guess. Known: {sorted(z)}")
    return z[name]


def reasoning_kwargs(name, path=ZOO):
    return row(name, path)["kwargs"]


def qwen3_rows_with_thinking_on(z):
    """Names of Qwen3-family rows that do not switch thinking off; the shipped table must give []."""
    return [n for n, r in z.items()
            if r["repo"].startswith("Qwen/Qwen3") and r["kwargs"].get("enable_thinking") is not False]


def serve_command(name, path=ZOO):
    r = row(name, path)
    return (f"vllm serve {r['repo']} --revision {r['revision']} --served-model-name {name} "
            f"--enable-auto-tool-choice --tool-call-parser {r['parser']} {r['extra']}").rstrip()


def _selfcheck():
    z = zoo()
    assert len(z) == 10, sorted(z)
    assert qwen3_rows_with_thinking_on(z) == [], qwen3_rows_with_thinking_on(z)
    assert reasoning_kwargs("qwen3.5-9b") == {"enable_thinking": False}
    assert reasoning_kwargs("gemma3-27b") == {}
    try:
        reasoning_kwargs("qwen3-not-in-the-table")
        raise AssertionError("an unknown name did not raise")
    except KeyError:
        pass
    # both checks can fail: a Qwen3 row without the switch, and a row with a missing column
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as t:
        t.write("q\tQwen/Qwen3.5-9B\tabc\tqwen3_xml\t-\t{}\n")
    assert qwen3_rows_with_thinking_on(zoo(t.name)) == ["q"]
    with open(t.name, "w") as f:
        f.write("q\tQwen/Qwen3.5-9B\tabc\tqwen3_xml\t{}\n")
    try:
        zoo(t.name)
        raise AssertionError("a five-column row did not raise")
    except ValueError:
        pass
    finally:
        os.unlink(t.name)
    print("selfcheck OK")


def _table():
    for n, r in zoo().items():
        print(f"{n:14s} {r['repo']:44s} {r['parser']:18s} {json.dumps(r['kwargs'])}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        _table()
    elif a[0] == "--selfcheck":
        _selfcheck()
    elif a[0] == "--toml" and len(a) == 2:
        kw = reasoning_kwargs(a[1])
        if kw:
            print("[sampling.extra_body.chat_template_kwargs]")
            for k, v in kw.items():
                print(f"{k} = {json.dumps(v)}")
    elif a[0] == "--serve" and len(a) == 2:
        print(serve_command(a[1]))
    else:
        sys.exit(__doc__)
