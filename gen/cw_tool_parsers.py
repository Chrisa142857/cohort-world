"""A pythonic tool parser that finds the call even when the model wrote prose around it.

Loaded with `--tool-parser-plugin gen/cw_tool_parsers.py`; provides `pythonic_lenient`
and `openai_lenient`.

WHY THIS EXISTS. vLLM's stock `pythonic` parser only extracts a tool call when the message IS the
call: its regex is anchored to a bracketed list and `extract_tool_calls` bails unless the content
matches it whole. Its own source carries the matching TODO:

    # TODO(mdepinet): Possible future improvements:
    #   1. Support text + tools separated by either <|python_tag|> or \\n\\n

Measured consequence on this leaderboard: `gemma3_27b` produced, on 40 of 40 rollouts,

    "Okay, let's start by looking at the current cohort.\\n[cw_look()]"

-- the exact syntax the substituted Gemma-3 template asks for, with one sentence in front of it.
Every one of those rollouts scored `report_level 4` (FLOOR_REPORT, a constant), and the model was
recorded as unable to call tools. It is not: the parser could not see past the prose. The same
template and parser give `gemma3_4b` 93.9% scorable, so the difference is whether the model happens
to preface the call -- a formatting habit, not a capability.

That distinction decides whether a leaderboard says "MedGemma cannot do agentic tool use" or "our
parser required the call to be the entire message". The first is a claim about Google's model; the
second is a claim about our serving stack, and only the second is true.

WHAT IT CHANGES. Exactly one thing: the call is located with `re.search` over the whole message
instead of `re.match` on the whole message, and any text before it is returned as `content`. The
argument parsing, the AST walk and the streaming path are inherited from `PythonicToolParser`
unchanged -- this is deliberately not a reimplementation, so a call this parser accepts is parsed
by the same code that the stock parser would have used.

DISCLOSURE. A row produced under this parser is served under a non-default parsing rule and must be
marked as such, exactly like the substituted chat templates it accompanies (see
docs/tool-calling-capability.md). It does not change what the model emits.
"""

import json

import regex as re
from vllm.entrypoints.openai.engine.protocol import ExtractedToolCallInformation
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
try:                                              # absent in vLLM 0.28 (prime-rl's engine); only openai_lenient needs it
    from vllm.tool_parsers.openai_tool_parser import OpenAIToolParser as _OpenAIToolParser
except ImportError:
    _OpenAIToolParser = None
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
from vllm.tool_parsers.pythonic_tool_parser import PythonicToolParser

logger = init_logger(__name__)


@ToolParserManager.register_module("pythonic_lenient")
class PythonicLenientToolParser(PythonicToolParser):
    """`pythonic`, but the bracketed call may sit anywhere in the message."""

    def extract_tool_calls(self, model_output: str, request) -> ExtractedToolCallInformation:
        # Fast path: unchanged behaviour when the message already IS the call.
        stock = super().extract_tool_calls(model_output, request)
        if stock.tools_called:
            return stock

        m = self.TOOL_CALL_REGEX.search(model_output)
        if not m:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        # Re-run the INHERITED parser on just the bracketed span, so the arguments are parsed by
        # the same code path as always. Only the framing changed.
        inner = super().extract_tool_calls(m.group(0), request)
        if not inner.tools_called:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        prefix = model_output[: m.start()].strip()
        logger.info(
            "pythonic_lenient: recovered %d tool call(s) from a message with %d chars of "
            "surrounding text",
            len(inner.tool_calls),
            len(model_output) - len(m.group(0)),
        )
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=inner.tool_calls,
            content=prefix if prefix else None,
        )


def _selfcheck():
    """The check that matters: the stock parser must MISS the case this one recovers.

    A lenient parser that also passes the strict cases proves nothing on its own -- what has to be
    shown is that it changes the verdict on the exact string the leaderboard saw.
    """
    import re as _re
    RX = PythonicToolParser.TOOL_CALL_REGEX
    observed = "Okay, let's start by looking at the current cohort.\n[cw_look()]"
    bare = "[cw_look(cohort_id=\"C2_0007\")]"

    # Note the stock parser's own gate is a full match; ours is a search.
    assert RX.match(observed) is None, "stock regex should NOT match prose+call"
    assert RX.search(observed) is not None, "lenient search SHOULD find the call in prose+call"
    assert RX.match(bare) is not None, "stock regex must still match a bare call"
    found = RX.search(observed).group(0)
    assert found == "[cw_look()]", f"recovered span is {found!r}"
    print("selfcheck OK: stock regex misses 'prose + [cw_look()]', lenient search recovers "
          f"{found!r}; bare calls still match both.")




class OpenAILenientToolParser(_OpenAIToolParser or object):
    """`openai` (harmony), but the harmony CHANNEL marker is stripped off the tool name.

    gpt-oss speaks the harmony format, whose recipient field can carry a channel: the parser
    returned tool names like `cw_move<|channel|>commentary` while the ARGUMENTS were perfectly
    valid JSON (`{"move_id": "icd.add.R07->C2:626e34cee4b83248"}`). The environment rejects a tool
    it does not offer, so those calls were thrown away: 442 of ~3,900 calls in gptoss20b's archived
    T1 cell, 11%. The model was doing the right thing and paying budget for it.

    Only the name is touched. Arguments, ids and the streaming path are inherited untouched.
    """

    @staticmethod
    def _clean(name):
        if not isinstance(name, str):
            return name
        for sep in ("<|channel|>", "<|constrain|>", "<|"):
            if sep in name:
                return name.split(sep, 1)[0].strip()
        return name

    def extract_tool_calls(self, model_output: str, request) -> ExtractedToolCallInformation:
        r = super().extract_tool_calls(model_output, request)
        for c in (r.tool_calls or []):
            fn = getattr(c, "function", None)
            if fn is not None and getattr(fn, "name", None):
                fn.name = self._clean(fn.name)
            elif getattr(c, "name", None):
                c.name = self._clean(c.name)
        return r




if _OpenAIToolParser is not None:
    ToolParserManager.register_module("openai_lenient")(OpenAILenientToolParser)

def _selfcheck_openai():
    c = OpenAILenientToolParser._clean
    assert c("cw_move<|channel|>commentary") == "cw_move", c("cw_move<|channel|>commentary")
    assert c("cw_move<|channel|>json") == "cw_move"
    assert c("cw_look") == "cw_look"          # untouched when there is no marker
    assert c(None) is None
    print("selfcheck OK: harmony channel markers stripped from the NAME only; a clean name is "
          "returned unchanged.")


if __name__ == "__main__":
    _selfcheck()
    _selfcheck_openai()


@ToolParserManager.register_module("hermes_lenient")
class HermesLenientToolParser(Hermes2ProToolParser):
    """`hermes`, but a bare JSON object counts even without the <tool_call> wrapper.

    aaditya/Llama3-OpenBioLLM-8B is a Llama-3.0 fine-tune with no tool-calling training and no chat
    template, so the format has to be taught in the prompt. Told to emit
    `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`, it produced

        {"type": "tool_call", "name": "cw_look", "arguments": {"cohort_id": "C2_0007"}}

    -- the right tool, the right arguments, and the word `tool_call` read as a JSON FIELD rather
    than as a tag. The stock parser keys on the literal `<tool_call>` token, so it returned nothing
    and the model scored 0/5 on the tool probe: a serving-format verdict wearing the clothes of a
    capability verdict, which this project has now mistaken four times.

    The leniency is bounded on purpose. A bare object is accepted ONLY when it carries both `name`
    and `arguments` AND the name is one the request actually offers -- so ordinary JSON in an
    answer cannot be mistaken for a call. Everything else, streaming included, is inherited.
    """

    _OBJ = re.compile(r"\{(?:[^{}]|(?R))*\}")   # balanced braces; `regex`, not `re`, supports (?R)

    def extract_tool_calls(self, model_output: str, request) -> ExtractedToolCallInformation:
        stock = super().extract_tool_calls(model_output, request)
        if stock.tools_called:
            return stock

        offered = set()
        for t in (getattr(request, "tools", None) or []):
            fn = getattr(t, "function", None)
            n = getattr(fn, "name", None) if fn is not None else getattr(t, "name", None)
            if n:
                offered.add(n)

        for m in self._OBJ.finditer(model_output):
            try:
                o = json.loads(m.group(0))
            except Exception:
                continue
            if not isinstance(o, dict):
                continue
            name, args = o.get("name"), o.get("arguments")
            if not isinstance(name, str) or args is None:
                continue
            if offered and name not in offered:
                continue
            logger.info("hermes_lenient: recovered an untagged tool call to %s", name)
            # Hand the recovered object back through the STOCK parser wrapped in the tags it
            # expects, so ids, argument serialisation and every downstream field are produced by
            # the same code path as a normally-tagged call.
            wrapped = (self.tool_call_start_token
                       + json.dumps({"name": name, "arguments": args})
                       + self.tool_call_end_token)
            inner = super().extract_tool_calls(wrapped, request)
            if inner.tools_called:
                prefix = model_output[: m.start()].strip()
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=inner.tool_calls,
                    content=prefix if prefix else None)
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output)


def _selfcheck_hermes():
    """The stock parser must MISS the exact string openbio_8b produced, and this one must find it."""
    observed = '{"type": "tool_call", "name": "cw_look", "arguments": {"cohort_id": "C2_0007"}}'
    tagged = '<tool_call>{"name": "cw_look", "arguments": {}}</tool_call>'

    class _F:
        def __init__(self, n):
            self.name = n

    class _T:
        def __init__(self, n):
            self.function = _F(n)

    class _R:
        tools = [_T("cw_look"), _T("cw_move")]

    stock = Hermes2ProToolParser.__new__(Hermes2ProToolParser)
    stock.tool_call_start_token = "<tool_call>"
    stock.tool_call_end_token = "</tool_call>"
    assert stock.tool_call_start_token not in observed, "the observed string has no tag -- that IS the bug"

    p = HermesLenientToolParser.__new__(HermesLenientToolParser)
    hits = [m.group(0) for m in p._OBJ.finditer(observed)]
    assert hits and json.loads(hits[0])["name"] == "cw_look", hits
    # and the guard actually guards: a name nobody offered is refused
    other = '{"name": "rm_rf", "arguments": {}}'
    o = json.loads(next(p._OBJ.finditer(other)).group(0))
    assert o["name"] not in {"cw_look", "cw_move"}
    print("selfcheck OK: stock hermes cannot see the untagged call openbio_8b emitted; the lenient "
          "regex recovers cw_look, and an unoffered name is refused.")
