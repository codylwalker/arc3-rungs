"""The patched system prompt carries the twin_tools section only when ARC3_TWIN_TOOLS is set.

Usage: python -I tests/prompt_check.py <path/to/patched/ARC3-Inference>
Imports the harness's tool_agent (needs its host dependencies, e.g. requests).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(sys.argv[1]))

from inference.agent import tool_agent  # noqa: E402
from inference.utils.twin_prompt import TWIN_TOOLS_PROMPT  # noqa: E402


def build():
    return tool_agent._build_system_prompt(tool_output_tokens=3072)


def main():
    os.environ.pop("ARC3_TWIN_TOOLS", None)
    off = build()
    os.environ["ARC3_TWIN_TOOLS"] = "1"
    on = build()
    ok = TWIN_TOOLS_PROMPT not in off and TWIN_TOOLS_PROMPT in on and len(on) - len(off) == len(TWIN_TOOLS_PROMPT)
    print("prompt off: %d chars, on: %d chars, section %d chars (~%d tokens)" % (
        len(off), len(on), len(TWIN_TOOLS_PROMPT), len(TWIN_TOOLS_PROMPT) // 4))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
