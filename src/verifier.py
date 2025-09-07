# src/verifier.py
import re
from typing import Any, Iterable

_WORD = re.compile(r"[A-Za-z0-9_\-./]+")

def _flatten_to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple, set)):
        return " ".join(_flatten_to_str(e) for e in x if e is not None)
    return str(x)

def keywords(s: Any):
    s = _flatten_to_str(s).lower()
    return _WORD.findall(s)

def lexical_overlap(ans: Any, bullets: Any) -> float:
    a = set(keywords(ans))
    b = set(keywords(bullets))
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / max(len(a), len(b))

def verify(ans: Any, bullets: Any) -> float:
    return lexical_overlap(ans, bullets)