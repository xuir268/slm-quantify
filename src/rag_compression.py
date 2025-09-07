from typing import List, Optional, Tuple
import re

# In production, replace this with an embedding-based index (e.g., SentenceTransformers + FAISS).
_DOCS: List[str] = [
    "CVE-2024-0001: Remote code execution via crafted JSON payload. Fixed in 3.2.5.",
    "CVE-2024-0002: SQL injection in search endpoint. Upgrade to 5.1.",
    "CVE-2024-0003: DoS via malformed image parsing. Patch available in 1.0.8."
]

def _tokens(s: str) -> List[str]:
    # alnum-ish tokens incl. cve-like ids and versions
    return re.findall(r"[a-z0-9][a-z0-9_.\-/:]*", s.lower())

def retrieve_topk(question: str, k: int = 3) -> List[str]:
    """
    Lightweight keyword-based retriever:
    - score docs by token overlap with the query
    - tie-break by shorter doc length
    - fallback: return first k if all scores are zero
    """
    if not _DOCS:
        return []
    qtok = set(_tokens(question))
    scored = []
    for d in _DOCS:
        dtok = set(_tokens(d))
        score = len(qtok & dtok)
        scored.append((score, len(d), d))
    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    # if everything scored 0, just return first k docs to keep behavior sane
    if scored and scored[0][0] == 0:
        return _DOCS[:k]
    return [d for _,__, d in scored[:k]]

def compress_to_bullets(ctx_docs: List[str], max_bullets: int = 6, query: Optional[str] = None) -> List[str]:
    """
    Create short, high-signal bullets from context docs.
    If a query is provided, rank sentences by keyword overlap with the query.
    Always returns at least one bullet if ctx_docs is non-empty.
    """
    def _sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[\.!?])\s+|\n+', text)
        return [re.sub(r'\s+', ' ', p).strip() for p in parts if p and p.strip()]

    def _score(sent: str, q: str) -> int:
        qs = set(re.findall(r'[a-z0-9\-_.\/]+', q.lower()))
        ss = set(re.findall(r'[a-z0-9\-_.\/]+', sent.lower()))
        if not qs or not ss:
            return 0
        return len(qs & ss)

    bullets: List[str] = []
    if query:
        cands: List[Tuple[int, str]] = []
        for doc in ctx_docs:
            for s in _sentences(doc):
                if len(s) < 6:
                    continue
                cands.append((_score(s, query), s))
        # Highest score first; if tie, prefer shorter sentences
        cands.sort(key=lambda t: (t[0], -len(t[1])), reverse=True)
        for sc, s in cands:
            if sc <= 0:
                continue
            bullets.append("• " + s[:220])
            if len(bullets) >= max_bullets:
                break

    # backfill if we still don't have enough (or none scored)
    if len(bullets) < max_bullets:
        for doc in ctx_docs:
            sents = _sentences(doc)
            for s in sents[:2]:  # diversify a bit
                b = "• " + s[:220]
                if s and b not in bullets:
                    bullets.append(b)
                    if len(bullets) >= max_bullets:
                        break
            if len(bullets) >= max_bullets:
                break

    # de-dup and enforce limit
    out: List[str] = []
    seen = set()
    for b in bullets:
        bb = b.strip(" •-—")
        if bb and bb not in seen:
            out.append("• " + bb)
            seen.add(bb)
        if len(out) >= max_bullets:
            break

    # If ctx_docs was empty, return empty list; else ensure at least one bullet
    if not out and ctx_docs:
        out = ["• " + (ctx_docs[0][:220].strip())]
    return out

def build_prompt(question: str, bullets: List[str]) -> str:
    """
    Build the final prompt. Handles empty bullets gracefully.
    """
    if bullets:
        ctx_block = "\n".join(bullets)
    else:
        ctx_block = "No external context available."
    return (
        "You are a concise cybersecurity analyst. "
        "Answer ONLY using the provided context.\n\n"
        f"Context:\n{ctx_block}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
