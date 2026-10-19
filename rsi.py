"""Juxy RSI - recursive self-improvement with no weight updates.

Four inference-time mechanisms:

  1. memory    every answer that scores well is persisted. A later prompt with
               high token overlap recalls it, so the model reuses what already
               worked instead of rediscovering it.
  2. verifier  cheap heuristic scorer - no second model, no extra tokens.
  3. refine    one self-critique pass, only when the draft scores badly.
  4. best-of-N sample N, keep the best-scoring (opt-in, costs N x tokens).

The model never learns. What improves is the store it draws on and the
selection pressure over its own samples. Honest framing: this is memory +
search, not gradient descent.
"""
import json
import os
import re
import sqlite3
import time

DB_PATH = os.environ.get("JUXY_RSI_DB", "/content/juxy_rsi.sqlite")
KEEP_SCORE = 0.62          # only persist answers we actually trust
RECALL_SCORE = 0.55        # minimum score for a recalled answer to be reused
RECALL_JACCARD = 0.45      # how similar a prompt must be to reuse

STOP = set("""a an the of to in for on and or is are was were be been being do
does did how what why when where which who with without from into about this
that these those it its as at by can could should would will i you he she they
we my your our their me him them please write make give show tell""".split())

REFUSAL = re.compile(
    r"\b(i can't|i cannot|i'm unable|i am unable|as an ai|i don't have access|"
    r"i do not have access|cannot assist|can't help with that)\b", re.I)
IDENTITY = re.compile(
    r"\b(i am claude|i'm claude|anthropic|i am (?:an? )?(?:openai|gpt|chatgpt|"
    r"deepseek|gemini)|developed by (?:openai|anthropic|google))\b", re.I)


def _tokens(text):
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOP and len(w) > 2}


def jaccard(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def score(prompt, answer):
    """0..1 heuristic quality gate. Deterministic, free, no extra tokens."""
    if not answer or not answer.strip():
        return 0.0
    a = answer.strip()
    s = 0.45
    fences = a.count("```")
    if fences >= 2:
        s += 0.20
        if fences % 2 == 0:
            s += 0.05            # properly closed block
    elif fences == 1:
        s -= 0.15                # code block never closed -> almost always truncation
    if len(a) < 1400:
        s += 0.08                # token efficiency
    if REFUSAL.search(a):
        s -= 0.45
    if IDENTITY.search(a):
        s -= 0.60                # identity hygiene is non-negotiable
    low = a.lower()
    if low.count("i'm sorry") or low.count("i apologize"):
        s -= 0.15
    # degenerate repetition: a long run of the same line. Skip code entirely -
    # braces and blank-ish lines repeat legitimately and were docking good answers.
    lines = [ln.strip() for ln in a.splitlines() if ln.strip()]
    if fences == 0 and len(lines) > 10:
        uniq = len(set(lines))
        if uniq / len(lines) < 0.4:
            s -= 0.30
    if re.search(r"(\b\w+\b)(?:\s+\1){4,}", a, re.I):
        s -= 0.25
    if a.endswith(("```", "}", ")", ".", "!", "?")):
        s += 0.03
    return max(0.0, min(1.0, s))


def _db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS memory(
        id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, answer TEXT,
        score REAL, uses INTEGER DEFAULT 0, ts REAL)""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON memory(ts)")
    return con


def recall(prompt, min_score=RECALL_SCORE, threshold=RECALL_JACCARD, limit=400):
    """Return the best previously-stored answer for a similar prompt, or None."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        con = _db()
        rows = con.execute(
            "SELECT id, prompt, answer FROM memory WHERE score >= ? "
            "ORDER BY score DESC, id DESC LIMIT ?", (min_score, limit)).fetchall()
        best, best_sim = None, threshold
        for rid, p, a in rows:
            sim = jaccard(prompt, p)
            if sim > best_sim:
                best, best_sim = (rid, a), sim
        if best:
            con.execute("UPDATE memory SET uses = uses + 1 WHERE id = ?", (best[0],))
            con.commit()
            con.close()
            return best[1]
        con.close()
    except Exception:
        pass
    return None


def remember(prompt, answer, sc=None):
    sc = score(prompt, answer) if sc is None else sc
    if sc < KEEP_SCORE or not answer.strip():
        return False
    try:
        con = _db()
        con.execute("INSERT INTO memory(prompt, answer, score, ts) VALUES (?,?,?,?)",
                    (prompt[:4000], answer[:8000], sc, time.time()))
        con.commit()
        con.close()
        return True
    except Exception:
        return False


CRITIQUE = ("Critique your previous answer. If it contains an error, a missing "
            "edge case, or unnecessary preamble, reply ONLY with the corrected "
            "answer. If it is already correct and complete, reply exactly: PASS")


def run(draft, regen, prompt, rsi_n=1, refine=True, prior=None):
    """Improve one answer.

    draft  - the first generation
    regen  - callable(extra_system: str) -> str, regenerates with a hint
    prompt - the user prompt (for scoring/memory)
    prior  - a recalled answer from memory, seeded as one more candidate
    """
    cands = [(score(prompt, draft), draft)]
    if prior:
        cands.append((score(prompt, prior), prior))

    for _ in range(max(0, rsi_n - 1)):
        try:
            alt = regen("Give a different, more complete answer than before.")
        except Exception:
            break
        cands.append((score(prompt, alt), alt))

    best_s, best = max(cands, key=lambda c: (c[0], -len(c[1])))

    if refine and best_s < 0.62:
        try:
            crit = regen(CRITIQUE)
            if crit and "PASS" not in crit.strip()[:12]:
                cs = score(prompt, crit)
                if cs > best_s:
                    best_s, best = cs, crit
        except Exception:
            pass

    remember(prompt, best, best_s)
    return best, {"rsi_score": round(best_s, 3), "rsi_candidates": len(cands),
                  "rsi_refined": best_s >= 0.62}


def stats():
    if not os.path.exists(DB_PATH):
        return {"memories": 0, "avg_score": 0, "uses": 0, "db": DB_PATH}
    try:
        con = _db()
        n, avg, uses = con.execute(
            "SELECT COUNT(*), COALESCE(AVG(score),0), COALESCE(SUM(uses),0) FROM memory").fetchone()
        con.close()
        return {"memories": n, "avg_score": round(avg, 3), "uses": uses, "db": DB_PATH}
    except Exception as e:
        return {"error": str(e)}
