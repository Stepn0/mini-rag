"""Lowercase, Unicode-aware word tokenizer with stopwords and a Porter-lite stemmer.

The stemmer implements a small subset of Porter's algorithm: plural stripping
(-s, -es, -ies, -sses), the -ed / -ing rules with Porter's "stem must contain a
vowel" guard and consonant-undoubling / e-restoration fix-ups, a -ly rule, and
the final-e removal step. It is deliberately tiny; the goal is that common
inflections of the same word collapse to one token, not linguistic perfection.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)

STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have he her
    his how i if in into is it its just me more most my no not of on only or our over she
    should so some such than that the their them then there these they this those to too
    very was we were what when where which who why will with would you your
    """.split()
)

_VOWELS = "aeiou"


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(word: str) -> int:
    """Porter's m: the number of vowel-consonant sequences in the word."""
    m = 0
    i = 0
    n = len(word)
    while i < n and _is_consonant(word, i):
        i += 1
    while i < n:
        while i < n and not _is_consonant(word, i):
            i += 1
        if i >= n:
            break
        while i < n and _is_consonant(word, i):
            i += 1
        m += 1
    return m


def _has_vowel(word: str) -> bool:
    return any(not _is_consonant(word, i) for i in range(len(word)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    n = len(word)
    return (
        n >= 3
        and _is_consonant(word, n - 3)
        and not _is_consonant(word, n - 2)
        and _is_consonant(word, n - 1)
        and word[-1] not in "wxy"
    )


def stem(word: str) -> str:
    """Return a crude stem of ``word`` (expects lowercase)."""
    w = word
    if len(w) <= 2:
        return w

    # Step 1a: plurals.
    if w.endswith("ies"):
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith(("xes", "zes")):
        w = w[:-2]
    elif w.endswith("s"):
        w = w[:-1]

    # Step 1b: -ed / -ing, only if what remains still contains a vowel.
    for suffix in ("ing", "ed"):
        if w.endswith(suffix):
            base = w[: -len(suffix)]
            if len(base) >= 2 and _has_vowel(base):
                w = base
                if w.endswith(("at", "bl", "iz")):
                    w += "e"
                elif _ends_double_consonant(w) and w[-1] not in "lsz":
                    w = w[:-1]
                elif _measure(w) == 1 and _ends_cvc(w):
                    w += "e"
            break

    # Step 2 (lite): adverbs.
    if w.endswith("ly") and len(w) - 2 >= 4:
        w = w[:-2]

    # Step 5a: drop a final e when the stem is long enough to stand alone.
    if w.endswith("e") and len(w) >= 3:
        base = w[:-1]
        m = _measure(base)
        if m > 1 or (m == 1 and not _ends_cvc(base)):
            w = base

    return w


def tokenize(text: str, *, stopwords: bool = True, stemming: bool = True) -> list[str]:
    """Split ``text`` into lowercase word tokens, dropping stopwords and stemming."""
    tokens = _WORD_RE.findall(text.lower())
    if stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stemming:
        tokens = [stem(t) for t in tokens]
    return tokens
