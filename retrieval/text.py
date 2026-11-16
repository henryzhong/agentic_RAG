"""Tokenisation and surface-form entity extraction (no LLM, fully reproducible)."""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOP = set("""a an the of in on at to for from by with and or as is are was were be been being
that this these those it its his her their our your my he she they we i you who whom whose which
what when where why how did does do done has have had not no nor but if then than so such also
into over under between during before after above below out up down off again further once here
there all any both each few more most other some only own same very can will just should now
""".split())

SMALL = {"of", "the", "and", "de", "van", "von", "del", "la", "le", "du", "di", "da", "bin", "al"}

# Relation / attribute words that appear in questions but are NOT entities.
# Linking them as seed entities is the classic cause of Graph-RAG drift.
RELATION_WORDS = {
    "child", "children", "son", "daughter", "spouse", "wife", "husband", "father", "mother",
    "parent", "sibling", "brother", "sister", "born", "birth", "death", "died", "city", "country",
    "capital", "name", "named", "nationality", "citizen", "director", "author", "writer",
    "founder", "member", "part", "located", "location", "release", "released", "published",
    "album", "song", "film", "movie", "book", "novel", "series", "season", "team", "club",
    "company", "school", "university", "year", "years", "date", "time", "number", "type",
    "left", "right", "hand", "hands", "batsman", "batsmen", "bowler", "fast", "first", "last",
    "following", "followed", "follow", "follows", "preceded", "next", "previous", "same",
    "different", "many", "person", "people", "become", "became", "becomes", "title", "titled",
    "known", "called", "much", "two", "three", "four", "five", "one", "who", "what", "when",
    "where", "which", "record", "label", "studio", "era", "period", "century", "companies",
    "make", "makes", "maker", "manufacturer", "product", "based", "work", "works", "place",
    "region", "area", "language", "genre", "award", "prize", "event", "war", "battle",
}

CAP_SPAN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'’\-\.]*)"
    r"(?:\s+(?:(?:of|the|and|de|van|von|del|la|le|du|di|da|bin|al)\s+)?[A-Z][A-Za-z0-9'’\-\.]*){0,5}")
QUOTED = re.compile(r"[\"“”']([A-Za-z][^\"“”']{2,60})[\"“”']")
YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
LEADING_WORD = re.compile(
    r"^(The|A|An|In|On|At|He|She|It|His|Her|Its|This|That|These|Those)\s+")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 1]


def ent_norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).strip(" .,;:'\"()")


def extract_entities(text: str, title: str = "") -> set[str]:
    """Surface entity candidates: the title, capitalised spans, quoted spans, years."""
    ents: set[str] = set()
    if title:
        ents.add(ent_norm(title))
    for m in CAP_SPAN.finditer(text):
        span = LEADING_WORD.sub("", m.group(0).strip())
        if len(span) < 3:
            continue
        words = span.split()
        if len(words) == 1 and len(span) < 4:
            continue
        if all(w.lower() in STOP for w in words):
            continue
        ents.add(ent_norm(span))
    for m in QUOTED.finditer(text):
        ents.add(ent_norm(m.group(1)))
    for m in YEAR.finditer(text):
        ents.add(m.group(1))
    return {e for e in ents if 2 <= len(e) <= 80}
