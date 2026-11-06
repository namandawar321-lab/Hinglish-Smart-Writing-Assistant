"""English-word protection and light Roman-Hinglish normalization.

The improved transliterator leaves English words in Latin script.
Short tokens that are usually Roman Hindi (to, do, so, me, is) are not
treated as English, so they still go to Devanagari.
"""

from __future__ import annotations

import re

# Whole-token expansions used only by the improved path.
# Values that should stay in Latin are also listed in ENGLISH.
TOKEN_MAP = {
    "nhi": "nahi",
    "nhin": "nahi",
    "nahin": "nahi",
    "kr": "kar",
    "krna": "karna",
    "krne": "karne",
    "krni": "karni",
    "kro": "karo",
    "rha": "raha",
    "rhe": "rahe",
    "rhi": "rahi",
    "kl": "kal",
    "tm": "tum",
    "kb": "kab",
    "yr": "yaar",
    "bht": "bahut",
    "bohot": "bahut",
    "acha": "accha",
    "achha": "accha",
    "thik": "theek",
    "pls": "please",
    "plz": "please",
    "plzz": "please",
    "msg": "message",
    "msgs": "messages",
    "bcoz": "because",
    "becoz": "because",
    "bcz": "because",
    "coz": "because",
    "thx": "thanks",
    "tysm": "thanks",
    "tmro": "tomorrow",
    "tmrw": "tomorrow",
    "dec": "December",
    "nov": "November",
    "jan": "January",
    "feb": "February",
    "apr": "April",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "sept": "September",
    "oct": "October",
}

# Explicit allow-list. Words that are English in spelling but usually Hindi
# in Hinglish are omitted on purpose: to (तो), do (दो), so (सो), me (में),
# is (इस), hi (ही), ho (हो), he (ही), in (इन).
ENGLISH = {
    "the", "and", "for", "with", "your", "you", "not", "will", "can", "have",
    "has", "was", "were", "are", "this", "that", "from", "just", "but",
    "please", "send", "thanks", "thank", "because", "tomorrow",
    "am", "pm",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "meeting", "report", "assignment", "movie", "free", "lunch", "late", "idea",
    "client", "update", "file", "email", "deadline", "project", "status",
    "manager", "feedback", "zoom", "call", "slides", "slide", "ready", "review",
    "laptop", "password", "server", "down", "bug", "fix", "link", "group",
    "share", "presentation", "office", "invoice", "payment", "pending",
    "screenshot", "error", "app", "crash", "login", "weekend", "plan",
    "message", "messages", "exam", "train", "reply", "urgent", "help", "flight",
    "bill", "class", "result", "room", "number", "cup", "rupees", "time",
    "start", "team", "schedule", "calendar", "document", "pdf", "download",
    "upload", "forward", "college", "professor", "semester", "marks", "website",
    "online", "offline", "party", "ticket", "hotel", "code", "test", "demo",
    "issue", "task", "note", "notes", "agenda", "invite", "join", "leave",
    "phone", "mobile", "whatsapp", "mail", "otp", "bank", "account", "card",
    "order", "delivery", "cab", "driver", "location", "map", "address",
    "school", "homework", "lecture", "lab", "home", "work", "today", "night",
    "morning", "good", "hello", "sorry", "bye", "ok", "okay", "yes", "asap",
    "eta", "eod", "wfh", "lol", "btw", "pin", "atm", "upi", "gps", "id", "hr",
    "cv", "ceo",
}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
TOKEN_RE = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)


def normalize_hinglish(text: str) -> str:
    """Expand a small set of SMS abbreviations before transliteration."""

    def repl(match: re.Match) -> str:
        word = match.group(0)
        mapped = TOKEN_MAP.get(word.lower())
        if mapped is None:
            return word
        if word.isupper() and len(word) > 1:
            return mapped.upper()
        if word[:1].isupper():
            return mapped[:1].upper() + mapped[1:]
        return mapped

    return re.sub(r"[A-Za-z]+", repl, text)


def is_protected(token: str) -> bool:
    """Return True if this token must not be sent to the transliterator."""
    if DEVANAGARI_RE.search(token):
        return True
    if any(ch.isdigit() for ch in token):
        return True
    lowered = token.lower()
    if "@" in token or lowered.startswith(("http://", "https://", "www.")):
        return True
    return lowered in ENGLISH


def _peel_edges(raw: str) -> tuple[str, str, str]:
    lead_ws = re.match(r"\s*", raw).group(0)
    rest = raw[len(lead_ws) :]
    trail_ws = re.search(r"\s*$", rest).group(0)
    core = rest[: len(rest) - len(trail_ws)] if trail_ws else rest
    match = re.match(r"^(\W*)(.*?)(\W*)$", core, flags=re.UNICODE)
    if not match:
        return lead_ws, core, trail_ws
    return lead_ws + match.group(1), match.group(2), match.group(3) + trail_ws


def apply_transliteration(text: str, translit_fn, preserve_english: bool = True) -> str:
    """Transliterate Hindi spans.

    preserve_english=False sends the whole sentence through translit_fn
    (the baseline). preserve_english=True normalizes a few SMS spellings
    and transliterates only the Hindi spans.
    """
    if not text or not text.strip():
        return ""
    if not preserve_english:
        return translit_fn(text).strip()

    source = normalize_hinglish(text)
    tokens = TOKEN_RE.findall(source)
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        raw = "".join(buf)
        buf.clear()
        lead, core, trail = _peel_edges(raw)
        if core and re.search(r"\w", core):
            out.append(lead + translit_fn(core).strip() + trail)
        else:
            out.append(raw)

    for tok in tokens:
        if re.fullmatch(r"\w+", tok) and is_protected(tok):
            flush()
            out.append(tok)
        elif re.fullmatch(r"\w+", tok):
            buf.append(tok)
        elif buf:
            buf.append(tok)
        else:
            out.append(tok)
    flush()
    return "".join(out).strip()
