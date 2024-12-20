import re

# HEADINGS

HEADING_KEYWORDS = [
    "предисловие",
    "содержание",
    "оглавление",
    "введение",
    "лист согласований",
]
CONTENTS_KEYWORDS = ["содержание", "оглавление", "лист согласований"]
HEADING_STOP_LIST = ["утверждаю"]

FOOTER_KEYWORDS = ["документ создан в электронной форме", "страница создана"]


# BULLETS

UNICODE_BULLETS = [
    "\u0095",
    "\u2022",
    "\u2023",
    "\u2043",
    "\u3164",
    "\u204C",
    "\u204D",
    "\u2219",
    "\u25CB",
    "\u25CF",
    "\u25D8",
    "\u25E6",
    "\u2619",
    "\u2765",
    "\u2767",
    "\u29BE",
    "\u29BF",
    "\u002D",
    "",
    r"\*",
    "\x95",
    "·",
]

BULLETS_PATTERN = "|".join(UNICODE_BULLETS)
UNICODE_BULLETS_RE = re.compile(f"(?:{BULLETS_PATTERN})(?!{BULLETS_PATTERN})")


def is_bulleted_text(text: str) -> bool:
    """Checks to see if the section of text is part of a bulleted list."""
    return UNICODE_BULLETS_RE.match(text.strip()) is not None
