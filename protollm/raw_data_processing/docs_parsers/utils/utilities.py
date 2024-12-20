from pathlib import Path
from typing import Union

import chardet
from ftfy import is_bad


def is_bad_encoding(lines: list[str]) -> bool:
    count_bad = sum([_is_bad(line) for line in lines], start=0)
    proportion_bad = count_bad / max(1, len(lines))
    return proportion_bad >= 0.5


def _is_bad(text: str) -> bool:
    if is_bad(text):
        return True
    # This verification works for russian docs !!!
    try:
        text.encode("sloppy-windows-1252")
    except UnicodeEncodeError:
        return False
    return True


def correct_path_encoding(path: Union[str, Path]) -> str:
    path = Path(path)
    path = Path(*[fix_zip_path(part) for part in path.parts])
    return str(path)


def fix_zip_path(path: str) -> str:
    try:
        string_bytes = path.encode("437")
        guessed_encoding = chardet.detect(string_bytes)["encoding"] or "cp1252"
        path = string_bytes.decode(guessed_encoding, "replace")
    except:
        pass
    return path
