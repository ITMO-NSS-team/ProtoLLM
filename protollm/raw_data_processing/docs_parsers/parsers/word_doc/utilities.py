import re

from protollm.raw_data_processing.docs_parsers.parsers.utilities import (
    HEADING_KEYWORDS,
    HEADING_STOP_LIST,
)


def _get_heading_hierarchy_level(text: str, metadata: dict) -> int:
    low_text = text.lower()
    if any([stop_word in low_text for stop_word in HEADING_STOP_LIST]):
        return -1

    low_text = "".join(re.findall(r"[^\W\d_]", low_text, re.UNICODE))
    if low_text in HEADING_KEYWORDS:
        return 0

    if metadata["bold"]:
        if metadata["list_level"] != -1:
            return metadata["list_level"]
        if metadata["is_bullet_list"] or text.isupper():
            return 0

    return -1


def add_headings_hierarchy(
    lines: list[str], metadata: list[dict]
) -> tuple[list[str], list[dict]]:
    new_lines, new_metadata = [], []
    hierarchy = [""]
    for line, line_meta in zip(lines, metadata):
        hierarchy_level = _get_heading_hierarchy_level(line, line_meta)
        if hierarchy_level == -1:
            new_meta = {**line_meta, "headings": list(hierarchy)}
            new_lines.append(line)
            new_metadata.append(new_meta)
        else:
            if hierarchy_level < len(hierarchy):
                hierarchy = hierarchy[:hierarchy_level]
            elif hierarchy_level > len(hierarchy):
                hierarchy.extend([""] * (hierarchy_level - len(hierarchy)))
            hierarchy.append(line)
    return new_lines, new_metadata


def get_chapters(
    lines: list[str], metadata: list[dict]
) -> tuple[list[str], list[dict]]:
    chapters, meta = [], []
    cur_chapter = ""
    for line, line_meta in zip(lines, metadata):
        if chapters and (cur_chapter == line_meta["headings"][0]):
            chapters[-1] = join_texts(chapters[-1], line, joiner="\n")
            update_metadata(meta[-1], line_meta)
            continue
        cur_chapter = line_meta["headings"][0]
        chapters.append(line)
        meta.append({"chapter": cur_chapter, "headings": [cur_chapter]})
        update_metadata(meta[-1], line_meta)
    return chapters, meta


def get_paragraphs(
    lines: list[str], metadata: list[dict]
) -> tuple[list[str], list[dict]]:
    paragraphs, meta = [], []
    cur_paragraph = ""
    for line, line_meta in zip(lines, metadata):
        if (
            paragraphs
            and line_meta["is_bullet_list"]
            and (cur_paragraph == line_meta["headings"][-1])
        ):
            paragraphs[-1] = join_texts(paragraphs[-1], line, joiner=" ")
            update_metadata(meta[-1], line_meta)
            continue
        cur_paragraph = line_meta["headings"][-1]
        paragraphs.append(line)
        meta.append({"headings": line_meta["headings"]})
        update_metadata(meta[-1], line_meta)
    return paragraphs, meta


def update_metadata(meta: dict, line_meta: dict):
    for key in ["images", "formulas", "urls", "tables"]:
        meta[key] = {**meta.get(key, {}), **line_meta.get(key, {})}


def join_texts(original: str, new: str, joiner: str = "\n") -> str:
    return joiner.join((original, new))
