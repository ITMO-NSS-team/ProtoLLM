import re


def fix_list_dots_separators(sentences: list[str]) -> list[str]:
    """
    Takes list of sentences and combines those of them that are list items
    that were incorrectly separated due to the use of a dot separator.
    Returns updated list of sentences
    """
    fixed_sentences_lst = []
    sentence_parts_lst = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i].strip()
        if len(chunk) > 0:
            # it means that the dot was used to separate list elements, and we should join such sentences
            if not chunk[0].isupper() and not chunk[0].isdigit():
                sentence_parts_lst.append(chunk)
            else:
                if len(sentence_parts_lst) != 0:
                    fixed_sentences_lst.append("; ".join(sentence_parts_lst))
                sentence_parts_lst = [chunk]
        i += 1

    if len(sentence_parts_lst) != 0:
        fixed_sentences_lst.append("; ".join(sentence_parts_lst))
    return fixed_sentences_lst
