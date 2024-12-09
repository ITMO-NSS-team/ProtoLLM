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
                    fixed_sentences_lst.append('; '.join(sentence_parts_lst))
                sentence_parts_lst = [chunk]
        i += 1

    if len(sentence_parts_lst) != 0:
        fixed_sentences_lst.append('; '.join(sentence_parts_lst))
    return fixed_sentences_lst


def get_list_hierarchy_splitting_patterns(sentence, markers, split_patterns):
    """
       Takes a sentence and finds all list hierarchy splitting patterns
       based on the given list of marker symbols (markers)
       and a list of basic splitting patterns (split_patterns).
       Returns a list of strings with the list hierarchy splitting patterns in the order of nesting
    """
    hierarchy_lst = [x.strip() for x in sentence.split(':')]
    hierarchy_levels = len(hierarchy_lst)

    patterns = []
    if hierarchy_levels >= 1:  # is there a list hierarchy in the sentence
        depth = hierarchy_levels
        for i in range(1, depth):  # in the order of nesting
            current_item = hierarchy_lst[i]
            if len(current_item) == 0:
                break
            else:  # prepare patterns for list hierarchy splitting
                if current_item[0] in markers:  # if the current list item starts with the marker symbol
                    hierarchy_split_pattern = []
                    for split_pattern in split_patterns:
                        hierarchy_split_pattern.append(split_pattern + '\\' + current_item[0])
                    patterns.append('|'.join(hierarchy_split_pattern))
                elif ')' in current_item:  # if the current list item starts with the marker which contains bracket
                    hierarchy_split_pattern = []
                    if current_item.split(')')[0].isalpha():  # list marker is a letter with a bracket
                        for split_pattern in split_patterns:
                            hierarchy_split_pattern.append(split_pattern + r'[а-яa-zё]\)')
                    elif hierarchy_lst[i].split(')')[0].isdigit():  # list marker is a number with a bracket
                        for split_pattern in split_patterns:
                            hierarchy_split_pattern.append(split_pattern + r'[0-9]+\)')
                    if len(hierarchy_split_pattern) != 0:
                        patterns.append('|'.join(hierarchy_split_pattern))

    unique_patterns = []
    for pattern in patterns:
        if pattern not in unique_patterns:  # filter only unique patterns in the order of nesting
            unique_patterns.append(pattern)
    return unique_patterns


def join_levels_in_list_hierarchy(split_items_lst):
    """
       Takes a list of the hierarchical items in the order of nesting and joins them.
       Returns a list of combined hierarchical list items
    """
    if len(split_items_lst) == 1:
        return split_items_lst
    else:
        pref = split_items_lst[0]
        combined_items_lst = []
        for i in range(1, len(split_items_lst)):
            combined_items_lst.append(pref + split_items_lst[i])
        return combined_items_lst


def split_hierarchical_sentences(sentences_lst, markers, split_patterns):
    """
       Takes a list of the hierarchical items in the order of nesting and joins them.
       Returns a list of combined hierarchical list items
    """
    upd_sentences_lst = []
    for sentence in sentences_lst:
        # for each sentence we create a list of list hierarchy splitting patterns
        patterns_lst = get_list_hierarchy_splitting_patterns(sentence, markers, split_patterns)
        if len(patterns_lst) == 0:
            upd_sentences_lst.extend([sentence])  # we have no list hierarchy in the sentence
        else:
            items_lst = [sentence]
            for pattern in patterns_lst:
                upd_items_lst = []
                for sent in items_lst:
                    # combining levels in the list hierarchy in the order of nesting
                    items = join_levels_in_list_hierarchy(re.split(pattern, sent))
                    upd_items_lst.extend(items)
                items_lst = upd_items_lst
                # Проверка, что влезает в контекст. Если да - то break
            upd_sentences_lst.extend(items_lst)
    return upd_sentences_lst
