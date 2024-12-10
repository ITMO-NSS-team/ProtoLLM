from typing import Any, Iterable, Optional, Sequence
import re

from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter

from protollm.docs_processing.splitting.utilities import fix_list_dots_separators, split_hierarchical_sentences


class ListHierarchySplitter(TextSplitter):
    def __init__(self,
                 markers: Optional[Sequence[str]] = None,
                 separators: Optional[Iterable[str]] = None,
                 lst_hierarchy_split_patterns: Optional[Iterable[str]] = None,
                 apply_chunks_merge: bool = False,
                 **kwargs: Any
                 ):
        super().__init__(**kwargs)
        self._markers = markers or ['-', '*', '–', '•']
        self._separators = separators or [r'\. (?<![0-9]\. | ^[0-9].)',]
        self._lst_hierarchy_split_patterns = lst_hierarchy_split_patterns or [':[ ]*', ';[ ]*',]
        self._apply_chunks_merge = apply_chunks_merge

    def _merge_chunks(self, chunks):
        merged_chunk = ''
        upd_chunks_lst = []
        for chunk in chunks:
            if self._length_function(merged_chunk + chunk) <= self._chunk_size:
                merged_chunk += ' ' + chunk
            else:
                upd_chunks_lst.append(merged_chunk)
                merged_chunk = chunk
        if len(upd_chunks_lst) == 0:
            upd_chunks_lst = chunks
        return upd_chunks_lst

    def split_text(self, text: str) -> list[str]:
        sentences_lst = [x.strip() for x in re.split(self._separators[0], text)]

        upd_sentences_lst = fix_list_dots_separators(sentences_lst)
        chunks = split_hierarchical_sentences(upd_sentences_lst, self._markers, self._lst_hierarchy_split_patterns)

        upd_splits = []

        for s in chunks:
            if self._length_function(s) < self._chunk_size:
                upd_splits.append(s)
            else:
                sentence_splitter = RecursiveCharacterTextSplitter(chunk_size=self._chunk_size,
                                                                   chunk_overlap=self._chunk_overlap,
                                                                   separators=self._separators)
                smaller_chunks = sentence_splitter.split_text(s)
                upd_splits.extend(smaller_chunks)
        if self._apply_chunks_merge:
            final_splits = self._merge_chunks(upd_splits)
        else:
            final_splits = upd_splits
        return [x.replace('\n', ' ') for x in final_splits]
