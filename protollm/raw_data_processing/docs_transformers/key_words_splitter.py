import logging
import os
from typing import Any, Iterable, Optional, List, Tuple

import spacy
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

logger = logging.getLogger(__name__)


class MultiMetadataAppender(TextSplitter):
    def __init__(self, separators: Optional[Iterable[str]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._separators = separators or [r"\. (?<![0-9]\. | ^[0-9].)"]
        self.keyword_extractor = KeywordExtractor()

    def _create_document(self, text: str, metadata: dict[str, Any]) -> Document:
        text_len = self._length_function(text)
        if text_len > self._chunk_size:
            logger.warning(
                f"A chunk of size {text_len} was encountered, "
                f"which is larger than the specified {self._chunk_size}"
            )
        return Document(page_content=text, metadata=metadata)

    def _split_with_additional_metadata(
        self, documents: Iterable[Document]
    ) -> List[Document]:
        texts, metadatas = [], []
        for doc in documents:
            doc_metadata = doc.metadata
            doc_metadata["object"], doc_metadata["action"] = (
                self.keyword_extractor.get_object_action_pair(doc.page_content)
            )
            doc_metadata["keywords"] = self.keyword_extractor.get_keywords(
                doc.page_content
            )
            texts.append(doc.page_content)
            metadatas.append(doc_metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def split_text(self, text: str) -> List[str]:
        return [text]

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        return self._split_with_additional_metadata(documents)


class KeywordExtractor:
    def __init__(self):
        os.system("python -m spacy download ru_core_news_sm")
        self.nlp = spacy.load("ru_core_news_sm")

    def get_keywords(self, text: str) -> List[str]:
        tokens = self.nlp(text)
        subjects = []
        for token in tokens:
            if "nsubj" in token.dep_ or "obj" in token.dep_:
                subjects.append(token)
                break

        result_tokens = subjects.copy()

        for token in subjects:
            children = [tk for tk in token.children if tk.dep_ == "nmod"]
            result_tokens.extend(children)
            while len(children) > 0:
                ch = children.pop()
                tmp = [child for child in ch.children if child.dep_ == "nmod"]
                result_tokens.extend(tmp)
                children.extend(tmp)

        return list(set([token.lemma_ for token in result_tokens]))

    def get_object_action_pair(self, text: str) -> Tuple[str, str]:
        tokens = self.nlp(text)
        for token in tokens:
            if "nsubj" in token.dep_ or "obj" in token.dep_:
                return token.lemma_, token.head.lemma_
        return "", ""
