import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Union

from langchain_core.document_loaders import Blob
from langchain_core.documents import Document

from protollm.raw_data_processing.docs_parsers.parsers.entities import DocType


class BaseParser(ABC):
    """
    Abstract interface for parsers.

    A parser provides a way to parse raw data into one or more documents.
    """

    @abstractmethod
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazy parsing interface.

        Subclasses are required to implement this method.

        Args:
            blob: representation of raw data from file

        Returns:
            Generator of documents
        """

    def parse(self, blob: Blob) -> List[Document]:
        """Eagerly parse raw data into a document or documents.

        This is a convenience method for interactive development environment.

        Production applications should favor the lazy_parse method instead.

        Subclasses should generally not over-ride this parse method.

        Args:
            blob: representation of raw data from file

        Returns:
            List of documents
        """
        return list(self.lazy_parse(blob))

    @staticmethod
    def get_doc_type(file: Union[str, Path]) -> DocType:
        mimetype = mimetypes.guess_type(file)[0]
        if mimetype is None:
            mimetype = Path(file).suffix.replace(".", "")

        match mimetype:
            case "application/pdf" | "pdf":
                return DocType.pdf
            case (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                | "docx"
            ):
                return DocType.docx
            case "application/msword" | "doc":
                return DocType.doc
            case "application/vnd.oasis.opendocument.text" | "odt":
                return DocType.odt
            case "application/rtf" | "rtf":
                return DocType.rtf
            case (
                "application/zip"
                | "application/x-zip-compressed"
                | "multipart/x-zip"
                | "zip"
            ):
                return DocType.zip
            case _:  # TODO: add txt, xlsx, pptx support
                return DocType.unsupported
