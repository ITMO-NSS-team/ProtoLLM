from enum import Enum


class DocType(str, Enum):
    docx = "docx"
    doc = "doc"
    odt = "odt"
    rtf = "rtf"
    pdf = "pdf"
    zip = "zip"
    unsupported = "unsupported"  # TODO: add txt, xlsx, pptx support


class ConvertingDocType(str, Enum):
    docx = "docx"  # TODO: add xlsx, pptx support


class ParsingScheme(str, Enum):
    paragraphs = "paragraphs"
    lines = "lines"
    chapters = "chapters"
    full = "full"
