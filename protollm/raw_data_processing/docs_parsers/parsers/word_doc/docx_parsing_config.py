from docx.document import Document

from protollm.raw_data_processing.docs_parsers.parsers.word_doc.xml.utilities import (
    _get_omml2mml_transformation,
    _get_mml2tex_transformation,
)


class DocxParsingConfig:
    def __init__(
        self,
        document: Document,
        extract_images: bool = False,
        parse_formulas: bool = False,
    ):
        self.__rels = document.part.rels
        self.__extract_images = extract_images
        self.__parse_formulas = parse_formulas
        self.__omml2mml = None
        self.__mml2tex = None

    @property
    def extract_images(self):
        return self.__extract_images

    @property
    def parse_formulas(self):
        return self.__parse_formulas

    @property
    def document_relationships(self):
        return self.__rels

    @property
    def omml2mml_transformation(self):
        if self.__omml2mml is None and self.__parse_formulas:
            self.__omml2mml = _get_omml2mml_transformation()

        return self.__omml2mml

    @property
    def mml2tex_transformation(self):
        if self.__mml2tex is None and self.__parse_formulas:
            self.__mml2tex = _get_mml2tex_transformation()

        return self.__mml2tex
