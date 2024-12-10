from functools import partial
from docs_processing.parsing.parsers.converting.converted_file import converted_file as __converted_file
from docs_processing.parsing.parsers.entities import ConvertingDocType

converted_file_to_docx = partial(__converted_file, target_doc_type=ConvertingDocType.docx)
