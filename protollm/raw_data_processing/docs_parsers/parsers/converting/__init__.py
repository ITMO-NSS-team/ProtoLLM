from functools import partial

from protollm.raw_data_processing.docs_parsers.parsers.converting.converted_file import (
    converted_file as __converted_file,
)
from protollm.raw_data_processing.docs_parsers.parsers.entities import ConvertingDocType

converted_file_to_docx = partial(
    __converted_file, target_doc_type=ConvertingDocType.docx
)
