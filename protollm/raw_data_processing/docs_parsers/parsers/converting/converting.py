import os
import subprocess
from pathlib import Path
from typing import Union

from protollm.raw_data_processing.docs_parsers.utils.exceptions import ConvertingError


def _convert_with_soffice(
    filename: Union[Path, str],
    output_directory: Union[Path, str],
    target_doc_type: str = "docx",
    timeout: int = None,
):
    """
    Converts a file to a target format using the libreoffice CLI.
    """
    command = [
        "soffice",
        "--headless",
        "--convert-to",
        target_doc_type,
        "--outdir",
        output_directory,
        str(filename),
    ]
    expected_path = Path(
        output_directory, ".".join((Path(filename).stem, target_doc_type))
    )
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
        )
        try:
            error = result.stderr.decode().strip()
        except UnicodeError:
            error = "*** Error message cannot be displayed ***"
        if error and not os.path.isfile(expected_path):
            raise ConvertingError(
                f"Could not convert file to {target_doc_type}\n{error}"
            )
    except subprocess.TimeoutExpired:
        raise ConvertingError(
            f"Converting file to {target_doc_type} hadn't terminated after {timeout} seconds"
        ) from None
    except FileNotFoundError:
        raise ConvertingError(
            "soffice command was not found. Please install libreoffice on your system and try again."
        ) from None
