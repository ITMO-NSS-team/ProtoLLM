import re
import warnings
from collections import Counter
from functools import reduce

import PyPDF2
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image
from ftfy import fix_text
from pdf2image import convert_from_path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTFigure, LTTextLine, LAParams
from tabulate import tabulate

from protollm.raw_data_processing.docs_parsers.utils.exceptions import (
    NoTextLayerError,
    ParseImageWarning,
    TitleExtractingWarning,
    PageNumbersExtractingWarning,
)
from protollm.raw_data_processing.docs_parsers.parsers.utilities import (
    HEADING_KEYWORDS,
    HEADING_STOP_LIST,
    FOOTER_KEYWORDS,
)

listmerge = lambda s: reduce(lambda d, el: d.extend(el) or d, s, [])


def text_extraction(element):
    """
    Main text extracting function from all elements in document layout
    :param element: element from the document's layout
    :return:
    """
    line_text = element.get_text()

    line_formats = []
    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            for character in text_line:
                if isinstance(character, LTChar):
                    line_formats.append(character.fontname)
                    line_formats.append(character.size)
    format_per_line = list(set(line_formats))
    return line_text, format_per_line


def extract_table(stream, page_num, table_num):
    """
    Function for extracting tables from the page
    :param stream: binary input
    :param page_num:
    :param table_num:
    :return:
    """
    pdf = pdfplumber.open(stream)
    table_page = pdf.pages[page_num]
    table = table_page.extract_tables()[table_num]

    return table


def convert_table_to_html(table) -> str:
    """
    Converts table to string in html format
    :param table:
    :return:
    """
    processed_table = [
        [item.replace("\n", " ") if item is not None else "" for item in row]
        for row in table
    ]

    return tabulate(processed_table, headers="firstrow", tablefmt="html")


def is_element_inside_any_table(element, page, tables):
    """
    Checks if the element is in any tables present in the page
    :param element:
    :param page:
    :param tables:
    :return:
    """
    x0, y0up, x1, y1up = element.bbox

    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for table in tables:
        tx0, ty0, tx1, ty1 = table.bbox
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return True
    return False


def find_table_for_element(element, page, tables):
    """
    Find the table's index for a given element
    :param element:
    :param page:
    :param tables:
    :return:
    """
    x0, y0up, x1, y1up = element.bbox
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for i, table in enumerate(tables):
        tx0, ty0, tx1, ty1 = table.bbox
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return i
    return None


def crop_image(element, pageObj):
    """
    Crops the image elements from PDFs
    :param element:
    :param pageObj:
    :return:
    """
    [image_left, image_top, image_right, image_bottom] = [
        element.x0,
        element.y0,
        element.x1,
        element.y1,
    ]

    pageObj.mediabox.lower_left = (image_left, image_bottom)
    pageObj.mediabox.upper_right = (image_right, image_top)

    cropped_pdf_writer = PyPDF2.PdfWriter()
    cropped_pdf_writer.add_page(pageObj)

    with open("cropped_image.pdf", "wb") as cropped_pdf_file:
        cropped_pdf_writer.write(cropped_pdf_file)


def convert_to_images(
    input_file,
):
    """
    Converts the PDF to images
    :param input_file:
    :return:
    """
    images = convert_from_path(input_file)
    image = images[0]
    output_file = "PDF_image.png"
    image.save(output_file, "PNG")


def image_to_text(image_path):
    # Read the image
    img = Image.open(image_path)
    # Extract the text from the image
    text = pytesseract.image_to_string(img)
    return text


def get_heading_info(element_info, heading_env, doc_info):
    element_text = element_info["element"].get_text().replace("\n", " ").strip()
    for stop_word in HEADING_STOP_LIST:
        if stop_word in element_text.lower():
            return -1

    special_symbols_cnt = 0
    letters_cnt = 0
    for char in element_text:
        if char.isalpha():
            letters_cnt += 1
        if not char.isalpha() and not char.isdigit() and not char == " ":
            special_symbols_cnt += 1
    if special_symbols_cnt > 4 or letters_cnt == 0:
        return -1

    if element_text == "":
        return -1

    # Get meta info about the element
    numeric_pref = get_numeric_prefix_str(element_info["element"])
    is_bold = element_info["meta"]["format"]["font_style"] == "bold"
    is_upper = element_info["element"].get_text().isupper()
    line_font = element_info["meta"]["format"]["font_name"]

    if (
        is_bold
    ):  # if the line starts with the numeric prefix and is bold - it is heading of the particular level
        if numeric_pref is not None:
            heading_lvl = len([x for x in numeric_pref.split(".") if x != ""])
            return heading_lvl
        else:
            if heading_env == 1:
                return heading_env  # it is a continuation of the heading
            if is_upper:
                return 1  # it is a first-level heading

    else:  # it is not bold
        if is_upper:
            if numeric_pref is not None:
                heading_lvl = len([x for x in numeric_pref.split(".") if x != ""])
                return heading_lvl
        if line_font != doc_info["font_name"]:
            if numeric_pref is not None:
                heading_lvl = len([x for x in numeric_pref.split(".") if x != ""])
                return heading_lvl
            # elif is_upper:
            #     return 1  # it is a first-level heading
            else:
                if heading_env == 1:
                    return heading_env  # it is a continuation of the heading

    for keyword in HEADING_KEYWORDS:
        element_text = element_info["element"].get_text().lower().strip()
        if element_text == keyword:
            return 1  # it is a first-level heading

    # If the element's font size refers to heading on some level
    element_size = element_info["meta"]["format"]["fontsize"]
    if element_size in doc_info["headings_sizes"]:
        return doc_info["headings_sizes"][element_size]

    return -1  # it is not a heading or a part of the heading


def get_numeric_prefix_str(text_line):
    line_text = text_line.get_text()
    numeric_pref = re.match(
        "^([0-9]+(\.)?)+", line_text
    )  # check if the line starts with the numeric prefix
    if numeric_pref is not None:
        return numeric_pref.group()
    else:
        return None


def check_layout(pages_layout):
    text_lines_stat = []
    for page_number, page in enumerate(pages_layout):
        if page_number > 10:
            break
        # Get text length statistics in all text elements
        for element in page:
            if isinstance(element, LTTextContainer):
                if not isinstance(
                    element, LTTextLine
                ):  # text element is a Box and should be unpacked to Lines
                    for line in element:
                        text_lines_stat.append(
                            len(line.get_text().split(" "))
                        )  # calculate number of words in parsed line
                else:
                    text_lines_stat.append(
                        len(element.get_text().split(" "))
                    )  # calculate number of words in parsed line
    avg_words_number = np.mean(text_lines_stat) if text_lines_stat else 0
    if avg_words_number < 4:
        return -1
    elif avg_words_number > 10:
        return 1
    else:
        return 0


def get_document_layout(stream, layout_parsing_params, tables_by_pages):
    # Get all line elements from the document, grouped by pages
    pages_layout = extract_pages(stream, laparams=layout_parsing_params)
    check_layout_res = check_layout(pages_layout)

    if check_layout_res == 1:
        layout_parsing_params.word_margin = layout_parsing_params.word_margin + 2
    elif check_layout_res == -1:
        layout_parsing_params.word_margin = 0.1
    pages_layout = extract_pages(stream, laparams=layout_parsing_params)

    pages_structure = []  # variable for the updated document's structure by pages

    for page_number, page in enumerate(pages_layout):
        page_structure = []

        # Get all tables from the page
        page_tables = tables_by_pages[page_number]

        # Analyze all elements on the page
        for element in page:
            if isinstance(element, LTTextContainer):
                text_lines_lst = []
                if not isinstance(
                    element, LTTextLine
                ):  # text element is a Box and should be unpacked to Lines
                    for line in element:
                        text_lines_lst.append(line)
                else:
                    text_lines_lst.append(element)

                for line in text_lines_lst:
                    if page_tables is not None:  # if there are any tables on the page
                        if is_element_inside_any_table(line, page, page_tables):
                            table_id_found = find_table_for_element(
                                line, page, page_tables
                            )
                            if (
                                table_id_found is not None
                            ):  # text element is a part of the table
                                page_structure.append(
                                    {
                                        "element": line,
                                        "meta": {"type": "table", "id": table_id_found},
                                    }
                                )
                                continue
                    # text line is not a part of the table
                    page_structure.append(
                        {"element": line, "meta": {"type": "text", "id": -1}}
                    )
            elif isinstance(element, LTFigure):  # element is an image
                page_structure.append(
                    {"element": element, "meta": {"type": "image", "id": -1}}
                )
        pages_structure.append(page_structure)

    # check if the first page is a title page
    # for element_info in pages_structure[0]:
    #     element = element_info['element']
    #     if isinstance(element, LTTextContainer):
    #         for keyword in HEADING_STOP_LIST:
    #             if keyword in element.get_text():
    #                 return pages_structure[1:]  # return document without title
    return pages_structure


def get_document_formatting(pages_structure):
    # margin_inf = 1000  # maximum value of the left margin attribute

    # Initialize main variables for the document formatting and layout structure
    doc_info = {}
    doc_structure = []

    # Variables for the whole document's formatting analysis
    doc_font_size_counter = Counter()
    doc_line_spacing_counter = Counter()
    doc_left_margin_counter = Counter()
    # for the cases, when information about the particular font is unavailable and there are only CID Fonts
    doc_main_font_counter = Counter()

    doc_headings_sizes = []  # list of headings' font sizes in hierarchical order

    for page in pages_structure:
        page_structure = []
        prev_line_bottom_border = (
            None  # y coordinate of the previous line's bottom border
        )

        for element_info in page:
            element_format_info = {}  # dict with info about text formatting
            elem_font_size_counter = Counter()
            elem_font_name_counter = Counter()

            elem_start_symbol = ""
            elem_font_style = "plain"

            # process only text elements (including ones which are parts of tables)
            if (
                element_info["meta"]["type"] != "image"
                and element_info["element"].get_text() != ""
            ):
                element = element_info["element"]
                if prev_line_bottom_border is not None:
                    elem_line_spacing = (
                        prev_line_bottom_border - element.y0
                    )  # space between the lines
                else:
                    elem_line_spacing = -1  # first line on the page

                prev_line_bottom_border = (
                    element.y1
                )  # update previous line bottom border attribute

                elem_left_margin = element.x0  # left margin of the line

                is_bold = True
                no_letters = True

                # Analyze element characters' formatting
                for character in element:
                    if character.get_text()[0].isalpha():
                        no_letters = False
                        if "Bold" not in character.fontname:
                            is_bold = False
                    if elem_start_symbol == "":
                        elem_start_symbol = (
                            "letter"
                            if character.get_text()[0].isalpha()
                            else (
                                "digit"
                                if character.get_text()[0].isdigit()
                                else "symbol"
                            )
                        )
                    if isinstance(character, LTChar):
                        elem_font_size_counter[round(character.size)] += 1  # font size
                        elem_font_name_counter[character.fontname] += 1  # font name

                # Get info about element's main font style (plain/bold)
                # if 'Bold' in elem_font_name_counter.most_common(1)[0][0] and len(elem_font_name_counter) == 1:
                #     elem_font_style = 'bold'
                if is_bold and not no_letters:
                    elem_font_style = "bold"

                # Get info about element's main font size
                elem_font_size = elem_font_size_counter.most_common(1)[0][0]
                elem_font_name = elem_font_name_counter.most_common(1)[0][0]

                # Update info about the element's formatting
                element_format_info["fontsize"] = elem_font_size
                element_format_info["font_style"] = elem_font_style
                element_format_info["font_name"] = elem_font_name
                element_format_info["left_margin"] = elem_left_margin
                element_format_info["line_spacing"] = elem_line_spacing
                element_format_info["start_symbol"] = elem_start_symbol

                # Update document's statistics
                doc_line_spacing_counter[
                    elem_line_spacing
                ] += 1  # add line spacing attribute
                doc_left_margin_counter[
                    elem_left_margin
                ] += 1  # add left margin attribute
                doc_font_size_counter[elem_font_size] += 1  # add font size attribute
                doc_main_font_counter[
                    elem_font_name
                ] += 1  # add font name info attribute

            element_info["meta"][
                "format"
            ] = element_format_info  # set format info for the element
            page_structure.append(element_info)
        doc_structure.append(page_structure)

    doc_font_size = doc_font_size_counter.most_common(1)[0][
        0
    ]  # main document's font size
    doc_line_spacing = doc_line_spacing_counter.most_common(1)[0][
        0
    ]  # main document's line spacing
    doc_left_margins = [
        x[0] for x in doc_left_margin_counter.most_common(2)
    ]  # two most common left margins
    doc_main_font = doc_main_font_counter.most_common(1)[0][0]

    for font_size in [x[0] for x in doc_font_size_counter.items()]:
        if font_size > doc_font_size:
            doc_headings_sizes.append(font_size)
    doc_headings_sizes.sort(reverse=True)  # sort font sizes in the descending order

    # Create dictionary to get heading level by heading font size
    doc_headings_sizes_dict = {}
    for i in range(len(doc_headings_sizes)):
        doc_headings_sizes_dict[doc_headings_sizes[i]] = i + 1

    # Prepare info about whole document's formatting
    doc_info["font_size"] = doc_font_size
    doc_info["font_name"] = doc_main_font
    doc_info["line_spacing"] = doc_line_spacing
    doc_info["left_margin"] = doc_left_margins
    doc_info["headings_sizes"] = doc_headings_sizes_dict

    return doc_info, doc_structure


def is_heading_correct(heading_str):
    if heading_str[0].isalpha() and heading_str[0].islower():
        return False

    if not heading_str[0].isalpha() and not heading_str[0].isdigit():
        return False

    if len(heading_str) > 128:
        return False

    return True


def extract_by_lines(
    stream,
    parse_images=False,
    parse_tables=True,
    parse_formulas=False,
    remove_service_info=False,
) -> tuple[list[str], list[dict]]:
    """
    Parses given pdf document to lines content and meta
    :param parse_images:
    :param parse_tables:
    :param parse_formulas:
    :param stream:
    :return:
    """
    document_content = []
    document_meta = []

    # Set up hyperparameters for the document's layout parsing by lines
    params = LAParams(
        line_overlap=0.5,
        char_margin=15.0,
        line_margin=0.1,
        word_margin=2.0,
        boxes_flow=1.0,
        detect_vertical=False,
        all_texts=True,
    )

    # Tables processing
    tables_by_pages = []
    tables_reader = pdfplumber.open(stream)

    # Extract all tables from the document using pdfplumber
    for page in tables_reader.pages:
        tables_by_pages.append(page.find_tables())

    # Get all line elements, grouped by pages, with the meta about the types: text, table or image
    pages_layout = get_document_layout(stream, params, tables_by_pages)

    is_text_in_doc = False

    for page_layout in pages_layout:
        for element_info in page_layout:
            if (
                element_info["meta"]["type"] == "text"
                or element_info["meta"]["type"] == "table"
            ):
                is_text_in_doc = True
                break
        if is_text_in_doc:
            break

    if not is_text_in_doc:
        raise NoTextLayerError("Document contains no text layer, only images")

    # Set up pdf reader for working with images
    img_pdf_reader = PyPDF2.PdfReader(stream)

    # Get info about the general document's formatting and each element's formatting
    doc_info, doc_structure = get_document_formatting(pages_layout)

    # Set up environmental variables
    heading_env = (
        -1
    )  # level of the current heading's environment (-1 if not the heading's environment)
    heading_lst = []
    current_heading_lvl = -1  # means that the element is not the heading (basic)
    tables_analysed = [set() for _ in range(len(doc_structure))]
    paragraph_id = 0
    headings_hierarchy = []

    for page_number, page in enumerate(doc_structure):
        document_content.append([])
        document_meta.append([])
        if parse_images:
            try:
                page_object = img_pdf_reader.pages[page_number]
            except IndexError:
                warnings.warn(
                    "Error in pages structure formatting", category=ParseImageWarning
                )
                page_object = None

        for element_info in page:
            is_heading = 0
            correct_heading = True
            headings = []
            paragraph = -1

            element = element_info["element"]
            element_meta = element_info["meta"]
            element_text = ""

            if element_meta["type"] == "text":
                element_text = fix_text(element.get_text().replace("\n", " ").strip())
                heading_lvl = get_heading_info(element_info, heading_env, doc_info)

                if heading_lvl != -1:  # text line is a part of some heading
                    if heading_lvl == heading_env:  # continuation
                        heading_lst.append(element_text)  # continue the heading
                    else:  # heading of another level or the new heading after other elements
                        if len(heading_lst) != 0:
                            # we should add info about the previous heading to the 'headings' list
                            headings_hierarchy = headings_hierarchy[: heading_env - 1]
                            heading_str = " ".join(heading_lst).strip()
                            if is_heading_correct(heading_str):
                                headings_hierarchy.append(heading_str.strip())
                            else:
                                correct_heading = False
                        heading_lst = [element_text]  # starting new heading
                        heading_env = heading_lvl

                    is_heading = 1
                else:  # text line is not a part of any heading
                    is_heading = 0
                    if heading_env != -1:  # previous element contained heading
                        paragraph_id += 1  # new paragraph
                        if len(heading_lst) != 0:
                            # we should add info about the previous heading to the 'headings' list
                            headings_hierarchy = headings_hierarchy[: heading_env - 1]
                            heading_str = " ".join(heading_lst).strip()
                            if is_heading_correct(heading_str):
                                headings_hierarchy.append(heading_str)
                            else:
                                correct_heading = False
                            heading_lst = []
                            heading_env = -1
                    else:  # plain text and previous text line was not a heading
                        # figuring out, should we start a new paragraph
                        element_spacing = element_info["meta"]["format"]["line_spacing"]
                        element_margin = element_info["meta"]["format"]["left_margin"]
                        start_symbol = element_info["meta"]["format"]["start_symbol"]

                        # if the line spacing is bigger than average or left margin is bigger than average,
                        # and it is not a list element (starts with letter or digit)
                        if start_symbol != "symbol":
                            if (
                                element_spacing > 1.1 * doc_info["line_spacing"]
                                or element_margin > doc_info["left_margin"][0]
                            ):
                                paragraph_id += 1  # new paragraph

                    headings = headings_hierarchy
                    paragraph = paragraph_id

            else:
                if heading_env != -1:  # previous element contained heading
                    paragraph_id += 1  # new paragraph
                    if len(heading_lst) != 0:
                        # we should add info about the previous heading to the 'headings' list
                        headings_hierarchy = headings_hierarchy[: heading_env - 1]
                        heading_str = " ".join(heading_lst).strip()
                        if is_heading_correct(heading_str):
                            headings_hierarchy.append(heading_str)
                        else:
                            correct_heading = False
                        heading_lst = []
                        heading_env = -1

                if element_meta["type"] == "table":
                    table_id = element_meta["id"]
                    if table_id not in tables_analysed[page_number] and parse_tables:
                        tables_analysed[page_number].add(table_id)
                        table = extract_table(stream, page_number, table_id)

                        # add string with table content to the document content
                        element_text = convert_table_to_html(table)

                        # prepare meta info for the table element
                        is_heading = 0
                        headings = headings_hierarchy
                        paragraph = paragraph_id
                else:  # element_meta['type'] == 'image'
                    if parse_images:
                        if page_object is not None:
                            crop_image(element["element"], page_object)
                            convert_to_images("cropped_image.pdf")
                            image_text = image_to_text("PDF_image.png")

                            # add recognized text from image to the document content
                            element_text = image_text

                        # prepare meta info for the image element
                        is_heading = 0  # only text element can be a heading
                        headings = headings_hierarchy
                        # paragraph_id += 1  # image related to the new paragraph
                        paragraph = paragraph_id

            if element_text != "" and (is_heading != 0 or paragraph != -1):
                if remove_service_info:
                    is_footer = False
                    for footer_key in FOOTER_KEYWORDS:
                        if footer_key in element_text.lower():
                            is_footer = True
                            break
                    if is_footer:
                        continue

                try:
                    element_text = element_text.encode("cp1252").decode("cp1251")
                    headings = [x.encode("cp1252").decode("cp1251") for x in headings]
                except (UnicodeDecodeError, UnicodeEncodeError):
                    pass

                document_content[-1].append(element_text)
                document_meta[-1].append(
                    {
                        "type": element_meta["type"],
                        "is_heading": is_heading,
                        "is_heading_extracting_correct": correct_heading,
                        "headings": headings,
                        "paragraph": paragraph,
                    }
                )

    if remove_service_info:
        # Check if the first document's page is a title
        try:
            potential_title_page = document_content[0]
            title_num = 0
            is_title = False

            for i in range(len(document_content)):
                page = document_content[i]
                if len(page) > 0:
                    potential_title_page = page
                    title_num = i
                    break
            for text_line in potential_title_page:
                for title_key in HEADING_STOP_LIST:
                    if title_key in text_line.lower():
                        is_title = True
                        break
                if is_title:
                    break
            if not is_title:
                for char in potential_title_page[-1].lower():
                    if char.isalpha() and char != "г":
                        break
            if is_title:
                document_content = document_content[title_num + 1 :]
                document_meta = document_meta[title_num + 1 :]
        except IndexError:
            warnings.warn(
                "Can not skip title-related service information due to unknown title formatting",
                category=TitleExtractingWarning,
            )

        # Remove page numbers
        try:
            for page_number in range(len(document_content)):
                page = document_content[page_number]
                if len(page) > 0:
                    only_digits = True
                    for char in page[0]:  # number is at the beginning of the page
                        if not char.isdigit():
                            only_digits = False
                    if only_digits:
                        document_content[page_number] = document_content[page_number][
                            1:
                        ]
                        document_meta[page_number] = document_meta[page_number][1:]
                        continue
                    for char in page[-1]:  # number is at the end of the page
                        if not char.isdigit():
                            only_digits = False
                    if only_digits:
                        document_content[page_number] = document_content[page_number][
                            :-1
                        ]
                        document_meta[page_number] = document_meta[page_number][:-1]
                        continue
        except IndexError:
            warnings.warn(
                "Can not delete page numbers due to unknown formatting",
                category=PageNumbersExtractingWarning,
            )

        # # Remove footers
        # try:
        #     for page_number in range(len(document_content)):
        #         page = document_content[page_number]
        #         if len(page) > 0:
        #
        # except IndexError:
        #     warnings.warn('Can not delete page numbers due to unknown formatting',
        #                   category=FooterExtractingWarning)

    final_content = listmerge(document_content)
    final_meta = listmerge(document_meta)

    return final_content, final_meta
