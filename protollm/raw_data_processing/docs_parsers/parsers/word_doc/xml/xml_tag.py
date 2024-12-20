from enum import Enum


_namespace_mapping = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _get_xml_tag_name(tag: str, namespace: str) -> str:
    namespace = _namespace_mapping[namespace]
    return f"{{{namespace}}}{tag}"


class XMLTag(str, Enum):
    raw = _get_xml_tag_name("r", "w")
    text = _get_xml_tag_name("t", "w")
    image = _get_xml_tag_name("drawing", "w")
    blip = _get_xml_tag_name("blip", "a")
    embed = _get_xml_tag_name("embed", "r")
    math = _get_xml_tag_name("oMath", "m")
    math_paragraph = _get_xml_tag_name("oMathPara", "m")
