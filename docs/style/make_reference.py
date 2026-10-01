"""Restyle pandoc's default reference.docx for the Korean briefing: Malgun Gothic, compact spacing,
bordered tables with a shaded header row, 2 cm margins.
    uv run --isolated --no-project --with python-docx python docs/style/make_reference.py"""
import os
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.docx")
d = Document(P)
for s in d.sections:
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Cm(2.0)


def font(style, size=None, bold=None, color=None):
    f = style.font
    f.name = "Malgun Gothic"
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(a), "Malgun Gothic")
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rfonts.get(qn(a)) is not None:
            del rfonts.attrib[qn(a)]
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if color:
        f.color.rgb = RGBColor.from_string(color)


names = {s.name for s in d.styles}
for n in ("Normal", "Body Text", "First Paragraph", "Compact"):
    if n in names:
        font(d.styles[n], 10)
        pf = d.styles[n].paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(2), Pt(4), 1.25
# table cells (pandoc's "Compact"): tight
pf = d.styles["Compact"].paragraph_format
pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(0), 1.05
font(d.styles["Title"], 18, True, "1F3864")
for n, size, color in (("Heading 1", 14, "1F3864"), ("Heading 2", 12, "2E5597"), ("Heading 3", 11, "2E5597")):
    font(d.styles[n], size, True, color)
    pf = d.styles[n].paragraph_format
    pf.space_before, pf.space_after = Pt(14 if n == "Heading 1" else 10), Pt(4)

# table style used by pandoc ("Table"): borders, cell padding, shaded header row
t = d.styles["Table"]
tblPr = t.element.find(qn("w:tblPr"))
if tblPr is None:
    tblPr = OxmlElement("w:tblPr")
    t.element.append(tblPr)
borders = OxmlElement("w:tblBorders")
for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
    e = OxmlElement(f"w:{side}")
    e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "A6A6A6")
    borders.append(e)
tblPr.append(borders)
mar = OxmlElement("w:tblCellMar")
for side, w in (("top", 40), ("bottom", 40), ("left", 90), ("right", 90)):
    e = OxmlElement(f"w:{side}")
    e.set(qn("w:w"), str(w)); e.set(qn("w:type"), "dxa")
    mar.append(e)
tblPr.append(mar)
hdr = OxmlElement("w:tblStylePr"); hdr.set(qn("w:type"), "firstRow")
tc = OxmlElement("w:tcPr"); shd = OxmlElement("w:shd")
shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "DCE6F2")
tc.append(shd); rpr = OxmlElement("w:rPr"); b = OxmlElement("w:b"); rpr.append(b)
hdr.append(rpr); hdr.append(tc)
t.element.append(hdr)
d.save(P)
print("saved", P)
