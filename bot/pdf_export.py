"""Render a vacancy as a standalone PDF -- a "printout" of the job ad
alongside its generated ansøgning, e.g. for jobcenter/kommune reporting
that expects proof of what was actually applied to.
"""

import html
import re
from pathlib import Path

from fpdf import FPDF

FONTS_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
REGULAR_FONT = FONTS_DIR / "DejaVuSans.ttf"
BOLD_FONT = FONTS_DIR / "DejaVuSans-Bold.ttf"


def _strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)  # &amp; -> &, &nbsp; -> space, etc.
    # Some job ads' raw text carries markdown-style asterisks that were
    # never meant to render literally in a plain PDF -- not always as a
    # matched **pair** either (e.g. "**Om rollen" starting several lines
    # with no closing **), so just strip every run of them outright.
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def vacancy_to_pdf(vacancy, output_path: str, contact=None):
    pdf = FPDF()
    pdf.add_font("DejaVu", "", str(REGULAR_FONT))
    pdf.add_font("DejaVu", "B", str(BOLD_FONT))
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    if contact is not None:
        # One summary block with everything (including fields that would
        # otherwise repeat below, like title/company/link) -- previously
        # this and the plain meta block under the title duplicated those
        # three fields.
        pdf.set_font("DejaVu", "", 11)
        summary_lines = [
            ("Job", vacancy.title),
            ("Virksomhed", vacancy.company),
            ("Sted", vacancy.location),
            ("Kilde", vacancy.source),
            ("Kontaktperson", contact.name),
            ("Telefon", contact.phone),
            ("Email", contact.email),
            ("Link", vacancy.url),
        ]
        for label, value in summary_lines:
            if not value:
                continue
            pdf.set_font("DejaVu", "B", 11)
            pdf.write(6, f"{label}: ")
            pdf.set_font("DejaVu", "", 11)
            pdf.write(6, value)
            pdf.ln(7)
        pdf.ln(6)
    else:
        pdf.set_font("DejaVu", "B", 14)
        pdf.multi_cell(0, 8, vacancy.title)
        pdf.ln(2)

        pdf.set_font("DejaVu", "", 11)
        meta_lines = [
            ("Virksomhed", vacancy.company),
            ("Sted", vacancy.location),
            ("Kilde", vacancy.source),
            ("Link", vacancy.url),
        ]
        for label, value in meta_lines:
            if not value:
                continue
            pdf.set_font("DejaVu", "B", 11)
            pdf.write(6, f"{label}: ")
            pdf.set_font("DejaVu", "", 11)
            pdf.write(6, value)
            pdf.ln(7)

        pdf.ln(4)
    pdf.set_font("DejaVu", "B", 12)
    pdf.multi_cell(0, 7, "Jobbeskrivelse")
    pdf.ln(1)

    pdf.set_font("DejaVu", "", 10.5)
    pdf.multi_cell(0, 6, _strip_html(vacancy.description) or "(ingen beskrivelse)")

    pdf.output(output_path)
    return output_path


def letter_to_pdf(letter_text: str, vacancy, output_path: str):
    pdf = FPDF()
    pdf.add_font("DejaVu", "", str(REGULAR_FONT))
    pdf.add_font("DejaVu", "B", str(BOLD_FONT))
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    pdf.set_font("DejaVu", "B", 13)
    pdf.multi_cell(0, 7, f"Ansøgning — {vacancy.title}")
    pdf.ln(6)

    pdf.set_x(pdf.l_margin)
    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 6, letter_text)

    pdf.output(output_path)
    return output_path


def cv_to_pdf(cv_text: str, summary: str, output_path: str) -> str:
    """A ready-to-submit CV PDF: the tailored profile up top, followed by
    the person's own CV content -- so applying doesn't require anyone to
    open a PDF, copy the summary out by hand, and re-save it as a CV
    themselves.
    """
    pdf = FPDF()
    pdf.add_font("DejaVu", "", str(REGULAR_FONT))
    pdf.add_font("DejaVu", "B", str(BOLD_FONT))
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    pdf.set_font("DejaVu", "B", 13)
    pdf.multi_cell(0, 7, "Profil")
    pdf.ln(2)

    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 6, summary.strip())
    pdf.ln(8)

    pdf.set_font("DejaVu", "", 10.5)
    pdf.multi_cell(0, 6, cv_text.strip())

    pdf.output(output_path)
    return output_path
