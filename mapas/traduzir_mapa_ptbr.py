"""Create a personal-use PT-BR derivative of the supplied Inner Sea PDF.

Only generic geographic labels are translated; proper names stay canonical.
The source PDF is never modified.
"""
from pathlib import Path
import re
import fitz
import pytesseract
from PIL import Image

SOURCE = Path(r"D:\Users\rapha\Documents\Projetos\RPG\livros\pathfinder-rpg-poster-map-folio-inner-sea-biblioteca-elfica.pdf")
OUTPUT = SOURCE.with_name("inner-sea-poster-map-folio-ptbr.pdf")
TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TRANSLATIONS = {
    "SEA": "MAR", "OCEAN": "OCEANO", "GULF": "GOLFO", "STRAIT": "ESTREITO",
    "COAST": "COSTA", "RIVER": "RIO", "LAKE": "LAGO", "MOUNTAINS": "MONTANHAS",
    "MOUNTAIN": "MONTANHA", "FOREST": "FLORESTA", "DESERT": "DESERTO",
    "PLAINS": "PLANÍCIES", "SWAMP": "PÂNTANO", "KINGDOM": "REINO",
    "EMPIRE": "IMPÉRIO", "NATION": "NAÇÃO", "CITY": "CIDADE", "RUINS": "RUÍNAS",
    "ISLAND": "ILHA", "ISLANDS": "ILHAS", "PENINSULA": "PENÍNSULA", "BAY": "BAÍA",
}

def main():
    if not SOURCE.exists():
        raise SystemExit(f"PDF não encontrado: {SOURCE}")
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT)
    source = fitz.open(SOURCE)
    output = fitz.open()
    replacements = 0
    for page in source:
        new_page = output.new_page(width=page.rect.width, height=page.rect.height)
        new_page.show_pdf_page(new_page.rect, source, page.number)
        pix = page.get_pixmap(dpi=220, alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        data = pytesseract.image_to_data(image, lang="eng+por", config="--psm 11", output_type=pytesseract.Output.DICT)
        scale_x, scale_y = page.rect.width / pix.width, page.rect.height / pix.height
        page_replacements = []
        for i, raw in enumerate(data["text"]):
            token = re.sub(r"[^A-Za-zÀ-ÿ]", "", raw or "").upper()
            translated = TRANSLATIONS.get(token)
            if not translated or int(data["conf"][i]) < 55:
                continue
            rect = fitz.Rect(
                data["left"][i] * scale_x, data["top"][i] * scale_y,
                (data["left"][i] + data["width"][i]) * scale_x,
                (data["top"][i] + data["height"][i]) * scale_y,
            )
            rect += (-1, -1, 1, 1)
            page_replacements.append((rect, translated))
        # Apply all masks once per page; applying inside the OCR loop can
        # invalidate annotations that were added earlier on the same page.
        for rect, _ in page_replacements:
            new_page.add_redact_annot(rect, fill=(0.82, 0.85, 0.72))
        if page_replacements:
            new_page.apply_redactions()
        for rect, translated in page_replacements:
            replacements += 1
            new_page.insert_textbox(
                rect, translated, fontsize=max(4, min(10, rect.height * 0.75)),
                color=(0.12, 0.16, 0.12), align=0,
            )
    output.set_metadata({"title": "Inner Sea Poster Map Folio — guia PT-BR (uso pessoal)", "subject": "Tradução parcial de rótulos genéricos; nomes próprios preservados"})
    output.save(OUTPUT, garbage=4, deflate=True)
    print(f"Criado: {OUTPUT}")
    print(f"Rótulos traduzidos: {replacements}")

if __name__ == "__main__":
    main()
