"""Erzeugt Screenshot und Demo-GIF des ausgeführten Notebooks für die README (docs/).

Das Notebook wird als HTML exportiert und im Browser durchgescrollt.
Aufruf (aus dem Projektordner):
  uv run --with playwright==1.56.0 --with pillow python scripts/demo_gif.py
"""

import io
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

PROJEKT = Path(__file__).resolve().parents[1]
DOCS = PROJEKT / "docs"
# Überschriften, an denen das GIF anhält (Reihenfolge wie im Notebook)
STATIONEN = [
    ("Trace Kunden – Customer Analytics & Churn-Prognose", 2200),
    ("1 · Bereinigung", 2600),
    ("Umsatz pro Monat", 2200),
    ("K-Means: Wie viele Segmente?", 2200),
    ("Die vier Segmente", 1400),
    ("Das Stichtag-Design und warum es nötig ist", 2600),
    ("Was die Zahlen sagen", 1400),
    ("Iteration: Warum 11 und nicht 14 Merkmale?", 2600),
    ("Was ist eine Rückgewinnungskampagne wert?", 2000),
    ("Ergebnis der Kampagnenrechnung", 3000),
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["uv", "run", "jupyter", "nbconvert", "--to", "html", "--output-dir", tmp,
                        str(PROJEKT / "notebooks" / "kundenanalyse.ipynb")], check=True, capture_output=True)
        html = Path(tmp) / "kundenanalyse.html"
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(html.as_uri())
            page.wait_for_timeout(800)
            page.screenshot(path=DOCS / "00-notebook.png")
            bilder: list[tuple[Image.Image, int]] = []
            for text, dauer in STATIONEN:
                ziel = page.get_by_text(text, exact=False).first
                ziel.evaluate("e => window.scrollTo(0, e.getBoundingClientRect().top + window.scrollY - 24)")
                page.wait_for_timeout(300)
                img = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
                img.thumbnail((1000, 10_000))
                bilder.append((img, dauer))
            browser.close()
    rahmen = [b.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for b, _ in bilder]
    rahmen[0].save(DOCS / "demo.gif", save_all=True, append_images=rahmen[1:],
                   duration=[d for _, d in bilder], loop=0, optimize=True)
    print("fertig:", DOCS / "demo.gif")


if __name__ == "__main__":
    main()
