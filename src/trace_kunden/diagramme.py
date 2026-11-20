"""Einheitlicher Diagramm-Stil in Tracewell-Farben und Export nach docs/."""

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from trace_kunden.konfig import DOCS, FARBEN, KATEGORIAL

# Sequenzielle Skala für Größen (Heatmaps): eine Farbe, hell → dunkel.
ZIEGEL_SKALA = LinearSegmentedColormap.from_list("ziegel", ["#fbf6f3", FARBEN["ziegel_hell"], "#c9785a",
                                                            FARBEN["ziegel"], "#5e2f1d"])


def stil_setzen() -> None:
    """Setzt flache, ruhige Diagramme: dünne Achsen, kein Rahmen oben/rechts, gedämpfte Beschriftung."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": FARBEN["gedaempft"],
        "axes.edgecolor": FARBEN["linie"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": FARBEN["linie"],
        "grid.linewidth": 0.6,
        "xtick.color": FARBEN["gedaempft"],
        "ytick.color": FARBEN["gedaempft"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "text.color": FARBEN["tinte"],
        "legend.frameon": False,
        "lines.linewidth": 2,
        "axes.prop_cycle": mpl.cycler(color=KATEGORIAL),
    })


def speichern(fig: Figure, name: str, ordner: Path = DOCS) -> Path:
    """Speichert ein Diagramm als PNG für die README und gibt den Pfad zurück."""
    ordner.mkdir(exist_ok=True)
    pfad = ordner / f"{name}.png"
    fig.savefig(pfad, bbox_inches="tight")
    return pfad


def tausender(x: float, _pos: int | None = None) -> str:
    """Achsenformat mit deutschem Tausenderpunkt."""
    return f"{x:,.0f}".replace(",", ".")


