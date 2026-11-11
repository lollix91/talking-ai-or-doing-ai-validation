"""Utility grafiche condivise per la Fase 4.

Usa un backend non interattivo (Agg) cosi' gli script possono girare senza
display e salvare direttamente i PNG in outputs/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

import config  # noqa: E402

FIG_DIR = config.OUTPUTS_DIR / "figures"
TAB_DIR = config.OUTPUTS_DIR / "tables"

sns.set_theme(style="whitegrid", context="paper")


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Salva una figura in outputs/figures/ come PNG ad alta risoluzione."""
    _ensure_dirs()
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_table(df, name: str, index: bool = True) -> Path:
    """Salva una tabella in outputs/tables/ come CSV."""
    _ensure_dirs()
    path = TAB_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    return path
