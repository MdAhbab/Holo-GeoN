"""
scripts/figures/style.py
========================
One look for every paper figure, built on the validated reference palette of the
data-viz method: categorical order fixed and never cycled, one axis per panel
(never dual-axis), thin marks, hairline solid grid, text in ink tokens and never
in a series colour, a legend whenever there are two or more series.

Colour follows the ENTITY, never its rank:

    Holo-GeoN v2 ................ slot 1 (blue)
    Holo-GeoN v1 ................ slot 2 (orange)
    ESM-2 + MLP control ......... slot 3 (aqua)
    every other model ........... muted gray, told apart by label and marker shape

At most three hues are ever on one figure: the first three slots are the ones that
validate for scatter and small multiples (all-pairs CVD / normal-vision gates,
checked with the method's validate_palette script against a white surface). The
aqua slot is 2.82:1 on white, so wherever it appears the figure also carries
visible labels and a table twin (``<id>_data.csv``) - the relief the method requires.

Nothing here knows about any result; it only draws what it is given.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- tokens (print surface is white; the validator was run against #ffffff) ---------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# fixed categorical order, never cycled; a 9th series is never generated
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

# sequential ramp (blue steps 100 -> 700 of the reference palette), light = small value
BLUE_STEPS = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
BLUES = LinearSegmentedColormap.from_list("holo_blue", BLUE_STEPS)

# slot 3 is shared by the two controls (matched-capacity MLP, Gaussian-NLL v2); they never share a figure
ENTITY_COLOR = {"v2": SERIES[0], "v1": SERIES[1], "mlp": SERIES[2], "nll": SERIES[2], "other": MUTED}
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

DISPLAY = {
    "holo_geon": "Holo-GeoN v2", "holo_geon_v2": "Holo-GeoN v2", "full": "Holo-GeoN v2",
    "holo_geon_v2_lean": "Holo-GeoN v2-Lean", "v2_lean": "Holo-GeoN v2-Lean",
    "holo_geon_v1": "Holo-GeoN v1", "v1_baseline": "Holo-GeoN v1",
    "esm_mlp": "ESM-2 + MLP (matched capacity)",
    "esm_mlp_matched_capacity": "ESM-2 + MLP (matched capacity)",
    "esm2_zero_shot": "ESM-2 zero-shot",
    "thermompnn": "ThermoMPNN", "spurs": "SPURS", "spurs_local": "SPURS (retrained)", "rasp": "RaSP",
    "prostata": "PROSTATA", "ddgemb": "DDGemb", "proteinmpnn_ddg": "ProteinMPNN-ddG",
}

ARM_LABEL = {
    "v1_baseline": "Holo-GeoN v1 (20.6M params)",
    "v2_lean": "Holo-GeoN v2-Lean (no E_mech, no SSM)",
    "multi_layer_ln": "multi-layer ESM (per-layer LN)",
    "no_esm_logits_prior": "no ESM-2 log-odds prior",
    "no_multi_layer": "no multi-layer ScalarMix",
    "no_contact_burial": "no contact-burial descriptors",
    "standard_nll": "Gaussian NLL (no beta-NLL)",
    "no_cds_dropout": "no CDS dropout",
    "no_mech_stream": "no E_mech stream",
    "no_geo_stream": "no E_geo stream",
    "no_ssm": "no sequence mixer",
    "global_pool_readout": "global-pool readout",
    "str_only": "B  3D frames only",
    "both_streams": "C  contacts + 3D frames",
    "str_topology_only": "D  3D k-NN topology only",
    "true_contact_edges": "E  true C-beta contacts",
}


def entity(name: str) -> str:
    """Which identity a model/arm name belongs to: v2, v1, mlp or other."""
    n = name.lower().replace("-", "_").replace(" ", "_")
    if n in {"holo_geon", "holo_geon_v2", "full"} or n.startswith("holo_geon_v2") or n.startswith("v2_"):
        return "v2"
    if n in {"v1_baseline", "holo_geon_v1"} or n.startswith("holo_geon_v1"):
        return "v1"
    if "standard_nll" in n or "gaussian" in n:
        return "nll"
    if "mlp" in n:
        return "mlp"
    return "other"


def color_of(name: str) -> str:
    return ENTITY_COLOR[entity(name)]


def display(name: str) -> str:
    return DISPLAY.get(name.lower(), name.replace("_", " "))


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "axes.titlelocation": "left",
        "axes.titleweight": "regular", "axes.titlepad": 7,
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 6.8, "legend.handlelength": 1.8,
        "legend.labelcolor": INK2,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
    })


# --- drawing helpers -----------------------------------------------------------------
def panel_label(ax, letter: str) -> None:
    """Bold panel letter just outside the top-left corner of the axes."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-3, 8),
                textcoords="offset points", fontsize=9, fontweight="bold", color=INK,
                ha="right", va="bottom")


def dot(ax, x, y, color, *, hollow: bool = False, size: float = 5.0, marker: str = "o", zorder: int = 4):
    """A marker (>= 5 pt) with a surface-coloured ring; hollow = outline only."""
    if hollow:   # the edge is centred on the outline, so shave the size to match a filled marker's ring
        ax.plot([x], [y], marker=marker, markersize=size - 0.3, markerfacecolor=SURFACE,
                markeredgecolor=color, markeredgewidth=1.3, linestyle="none", zorder=zorder)
    else:
        ax.plot([x], [y], marker=marker, markersize=size, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.0, linestyle="none", zorder=zorder)


def interval(ax, pos, lo, hi, color, *, horizontal: bool = True, lw: float = 1.3, zorder: int = 3):
    """A thin confidence whisker (no caps)."""
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return
    if horizontal:
        ax.plot([lo, hi], [pos, pos], color=color, lw=lw, solid_capstyle="round", zorder=zorder)
    else:
        ax.plot([pos, pos], [lo, hi], color=color, lw=lw, solid_capstyle="round", zorder=zorder)


def fmt_ci(v, lo=None, hi=None, nd: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    s = f"{v:.{nd}f}"
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{lo:.{nd}f}, {hi:.{nd}f}]"
    return s


# --- LaTeX ---------------------------------------------------------------------------
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "<": r"$<$", ">": r"$>$"}
_UNI = {"≥": r"$\geq$", "≤": r"$\leq$", "Δ": r"$\Delta$", "ρ": r"$\rho$", "±": r"$\pm$",
        "−": "-", "×": r"$\times$", "→": r"$\rightarrow$", "–": "--", "Å": r"\AA{}"}


def tex(s) -> str:
    s = "" if s is None else str(s)
    out = "".join(_TEX.get(c, c) for c in s)
    return "".join(_UNI.get(c, c) for c in out)


def latex_table(df: pd.DataFrame, caption: str, label: str, *, col_format: str | None = None,
                note: str | None = None) -> str:
    cols = list(df.columns)
    fmt = col_format or ("l" + "r" * (len(cols) - 1))
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{tex(caption)}}}",
             rf"\label{{{label}}}", rf"\begin{{tabular}}{{{fmt}}}", r"\toprule",
             " & ".join(tex(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(tex(v) for v in row.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\par\smallskip{{\footnotesize {tex(note)}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# --- files, hashes, provenance -------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    """UTF-8 with LF endings on every platform, so artifacts are byte-reproducible."""
    Path(path).write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def watermark(fig) -> None:
    fig.text(0.5, 0.5, "SYNTHETIC FIXTURE\nNOT RESULTS", rotation=28, ha="center", va="center",
             fontsize=26, color=SERIES[7], alpha=0.16, fontweight="bold", zorder=100)


def save_figure(fig, out_dir: Path, stem: str, *, data: pd.DataFrame | None, sources: list,
                demo: bool, caption: str = "") -> dict:
    """Write <stem>.pdf/.png, the table twin and a draft caption; return its manifest record."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_DEMO" if demo else stem
    if demo:
        watermark(fig)
    files: list[str] = []
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04,
                metadata={"CreationDate": None, "Creator": "Holo-StabBench figures"})
    files.append(f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", pad_inches=0.04)
    files.append(f"{name}.png")
    plt.close(fig)
    if data is not None:
        write_text_lf(out_dir / f"{name}_data.csv", data.to_csv(index=False))
        files.append(f"{name}_data.csv")
    if caption:
        write_text_lf(out_dir / f"{name}_caption.txt", caption.strip() + "\n")
        files.append(f"{name}_caption.txt")
    return {"id": stem, "kind": "figure", "demo": demo, "files": files,
            "inputs": {str(p): sha256_file(Path(p)) for p in sources if Path(p).is_file()}}


def save_table(df: pd.DataFrame, out_dir: Path, stem: str, *, caption: str, label: str,
               tex_df: pd.DataFrame | None, sources: list, demo: bool, note: str | None = None,
               col_format: str | None = None) -> dict:
    """Write <stem>.csv (full numeric twin) and <stem>.tex (booktabs); return its record."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_DEMO" if demo else stem
    write_text_lf(out_dir / f"{name}.csv", df.to_csv(index=False))
    cap = caption + (" [SYNTHETIC FIXTURE, NOT RESULTS]" if demo else "")
    write_text_lf(out_dir / f"{name}.tex",
                  latex_table(tex_df if tex_df is not None else df, cap, label,
                              col_format=col_format, note=note))
    return {"id": stem, "kind": "table", "demo": demo, "files": [f"{name}.csv", f"{name}.tex"],
            "inputs": {str(p): sha256_file(Path(p)) for p in sources if Path(p).is_file()}}


def figure_provenance() -> dict:
    """Light provenance (no torch import): code commit and plotting stack."""
    try:
        from src.provenance import _git_commit
        git = _git_commit()
    except Exception:
        git = None
    return {"git": git, "python": platform.python_version(), "matplotlib": matplotlib.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}


__all__ = ["apply_style", "panel_label", "dot", "interval", "fmt_ci", "latex_table", "tex",
           "save_figure", "save_table", "figure_provenance", "color_of", "display", "entity",
           "SERIES", "BLUES", "INK", "INK2", "MUTED", "GRID", "AXIS", "SURFACE", "ARM_LABEL",
           "ENTITY_COLOR", "MARKERS", "sha256_file", "write_text_lf"]
