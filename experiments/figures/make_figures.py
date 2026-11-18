"""Draw the paper's data figures from the result files.

Writes four vector PDFs: the decomposition of Experiment 1, the bound check of
the appendix, the survey grid of Experiment 2, and the CodeNet panels of
Experiment 3. Input files are read from ``results/`` by default; pass
``--results-dir outputs`` to plot a fresh run instead.

Run from the repository root:
    uv run python experiments/figures/make_figures.py
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from mode_collapse.paths import OUTPUT_DIR, RESULTS_DIR

# ---------------------------------------------------------------- config

DIFFUSE_TERMS = Path("synthetic") / "diffuse_terms.csv"
ALIGNED_TERMS = Path("synthetic") / "aligned_terms.csv"
BOUND_SUMMARY = Path("synthetic") / "bound_per_model.csv"
SURVEY_RATIOS = Path("surveys") / "survey_collision_ratio.csv"
CODENET_RATIOS = Path("codenet") / "codenet_collision_ratio.csv"

MODELS = {
    "gemma-2-2b-it": "Gemma-2-2B-it",
    "qwen3.5-2b": "Qwen3.5-2B",
    "qwen3.5-4b": "Qwen3.5-4B",
    "gemma-3-4b-it": "Gemma-3-4B-it",
}
SURVEYS = ["gss", "wvs", "anes"]
SURVEY_CHECKPOINTS = ["base", "early_stopped"]
CODENET_CHECKPOINTS = ["base", "fully_trained"]
EXPECTED_SURVEY_ARMS = 236
EXPECTED_CODENET_ARMS = 148
CODENET_GRID_SIZES = 18

RATIO_COLUMNS = [
    "collision_ratio_q05",
    "collision_ratio_q25",
    "collision_ratio_median",
    "collision_ratio_q75",
    "collision_ratio_q95",
]

ARROW_PROPS = {"arrowstyle": "-|>", "color": "0.2", "lw": 1.2, "mutation_scale": 12}
MORE_DIVERSE_LABEL = "LLM outputs\nmore diverse"
LESS_DIVERSE_LABEL = "LLM outputs\nless diverse"
SUMMARY_COLOR = "#1f77b4"


@dataclass(frozen=True)
class TermSeries:
    """One curve of the decomposition figure."""

    column: str
    color: str
    label: str


TERM_SERIES = [
    TermSeries(
        "var_term",
        "C0",
        r"variance $\mathrm{Tr}\,\mathrm{Cov}(q_{\hat\theta,h})/C_h(p)$",
    ),
    TermSeries("bias2_term", "C1", r"squared bias $\|b_h\|_2^2/C_h(p)$"),
    TermSeries("cross_term", "C2", r"cross term $2p_h^\top b_h/C_h(p)$"),
]


@dataclass(frozen=True)
class PriorPanel:
    """One panel of the decomposition figure."""

    panel_id: str
    letter: str
    terms_file: Path
    stage: str


PRIOR_PANELS = [
    PriorPanel("diffuse", "(a)", DIFFUSE_TERMS, "Pre-training"),
    PriorPanel("aligned", "(b)", ALIGNED_TERMS, "Fine-tuning"),
]


@dataclass(frozen=True)
class SurveyCell:
    """One panel of the survey grid."""

    row: int
    column: int
    survey: str
    model: str


def configure_matplotlib() -> None:
    """Embed TrueType fonts so text stays selectable in the PDFs."""
    mpl.rcParams["pdf.fonttype"] = 42


def validate_ratio_summary(arm_ratio_df: pd.DataFrame, name: str) -> None:
    """Check that the quantile summaries are finite, positive, and ordered.

    Args:
        arm_ratio_df: One row per arm with collision-ratio quantiles.
        name: Dataset name used in the error message.

    Raises:
        ValueError: If any quantile is nonfinite, nonpositive, or unordered.
    """
    values = arm_ratio_df[RATIO_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError(f"nonpositive or nonfinite ratio summary: {name}")
    if not (np.diff(values, axis=1) >= 0).all():
        raise ValueError(f"unordered ratio quantiles: {name}")


def add_diversity_arrows(
    fig: Figure,
    rect: tuple[float, float, float, float],
    fontsize: float = 7.5,
) -> None:
    """Draw an upward "less diverse" arrow and a downward "more diverse" arrow.

    Args:
        fig: Figure to annotate.
        rect: Axes rectangle (left, bottom, width, height) in figure units.
        fontsize: Label size in points.
    """
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.annotate("", xy=(0.16, 0.94), xytext=(0.16, 0.56), arrowprops=ARROW_PROPS)
    ax.annotate("", xy=(0.16, 0.06), xytext=(0.16, 0.44), arrowprops=ARROW_PROPS)
    text_style = {
        "ha": "left",
        "fontsize": fontsize,
        "color": "0.2",
        "linespacing": 1.2,
        "clip_on": False,
    }
    ax.text(0.36, 0.94, LESS_DIVERSE_LABEL, va="top", **text_style)
    ax.text(0.36, 0.06, MORE_DIVERSE_LABEL, va="bottom", **text_style)


def plot_decomposition(results_dir: Path, out_dir: Path) -> None:
    """Plot Experiment 1: the decomposition terms against sample size.

    Args:
        results_dir: Directory holding the result files.
        out_dir: Directory for the PDF.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    panel_axes = dict(
        zip([panel.panel_id for panel in PRIOR_PANELS], axes, strict=True)
    )
    for panel in PRIOR_PANELS:
        ax = panel_axes[panel.panel_id]
        size_term_df = pd.read_csv(results_dir / panel.terms_file).sort_values("N")
        ax.axhline(0, color="0.6", lw=0.8)
        for series in TERM_SERIES:
            ax.plot(
                size_term_df["N"],
                size_term_df[series.column],
                "-",
                color=series.color,
                label=series.label,
            )
        ax.plot(
            size_term_df["N"],
            size_term_df["measured"],
            "-",
            color="0.0",
            label=r"$\mathbb{E}[R]-1$",
        )
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlim(size_term_df["N"].min(), size_term_df["N"].max())
        ax.set_xlabel(f"{panel.stage} examples $N$ (log scale)")
        ax.set_title(panel.letter, loc="left", fontsize=11)
    left_ax = panel_axes["diffuse"]
    left_ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    left_ax.set_ylabel(r"$\mathbb{E}[R]-1$")

    handles, labels = left_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout(rect=[0.0, 0.03, 0.86, 1.0])
    fig.canvas.draw()
    left_box = left_ax.get_position()
    right_box = panel_axes["aligned"].get_position()
    arrow_height = 1.05 * left_box.height
    arrow_bottom = 0.5 * (left_box.y0 + left_box.y1) - 0.5 * arrow_height
    add_diversity_arrows(fig, (right_box.x1 + 0.014, arrow_bottom, 0.13, arrow_height))
    fig.savefig(out_dir / "decomposition.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_bound_check(results_dir: Path, out_dir: Path) -> None:
    """Plot the appendix check of the relative collision bound.

    Args:
        results_dir: Directory holding the result files.
        out_dir: Directory for the PDF.
    """
    model_bound_df = pd.read_csv(results_dir / BOUND_SUMMARY).sort_values("N")
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 6.1))

    ax = axes[0]
    lowest = min(
        model_bound_df["median_measured"].min(), model_bound_df["median_bound_l2"].min()
    )
    limits = [lowest * 0.8, model_bound_df["median_bound_l2"].max() * 1.2]
    ax.plot(limits, limits, "-", color="0.35", lw=1.0, label="$y=x$ (the bound)")
    ax.scatter(
        model_bound_df["median_bound_l2"],
        model_bound_df["median_measured"],
        s=18,
        color="C0",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Median provable bound (log scale)")
    ax.set_ylabel(r"Median measured $|C(q_h)/C(p_h)-1|$")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(a)", loc="left", fontsize=11)

    ax = axes[1]
    sizes = model_bound_df["N"]
    ax.fill_between(
        sizes,
        model_bound_df["min_ratio"],
        model_bound_df["max_ratio"],
        color="C0",
        alpha=0.15,
        label="range",
    )
    ax.fill_between(
        sizes,
        model_bound_df["q25_ratio"],
        model_bound_df["q75_ratio"],
        color="C0",
        alpha=0.35,
        label="IQR",
    )
    ax.plot(sizes, model_bound_df["median_ratio"], "-", color="C0", label="median")
    ax.axhline(1.0, color="0.35", lw=1.0, label="the bound")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Pre-training examples $N$ (log scale)")
    ax.set_ylabel("Measured $\\div$ provable bound")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title("(b)", loc="left", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_dir / "bound_check.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_surveys(results_dir: Path, out_dir: Path) -> None:
    """Plot Experiment 2: collision ratio by survey and model.

    Args:
        results_dir: Directory holding the result files.
        out_dir: Directory for the PDF.

    Raises:
        ValueError: If the summary does not hold every plotted arm.
    """
    arm_ratio_df = pd.read_csv(results_dir / SURVEY_RATIOS)
    arm_ratio_plot_df = arm_ratio_df[
        arm_ratio_df["checkpoint"].isin(SURVEY_CHECKPOINTS)
    ]
    validate_ratio_summary(arm_ratio_plot_df, "surveys")
    if len(arm_ratio_plot_df) != EXPECTED_SURVEY_ARMS:
        raise ValueError(
            f"expected {EXPECTED_SURVEY_ARMS} survey arms, found {len(arm_ratio_plot_df)}"
        )

    cells = [
        SurveyCell(row, column, survey, model)
        for row, survey in enumerate(SURVEYS)
        for column, model in enumerate(MODELS)
    ]
    fig, axes = plt.subplots(len(SURVEYS), len(MODELS), figsize=(9.0, 3.938))
    for cell in cells:
        ax = axes[cell.row, cell.column]
        is_cell = (arm_ratio_plot_df["dataset"] == cell.survey) & (
            arm_ratio_plot_df["model"] == cell.model
        )
        arm_ratio_cell_df = arm_ratio_plot_df[is_cell]
        low = min(1.0, float(arm_ratio_cell_df["collision_ratio_q05"].min()))
        high = max(1.0, float(arm_ratio_cell_df["collision_ratio_q95"].max()))
        pad = 0.05 * (high - low)
        y_limits = (max(0.0, low - pad), high + pad)
        is_tuned = arm_ratio_cell_df["checkpoint"] == "early_stopped"
        arm_ratio_tuned_df = arm_ratio_cell_df[is_tuned].sort_values("n")
        base = arm_ratio_cell_df[arm_ratio_cell_df["checkpoint"] == "base"].iloc[0]
        sizes = arm_ratio_tuned_df["n"]
        ax.fill_between(
            sizes,
            arm_ratio_tuned_df["collision_ratio_q05"],
            arm_ratio_tuned_df["collision_ratio_q95"],
            color="C0",
            alpha=0.15,
            lw=0,
        )
        ax.fill_between(
            sizes,
            arm_ratio_tuned_df["collision_ratio_q25"],
            arm_ratio_tuned_df["collision_ratio_q75"],
            color="C0",
            alpha=0.35,
            lw=0,
        )
        ax.plot(
            sizes,
            arm_ratio_tuned_df["collision_ratio_median"],
            "-o",
            color="C0",
            ms=2.6,
            lw=1.3,
        )
        outer_band = [base["collision_ratio_q05"], base["collision_ratio_q95"]]
        inner_band = [base["collision_ratio_q25"], base["collision_ratio_q75"]]
        ax.plot([0, 0], outer_band, color="C0", alpha=0.3, lw=2.2)
        ax.plot([0, 0], inner_band, color="C0", alpha=0.6, lw=2.2)
        ax.plot(0, base["collision_ratio_median"], "o", color="C0", ms=4.5)
        ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
        ax.set_xscale("symlog", linthresh=float(sizes.min()))
        ax.set_xlim(0, sizes.max() * 1.6)
        decades = [
            10**exponent for exponent in range(1, 8) if 10**exponent <= sizes.max()
        ]
        ax.set_xticks([0] + decades)
        decade_labels = [f"$10^{{{int(np.log10(decade))}}}$" for decade in decades]
        ax.set_xticklabels(["0"] + decade_labels)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_ylim(*y_limits)
        ax.tick_params(labelsize=7)
        if cell.row == 0:
            ax.set_title(MODELS[cell.model], fontsize=9)
        if cell.row != len(SURVEYS) - 1:
            ax.set_xticklabels([])
        if cell.column == 0:
            ax.set_ylabel(cell.survey.upper(), fontsize=8)

    handles = [
        plt.Line2D([], [], color="C0", marker="o", ms=3.5, lw=1.3),
        plt.Rectangle((0, 0), 1, 1, color="C0", alpha=0.35),
        plt.Rectangle((0, 0), 1, 1, color="C0", alpha=0.15),
        plt.Line2D([], [], color="0.5", ls=":", lw=1.0),
    ]
    labels = [
        "median across (question, group) pairs",
        "25-75% of pairs",
        "5-95% of pairs",
        "calibrated ($=1$)",
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.022),
    )
    fig.supylabel(r"$R_{\mathrm{U}}=C(q)/\hat C_{\mathrm{U}}$", fontsize=9)
    fig.supxlabel("Fine-tuning examples (symlog scale)", fontsize=9, y=0.058)
    fig.tight_layout(rect=[0.015, 0.05, 0.80, 1])
    fig.canvas.draw()
    for row in range(len(SURVEYS)):
        row_box = axes[row, -1].get_position()
        pad = 0.06 * row_box.height
        arrow_rect = (
            row_box.x1 + 0.014,
            row_box.y0 + pad,
            0.14,
            row_box.height - 2 * pad,
        )
        add_diversity_arrows(fig, arrow_rect)
    fig.savefig(out_dir / "surveys.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_codenet(results_dir: Path, out_dir: Path) -> None:
    """Plot Experiment 3: collision ratio under Zhang-Shasha similarity.

    Args:
        results_dir: Directory holding the result files.
        out_dir: Directory for the PDF.

    Raises:
        ValueError: If the summary does not hold the full model grid.
    """
    arm_ratio_df = pd.read_csv(results_dir / CODENET_RATIOS)
    validate_ratio_summary(arm_ratio_df, "CodeNet")
    if len(arm_ratio_df) != EXPECTED_CODENET_ARMS:
        raise ValueError(
            f"expected {EXPECTED_CODENET_ARMS} CodeNet arms, found {len(arm_ratio_df)}"
        )
    arms_per_model = arm_ratio_df.groupby("model").size()
    arms_expected = 1 + 2 * CODENET_GRID_SIZES
    if (
        set(arms_per_model.index) != set(MODELS)
        or not arms_per_model.eq(arms_expected).all()
    ):
        raise ValueError(f"incomplete CodeNet model grid: {arms_per_model.to_dict()}")
    training_grid = sorted(arm_ratio_df.loc[arm_ratio_df["n"] > 0, "n"].unique())
    if len(training_grid) != CODENET_GRID_SIZES:
        raise ValueError(
            f"expected {CODENET_GRID_SIZES} fine-tuning sizes, found {training_grid}"
        )

    break_marker = {
        "marker": [(-1, -0.5), (1, 0.5)],
        "markersize": 8,
        "linestyle": "none",
        "color": "black",
        "markeredgecolor": "black",
        "markeredgewidth": 1,
        "clip_on": False,
    }
    font_settings = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    band_columns = [
        "collision_ratio_q05",
        "collision_ratio_q25",
        "collision_ratio_q75",
        "collision_ratio_q95",
    ]
    with mpl.rc_context(font_settings):
        figure = Figure(figsize=(8.46, 2.7))
        FigureCanvasPdf(figure)
        outer = figure.add_gridspec(
            1, 4, wspace=0.30, top=0.885, bottom=0.225, left=0.075, right=0.865
        )
        sweep_axes = []
        for index, model in enumerate(MODELS):
            inner = outer[index].subgridspec(1, 2, width_ratios=[1, 6], wspace=0.05)
            base_axis = figure.add_subplot(inner[0])
            sweep_axis = figure.add_subplot(inner[1], sharey=base_axis)
            arm_ratio_model_df = arm_ratio_df[arm_ratio_df["model"] == model]
            is_plotted = arm_ratio_model_df["checkpoint"].isin(CODENET_CHECKPOINTS)
            band_values = (
                arm_ratio_model_df.loc[is_plotted, band_columns].to_numpy().ravel()
            )
            y_low = min(1.0, float(np.min(band_values)))
            y_high = max(1.0, float(np.max(band_values)))
            y_padding = 0.05 * (y_high - y_low) if y_high > y_low else 0.1
            y_limits = (max(0.0, y_low - y_padding), y_high + y_padding)

            arm_ratio_base_df = arm_ratio_model_df[
                arm_ratio_model_df["checkpoint"] == "base"
            ]
            if len(arm_ratio_base_df) != 1:
                raise ValueError(f"expected one base arm for {model}")
            base = arm_ratio_base_df.iloc[0]
            outer_band = [
                float(base["collision_ratio_q05"]),
                float(base["collision_ratio_q95"]),
            ]
            inner_band = [
                float(base["collision_ratio_q25"]),
                float(base["collision_ratio_q75"]),
            ]
            base_axis.plot(
                [0, 0],
                outer_band,
                color=SUMMARY_COLOR,
                linewidth=2.4,
                alpha=0.15,
                solid_capstyle="butt",
            )
            base_axis.plot(
                [0, 0],
                inner_band,
                color=SUMMARY_COLOR,
                linewidth=2.4,
                alpha=0.40,
                solid_capstyle="butt",
            )
            base_axis.plot(
                [0],
                [float(base["collision_ratio_median"])],
                marker="o",
                markersize=7,
                color=SUMMARY_COLOR,
                linestyle="none",
            )

            is_final = arm_ratio_model_df["checkpoint"] == "fully_trained"
            arm_ratio_final_df = arm_ratio_model_df[is_final].sort_values("n")
            if len(arm_ratio_final_df) != len(training_grid):
                raise ValueError(f"incomplete final-adapter trajectory for {model}")
            x_values = arm_ratio_final_df["n"].to_numpy(float)
            sweep_axis.fill_between(
                x_values,
                arm_ratio_final_df["collision_ratio_q05"].to_numpy(float),
                arm_ratio_final_df["collision_ratio_q95"].to_numpy(float),
                color=SUMMARY_COLOR,
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
            sweep_axis.fill_between(
                x_values,
                arm_ratio_final_df["collision_ratio_q25"].to_numpy(float),
                arm_ratio_final_df["collision_ratio_q75"].to_numpy(float),
                color=SUMMARY_COLOR,
                alpha=0.32,
                linewidth=0,
                zorder=2,
            )
            sweep_axis.plot(
                x_values,
                arm_ratio_final_df["collision_ratio_median"].to_numpy(float),
                color=SUMMARY_COLOR,
                marker="o",
                markersize=3.8,
                linewidth=1.6,
                zorder=3,
            )

            for axis in (base_axis, sweep_axis):
                axis.axhline(1.0, color="gray", linestyle=":", linewidth=1.3)
                axis.tick_params(labelsize=8)
                axis.set_ylim(*y_limits)
            base_axis.set_xlim(-0.8, 0.8)
            base_axis.set_xticks([0])
            base_axis.set_xticklabels(["0"])
            base_axis.spines["right"].set_visible(False)
            sweep_axis.spines["left"].set_visible(False)
            sweep_axis.tick_params(axis="y", which="both", left=False, labelleft=False)
            base_axis.plot(
                [1, 1], [0, 1], transform=base_axis.transAxes, **break_marker
            )
            sweep_axis.plot(
                [0, 0], [0, 1], transform=sweep_axis.transAxes, **break_marker
            )
            sweep_axis.set_xscale("log")
            sweep_axis.set_xlim(min(training_grid) / 1.25, max(training_grid) * 1.25)
            sweep_axis.set_title(MODELS[model], fontsize=10)
            sweep_axes.append(sweep_axis)

        figure.supxlabel(
            "Fine-tuning examples (log scale)", fontsize=9, x=0.47, y=0.095
        )
        figure.supylabel(
            r"$R=C_{\mathrm{ZS}}(q)/C_{\mathrm{ZS}}(p)$", fontsize=10.5, x=0.014
        )
        legend_handles = [
            Line2D(
                [],
                [],
                color=SUMMARY_COLOR,
                marker="o",
                markersize=5,
                label="median across problems (final adapter)",
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=SUMMARY_COLOR,
                alpha=0.32,
                edgecolor="none",
                label="25\u201375% of problems",
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=SUMMARY_COLOR,
                alpha=0.12,
                edgecolor="none",
                label="5\u201395% of problems",
            ),
            Line2D(
                [],
                [],
                color=SUMMARY_COLOR,
                marker="o",
                markersize=7,
                linestyle="none",
                label="base zero-shot (n=0)",
            ),
            Line2D(
                [],
                [],
                color="gray",
                linestyle=":",
                linewidth=1.3,
                label="calibrated (=1)",
            ),
        ]
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=5,
            fontsize=8,
            columnspacing=1.91015625,
            bbox_to_anchor=(0.5, 0.0),
            frameon=False,
        )

        panel_box = sweep_axes[-1].get_position()
        arrow_rect = (
            panel_box.x1 + 0.012,
            panel_box.y0 + 0.06 * panel_box.height,
            0.115,
            panel_box.height * 0.88,
        )
        add_diversity_arrows(figure, arrow_rect)
        figure.canvas.draw()
        figure.savefig(out_dir / "codenet.pdf", format="pdf", bbox_inches="tight")


def main() -> None:
    """Parse arguments and write the four figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "figures")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    plot_decomposition(args.results_dir, args.out_dir)
    plot_bound_check(args.results_dir, args.out_dir)
    plot_surveys(args.results_dir, args.out_dir)
    plot_codenet(args.results_dir, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
