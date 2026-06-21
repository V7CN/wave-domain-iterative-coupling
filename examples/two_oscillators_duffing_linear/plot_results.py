#!/usr/bin/env python3
"""Generate paper figures from the C++ benchmark output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


_TRAJECTORY_PATTERN = re.compile(r"trajectory_K(\d+)\.csv$")
_OUTPUT_DIRECTORY: Path | None = None


def _load_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    data = np.atleast_1d(data)
    if data.size == 0:
        raise ValueError(f"Data file is empty: {path}")
    return data


def _require_columns(data: np.ndarray, path: Path, columns: Iterable[str]) -> None:
    names = set(data.dtype.names or ())
    missing = [name for name in columns if name not in names]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def _discover_kn(data_dir: Path) -> list[int]:
    values = []
    for path in data_dir.glob("trajectory_K*.csv"):
        match = _TRAJECTORY_PATTERN.match(path.name)
        if match:
            values.append(int(match.group(1)))
    values = sorted(set(values))
    if not values:
        raise FileNotFoundError(f"No trajectory_K*.csv files found in {data_dir}")
    return values


def _check_time_grid(reference_t: np.ndarray, candidate_t: np.ndarray, path: Path) -> None:
    if candidate_t.shape != reference_t.shape or not np.array_equal(candidate_t, reference_t):
        raise ValueError(f"Time grid in {path} differs from reference_bigK.csv")


def load_cpp_results(data_dir: Path, kn_list: list[int]):
    reference_path = data_dir / "reference_bigK.csv"
    mono = _load_csv(reference_path)
    _require_columns(mono, reference_path, ("t", "q1", "q2"))

    t = np.asarray(mono["t"], dtype=float)
    q1_mono = np.asarray(mono["q1"], dtype=float)
    q2_mono = np.asarray(mono["q2"], dtype=float)
    outs = {}
    diagnostics_available = True
    for kn in kn_list:
        trajectory_path = data_dir / f"trajectory_K{kn}.csv"
        diagnostics_path = data_dir / f"diagnostics_K{kn}.csv"
        trajectory = _load_csv(trajectory_path)

        _require_columns(trajectory, trajectory_path, ("t", "q1", "q2"))
        _check_time_grid(t, np.asarray(trajectory["t"], dtype=float), trajectory_path)

        outs[kn] = {
            "q1": np.asarray(trajectory["q1"], dtype=float),
            "q2": np.asarray(trajectory["q2"], dtype=float),
        }
        if diagnostics_path.is_file():
            diagnostics = _load_csv(diagnostics_path)
            _require_columns(
                diagnostics,
                diagnostics_path,
                (
                    "fne_min_delta_A",
                    "fne_min_delta_B",
                    "fne_viol_A",
                    "fne_viol_B",
                    "rA",
                    "rB",
                    "augmented_residual",
                ),
            )
            if diagnostics.size != max(t.size - 1, 0):
                raise ValueError(
                    f"{diagnostics_path} has {diagnostics.size} rows; "
                    f"expected {max(t.size - 1, 0)}"
                )
            outs[kn]["audit"] = {
                "fne_min_delta_A": np.asarray(diagnostics["fne_min_delta_A"], dtype=float),
                "fne_min_delta_B": np.asarray(diagnostics["fne_min_delta_B"], dtype=float),
                "fne_viol_A": np.asarray(diagnostics["fne_viol_A"], dtype=int),
                "fne_viol_B": np.asarray(diagnostics["fne_viol_B"], dtype=int),
                "rA": np.asarray(diagnostics["rA"], dtype=float),
                "rB": np.asarray(diagnostics["rB"], dtype=float),
                "enh_lhs": np.asarray(diagnostics["augmented_residual"], dtype=float),
            }
        else:
            diagnostics_available = False

    return t, q1_mono, q2_mono, outs, diagnostics_available


def _save_or_show(fig, filename: str, dpi: int = 200) -> None:
    if _OUTPUT_DIRECTORY is None:
        raise RuntimeError("Plot output directory has not been configured")
    gs = shutil.which("gs")
    if gs is None:
        plt.close(fig)
        raise RuntimeError("Ghostscript (gs) is required for outlined EPS output")

    base, _ = os.path.splitext(filename)
    root = _OUTPUT_DIRECTORY / base
    png_path = root.with_suffix(".png")
    eps_path = root.with_suffix(".eps")
    outlined_eps = _OUTPUT_DIRECTORY / f"{base}_outlined.eps"

    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    print(f"[plot] Saved PNG to {png_path}")
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"[plot] Saved EPS (intermediate) to {eps_path}")
    try:
        subprocess.run(
            [gs, "-o", str(outlined_eps), "-sDEVICE=eps2write", "-dNoOutputFonts", str(eps_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Ghostscript outlining failed: {error.stderr.strip()}") from error
    finally:
        plt.close(fig)
    print(f"[plot] Saved outlined EPS to {outlined_eps}")



def _kn_color_map(Kn_list: list[int]) -> dict[int, object]:
    """
    Stable color mapping for lines indexed by Kn.

    Paper requirement: the same Kn should have the same color across all figures.
    """
    Kn_sorted = sorted({int(Kn) for Kn in Kn_list})
    cmap = plt.get_cmap("tab10")
    return {Kn: cmap(i % 10) for i, Kn in enumerate(Kn_sorted)}


def plot_trajectory_comparison(t, q1_mono, q2_mono, outs, Kn_list):
    colors = _kn_color_map(Kn_list)

    # Combined (stacked) trajectory figure:
    # - shared x-axis (ticks/label only at bottom),
    # - one shared legend (outside axes) with a non-transparent box,
    # - per-panel tag in the top-right corner ("q1", "q2").
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.8, 3.2))
    ax1, ax2 = axes[0], axes[1]

    # q1
    ax1.plot(t, q1_mono, linewidth=2.5, color="k", linestyle="--", label=r"$\mathrm{mono}$")
    for Kn in Kn_list:
        ax1.plot(
            t,
            outs[Kn]["q1"],
            linewidth=1.6,
            color=colors.get(int(Kn), None),
            label=rf"$K_n={int(Kn)}$",
        )
    ax1.set_ylabel(r"$q_1$")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)
    ax1.text(
        0.98,
        0.92,
        r"$q_1$",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
    )

    # q2
    ax2.plot(t, q2_mono, linewidth=2.5, color="k", linestyle="--", label=r"$\mathrm{mono}$")
    for Kn in Kn_list:
        ax2.plot(
            t,
            outs[Kn]["q2"],
            linewidth=1.6,
            color=colors.get(int(Kn), None),
            label=rf"$K_n={int(Kn)}$",
        )
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$q_2$")
    ax2.grid(True, alpha=0.3)
    ax2.text(
        0.98,
        0.92,
        r"$q_2$",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
    )

    # One shared legend: collect handles once, then place above the axes.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.7",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save_or_show(fig, "traj_q.png")


def plot_position_error(t, q1_mono, q2_mono, outs, Kn_list):
    colors = _kn_color_map(Kn_list)
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.2))
    for Kn in Kn_list:
        delta_q1 = outs[Kn]["q1"] - q1_mono
        delta_q2 = outs[Kn]["q2"] - q2_mono
        e = np.sqrt(delta_q1**2 + delta_q2**2)
        # log-y plot; add a small floor to avoid log(0) when Kn is large
        ax.semilogy(t, np.maximum(e, 1e-18), color=colors.get(int(Kn), None), label=rf"$K_n={int(Kn)}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\|[\Delta q_1(t),\Delta q_2(t)]\|_2$")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=10, frameon=False)
    fig.tight_layout()
    _save_or_show(fig, "position_error.png")


def plot_convergence_vs_kn(q1_mono, q2_mono, outs, Kn_list):
    """Reproduce the paper's state-error summary from the plotted q1/q2 components."""
    Kn_arr = np.asarray(Kn_list, dtype=float)
    max_err = []
    rms_err = []
    final_err = []
    for Kn in Kn_list:
        delta_q1 = outs[Kn]["q1"] - q1_mono
        delta_q2 = outs[Kn]["q2"] - q2_mono
        e = np.sqrt(delta_q1**2 + delta_q2**2)
        max_err.append(float(np.max(e)))
        rms_err.append(float(np.sqrt(np.mean(e**2))))
        final_err.append(float(e[-1]))

    max_err = np.asarray(max_err, dtype=float)
    rms_err = np.asarray(rms_err, dtype=float)
    final_err = np.asarray(final_err, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.2))
    ax.semilogy(Kn_arr, np.maximum(max_err, 1e-18), marker="o", label=r"$\max_t \|\Delta q(t)\|_2$")
    ax.semilogy(Kn_arr, np.maximum(rms_err, 1e-18), marker="o", label=r"$\mathrm{rms}_t \|\Delta q(t)\|_2$")
    ax.semilogy(Kn_arr, np.maximum(final_err, 1e-18), marker="o", label=r"$\|\Delta q(T)\|_2$")
    ax.set_xlabel(r"$K_n$")
    ax.set_ylabel(r"error vs. monolithic")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    _save_or_show(fig, "convergence_vs_Kn.png")


def plot_fne_passivity_summary(outs, Kn_list):
    """Compact, Kn-indexed summaries to support the three experimental claims."""
    Kn_arr = np.asarray(Kn_list, dtype=float)

    # ---- FNE summaries ----
    min_margin_A = []
    min_margin_B = []
    max_viol_A = []
    max_viol_B = []
    for Kn in Kn_list:
        aud = outs[Kn]["audit"]
        min_margin_A.append(float(np.min(np.asarray(aud["fne_min_delta_A"], dtype=float))))
        min_margin_B.append(float(np.min(np.asarray(aud["fne_min_delta_B"], dtype=float))))
        max_viol_A.append(int(np.max(np.asarray(aud["fne_viol_A"], dtype=int))))
        max_viol_B.append(int(np.max(np.asarray(aud["fne_viol_B"], dtype=int))))

    min_margin_A = np.asarray(min_margin_A, dtype=float)
    min_margin_B = np.asarray(min_margin_B, dtype=float)
    max_viol_A = np.asarray(max_viol_A, dtype=float)
    max_viol_B = np.asarray(max_viol_B, dtype=float)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.4, 5.0))
    ax = axes[0]
    ax.plot(Kn_arr, min_margin_A, marker="o", label=r"$\min_n \min\,\Delta_A^n$")
    ax.plot(Kn_arr, min_margin_B, marker="o", label=r"$\min_n \min\,\Delta_B^n$")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel(r"margin")
    ax.set_title("FNE certificate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, frameon=False)
    ax = axes[1]
    ax.plot(Kn_arr, max_viol_A, marker="o", label=r"$\max_n\,\#\{\Delta_A^n<-\mathrm{tol}\}$")
    ax.plot(Kn_arr, max_viol_B, marker="o", label=r"$\max_n\,\#\{\Delta_B^n<-\mathrm{tol}\}$")
    ax.set_xlabel(r"$K_n$")
    ax.set_ylabel(r"count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    _save_or_show(fig, "summary_FNE_vs_Kn.png")

    # ---- Passivity summaries ----
    max_rA_pos = []
    max_rB_pos = []
    sum_rA_pos = []
    sum_rB_pos = []
    max_enh_pos = []
    sum_enh_pos = []
    for Kn in Kn_list:
        aud = outs[Kn]["audit"]
        rA = np.asarray(aud["rA"], dtype=float)
        rB = np.asarray(aud["rB"], dtype=float)
        rA_pos = np.maximum(rA, 0.0)
        rB_pos = np.maximum(rB, 0.0)
        max_rA_pos.append(float(np.max(rA_pos)) if rA_pos.size else 0.0)
        max_rB_pos.append(float(np.max(rB_pos)) if rB_pos.size else 0.0)
        sum_rA_pos.append(float(np.sum(rA_pos)) if rA_pos.size else 0.0)
        sum_rB_pos.append(float(np.sum(rB_pos)) if rB_pos.size else 0.0)

        enh = np.asarray(aud["enh_lhs"], dtype=float)
        enh_pos = np.maximum(enh, 0.0)
        max_enh_pos.append(float(np.max(enh_pos)) if enh_pos.size else 0.0)
        sum_enh_pos.append(float(np.sum(enh_pos)) if enh_pos.size else 0.0)

    max_rA_pos = np.asarray(max_rA_pos, dtype=float)
    max_rB_pos = np.asarray(max_rB_pos, dtype=float)
    sum_rA_pos = np.asarray(sum_rA_pos, dtype=float)
    sum_rB_pos = np.asarray(sum_rB_pos, dtype=float)
    max_enh_pos = np.asarray(max_enh_pos, dtype=float)
    sum_enh_pos = np.asarray(sum_enh_pos, dtype=float)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.4, 5.0))
    ax = axes[0]
    ax.semilogy(Kn_arr, np.maximum(max_rA_pos, 1e-18), marker="o", label=r"$\max_n\,(r_A^n)_+$")
    ax.semilogy(Kn_arr, np.maximum(max_rB_pos, 1e-18), marker="o", label=r"$\max_n\,(r_B^n)_+$")
    ax.semilogy(Kn_arr, np.maximum(max_enh_pos, 1e-18), marker="o", label=r"$\max_n\,(\mathrm{AugRes}^n)_+$")
    ax.set_ylabel("max positive residual")
    ax.set_title("Discrete Passivity Certificate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, frameon=False)

    ax = axes[1]
    ax.semilogy(Kn_arr, np.maximum(sum_rA_pos, 1e-18), marker="o", label=r"$\sum_n (r_A^n)_+$")
    ax.semilogy(Kn_arr, np.maximum(sum_rB_pos, 1e-18), marker="o", label=r"$\sum_n (r_B^n)_+$")
    ax.semilogy(Kn_arr, np.maximum(sum_enh_pos, 1e-18), marker="o", label=r"$\sum_n (\mathrm{AugRes}^n)_+$")
    ax.set_xlabel(r"$K_n$")
    ax.set_ylabel("cumulative positive residual")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    _save_or_show(fig, "summary_passivity_vs_Kn.png")

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot C++ iterative-coupling benchmark CSV files with the reference paper style."
    )
    parser.add_argument("data_dir", type=Path, help="Directory containing reference_bigK.csv and K-indexed CSV files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "out_figs",
        help="EPS output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--kn",
        type=int,
        nargs="+",
        help="K_n values to plot; defaults to all trajectory_K*.csv files in data_dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    kn_list = args.kn if args.kn is not None else _discover_kn(data_dir)
    kn_list = sorted(set(kn_list))

    t, q1_mono, q2_mono, outs, diagnostics_available = load_cpp_results(data_dir, kn_list)
    global _OUTPUT_DIRECTORY
    _OUTPUT_DIRECTORY = args.output_dir.resolve()
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    plot_trajectory_comparison(t, q1_mono, q2_mono, outs, kn_list)
    plot_position_error(t, q1_mono, q2_mono, outs, kn_list)
    plot_convergence_vs_kn(q1_mono, q2_mono, outs, kn_list)
    if diagnostics_available:
        plot_fne_passivity_summary(outs, kn_list)
    else:
        print("[plot] Diagnostics are absent; skipped FNE/passivity summary figures")


if __name__ == "__main__":
    main()
