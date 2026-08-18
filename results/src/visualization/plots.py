"""Publication-ready plotting module for Score-Based Generative Model Ablation Studies."""

from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set publication style
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_fid_vs_steps(df: pd.DataFrame, save_dir: Path) -> Optional[Path]:
    """Plot 1: FID vs Training Steps across models/SDEs."""
    if df.empty or "steps" not in df.columns or "fid" not in df.columns:
        return None
    plt.figure(figsize=(7, 4.5))
    sns.lineplot(data=df, x="steps", y="fid", hue="model", style="sde", markers=True, dashes=False, linewidth=2)
    plt.title("FID vs Training Steps (Lower is Better)")
    plt.xlabel("Training Steps")
    plt.ylabel("Fréchet Inception Distance (FID)")
    plt.tight_layout()
    out = save_dir / "01_fid_vs_steps.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_is_vs_steps(df: pd.DataFrame, save_dir: Path) -> Optional[Path]:
    """Plot 2: Inception Score vs Training Steps."""
    if df.empty or "steps" not in df.columns or "inception_score" not in df.columns:
        return None
    plt.figure(figsize=(7, 4.5))
    sns.lineplot(data=df, x="steps", y="inception_score", hue="model", style="sde", markers=True, dashes=False, linewidth=2)
    plt.title("Inception Score vs Training Steps (Higher is Better)")
    plt.xlabel("Training Steps")
    plt.ylabel("Inception Score (IS)")
    plt.tight_layout()
    out = save_dir / "02_is_vs_steps.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_fid_vs_width(df: pd.DataFrame, save_dir: Path) -> Optional[Path]:
    """Plot 3: FID vs Model Width (Base Channels)."""
    if df.empty or "width" not in df.columns or "fid" not in df.columns:
        return None
    plt.figure(figsize=(7, 4.5))
    sns.lineplot(data=df, x="width", y="fid", hue="model", marker="o", linewidth=2)
    plt.title("Effect of Model Width on FID")
    plt.xlabel("Base Channels (Width)")
    plt.ylabel("FID Score")
    plt.tight_layout()
    out = save_dir / "03_fid_vs_width.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_fid_vs_depth(df: pd.DataFrame, save_dir: Path) -> Optional[Path]:
    """Plot 4: FID vs Model Depth (Residual Blocks per level)."""
    if df.empty or "depth" not in df.columns or "fid" not in df.columns:
        return None
    plt.figure(figsize=(7, 4.5))
    sns.lineplot(data=df, x="depth", y="fid", hue="model", marker="s", linewidth=2)
    plt.title("Effect of Model Depth on FID")
    plt.xlabel("ResNet Blocks per Resolution (Depth)")
    plt.ylabel("FID Score")
    plt.tight_layout()
    out = save_dir / "04_fid_vs_depth.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_model_comparison(df: pd.DataFrame, save_dir: Path) -> Optional[Path]:
    """Plot 5: Model Architecture Comparison (FID & Inception Score)."""
    if df.empty or "model" not in df.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(data=df, x="model", y="fid", ax=axes[0], palette="viridis")
    axes[0].set_title("Architecture FID Comparison (Lower is Better)")
    axes[0].set_ylabel("FID")
    axes[0].set_xlabel("Architecture")

    sns.barplot(data=df, x="model", y="inception_score", ax=axes[1], palette="magma")
    axes[1].set_title("Architecture IS Comparison (Higher is Better)")
    axes[1].set_ylabel("Inception Score")
    axes[1].set_xlabel("Architecture")

    plt.tight_layout()
    out = save_dir / "05_model_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_sde_comparison(df: pd.DataFrame, save_dir: Path) -> Optional[Path]:
    """Plot 6: SDE Formulation Comparison (VP vs VE vs Sub-VP)."""
    if df.empty or "sde" not in df.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(data=df, x="sde", y="fid", hue="model", ax=axes[0], palette="coolwarm")
    axes[0].set_title("SDE Formulation FID (Lower is Better)")
    axes[0].set_ylabel("FID")
    axes[0].set_xlabel("SDE Formulation")

    sns.barplot(data=df, x="sde", y="inception_score", hue="model", ax=axes[1], palette="coolwarm")
    axes[1].set_title("SDE Formulation Inception Score")
    axes[1].set_ylabel("Inception Score")
    axes[1].set_xlabel("SDE Formulation")

    plt.tight_layout()
    out = save_dir / "06_sde_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_training_loss_curves(df: pd.DataFrame, save_dir: Path) -> Optional[Path]:
    """Plot 7: Training Loss / Time comparison."""
    if df.empty or "training_time" not in df.columns:
        return None
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=df, x="model", y="training_time", hue="sde", palette="Set2")
    plt.title("Total Training Duration by Model & SDE")
    plt.xlabel("Model Architecture")
    plt.ylabel("Training Time (seconds)")
    plt.tight_layout()
    out = save_dir / "07_training_time_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_sampling_time_vs_quality(df: pd.DataFrame, save_dir: Path) -> Optional[Path]:
    """Plot 8: Sampling Time vs Image Quality Trade-off."""
    if df.empty or "sampling_time" not in df.columns or "fid" not in df.columns:
        return None
    plt.figure(figsize=(7.5, 4.5))
    sns.scatterplot(
        data=df,
        x="sampling_time",
        y="fid",
        hue="model",
        style="sde",
        size="parameters",
        sizes=(40, 200),
        alpha=0.8,
    )
    plt.title("Sampling Latency vs FID Quality Trade-off")
    plt.xlabel("Total Sampling Time (s)")
    plt.ylabel("FID Score (Lower is Better)")
    plt.tight_layout()
    out = save_dir / "08_sampling_time_vs_quality.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_pareto_params_vs_fid(df: pd.DataFrame, save_dir: Path) -> Optional[Path]:
    """Plot 9: Parameter Count vs FID Pareto Frontier."""
    if df.empty or "parameters" not in df.columns or "fid" not in df.columns:
        return None
    plt.figure(figsize=(7.5, 4.5))
    sns.scatterplot(
        data=df,
        x="parameters",
        y="fid",
        hue="model",
        style="sde",
        s=100,
    )
    plt.title("Model Capacity (Parameters) vs Generation Quality (FID)")
    plt.xlabel("Trainable Parameters")
    plt.ylabel("FID Score (Lower is Better)")
    plt.tight_layout()
    out = save_dir / "09_parameters_vs_fid.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_width_depth_heatmap(df: pd.DataFrame, save_dir: Path) -> Optional[Path]:
    """Plot 10: Width vs Depth Heatmap on FID."""
    if df.empty or "width" not in df.columns or "depth" not in df.columns or "fid" not in df.columns:
        return None
    try:
        pivot = df.pivot_table(index="depth", columns="width", values="fid", aggfunc="mean")
        if pivot.empty or pivot.shape[0] < 1 or pivot.shape[1] < 1:
            return None
        plt.figure(figsize=(6.5, 5))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu_r", cbar_kws={"label": "FID (Lower is Better)"})
        plt.title("FID Heatmap: Depth vs Width Interaction")
        plt.xlabel("Width (Base Channels)")
        plt.ylabel("Depth (Blocks per Resolution)")
        plt.tight_layout()
        out = save_dir / "10_width_depth_heatmap.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        return out
    except Exception:
        return None


def generate_all_plots(
    csv_path: Union[str, Path] = "results/metrics/results.csv",
    save_dir: Union[str, Path] = "results/plots",
) -> List[Path]:
    """Generate all 10 publication-ready research ablation figures."""
    cpath = Path(csv_path)
    sdir = Path(save_dir)
    sdir.mkdir(parents=True, exist_ok=True)

    if not cpath.exists():
        print(f"Results CSV not found at: {cpath}")
        return []

    df = pd.read_csv(cpath)
    if df.empty:
        print("Results CSV is empty. Run experiments first.")
        return []

    plot_funcs = [
        plot_fid_vs_steps,
        plot_is_vs_steps,
        plot_fid_vs_width,
        plot_fid_vs_depth,
        plot_model_comparison,
        plot_sde_comparison,
        plot_training_loss_curves,
        plot_sampling_time_vs_quality,
        plot_pareto_params_vs_fid,
        plot_width_depth_heatmap,
    ]

    generated = []
    for fn in plot_funcs:
        out = fn(df, sdir)
        if out is not None and out.exists():
            generated.append(out)

    return generated
