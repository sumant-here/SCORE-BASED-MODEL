"""Model comparison tables and analytical summaries."""

from pathlib import Path
from typing import Union
import pandas as pd


def generate_leaderboard_table(
    df: pd.DataFrame,
    sort_by: str = "fid",
    ascending: bool = True,
) -> pd.DataFrame:
    """Format and sort experiment metrics into a clean leaderboard."""
    if df.empty:
        return pd.DataFrame()

    required_cols = [
        "experiment_id",
        "model",
        "sde",
        "width",
        "depth",
        "parameters",
        "training_time",
        "sampling_time",
        "fid",
        "inception_score",
    ]
    present_cols = [c for c in required_cols if c in df.columns]
    table = df[present_cols].copy()

    if sort_by in table.columns:
        table = table.sort_values(by=sort_by, ascending=ascending)

    # Format parameter count
    if "parameters" in table.columns:
        table["params_formatted"] = table["parameters"].apply(lambda p: f"{p/1e6:.2f}M" if p >= 1e6 else f"{p/1e3:.1f}K")

    return table


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table without requiring tabulate."""
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[c]) for c in df.columns) + " |"
        rows.append(row_str)
    return "\n".join([header_line, separator_line] + rows) + "\n"


def save_markdown_summary(
    df: pd.DataFrame,
    save_path: Union[str, Path] = "results/tables/summary.md",
) -> Path:
    """Generate Markdown summary table and save to disk."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    leaderboard = generate_leaderboard_table(df)
    md_content = "# Score-Based Generative Models: Ablation Leaderboard\n\n"
    if not leaderboard.empty:
        try:
            md_content += leaderboard.to_markdown(index=False)
        except Exception:
            md_content += dataframe_to_markdown(leaderboard)
    else:
        md_content += "_No experiments recorded yet._\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return path
