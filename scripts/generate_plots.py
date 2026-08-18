"""Plot generation CLI script."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization.plots import generate_all_plots
from src.evaluation.metrics import load_metrics_from_csv
from src.visualization.comparison import save_markdown_summary


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade ablation figures from results.csv.")
    parser.add_argument("--csv", type=str, default="results/metrics/results.csv", help="Path to results CSV.")
    parser.add_argument("--out_dir", type=str, default="results/plots", help="Directory to save generated plots.")
    args = parser.parse_args()

    plots = generate_all_plots(csv_path=args.csv, save_dir=args.out_dir)
    print(f"Successfully generated {len(plots)} publication plots in: '{args.out_dir}'")
    for p in plots:
        print(f"  - {p.name}")

    df = load_metrics_from_csv(args.csv)
    if not df.empty:
        summary_file = save_markdown_summary(df, save_path="results/tables/summary.md")
        print(f"Updated summary leaderboard at: '{summary_file}'")


if __name__ == "__main__":
    main()
