"""Ablation Study CLI Runner: executes sweeping experiment matrices and logs benchmarks."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.runner import ExperimentRunner
from src.visualization.plots import generate_all_plots
from src.visualization.comparison import save_markdown_summary


def main():
    parser = argparse.ArgumentParser(description="Run automated ablation sweeps across models, SDEs, and hyperparams.")
    parser.add_argument("--config", type=str, required=True, help="Path to ablation experiment YAML configuration.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for metrics and outputs.")
    args = parser.parse_args()

    runner = ExperimentRunner(output_dir=args.output_dir)
    results_df = runner.run_ablation(args.config)

    print("\n" + "=" * 60)
    print("[+] ABLATION STUDY COMPLETE -- SUMMARY TABLE")
    print("=" * 60)
    if not results_df.empty:
        display_cols = [c for c in ["experiment_id", "model", "sde", "width", "depth", "fid", "inception_score", "parameters"] if c in results_df.columns]
        print(results_df[display_cols].to_string(index=False))

        # Generate all 10 plots
        print("\nGenerating publication-ready research plots...")
        plots = generate_all_plots(csv_path=Path(args.output_dir) / "metrics" / "results.csv", save_dir=Path(args.output_dir) / "plots")
        print(f"Generated {len(plots)} research plots in '{args.output_dir}/plots/'")

        # Save Markdown summary
        md_summary = save_markdown_summary(results_df, save_path=Path(args.output_dir) / "tables" / "summary.md")
        print(f"Saved Markdown summary to '{md_summary}'")
    else:
        print("No results recorded.")
    print("=" * 60)


if __name__ == "__main__":
    main()
