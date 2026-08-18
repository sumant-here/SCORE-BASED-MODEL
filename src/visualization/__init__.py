"""Visualization package."""

from src.visualization.samples import (
    generate_sample_grid,
    generate_fixed_seed_comparison,
)
from src.visualization.plots import (
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
    generate_all_plots,
)
from src.visualization.comparison import (
    generate_leaderboard_table,
    save_markdown_summary,
)

__all__ = [
    "generate_sample_grid",
    "generate_fixed_seed_comparison",
    "plot_fid_vs_steps",
    "plot_is_vs_steps",
    "plot_fid_vs_width",
    "plot_fid_vs_depth",
    "plot_model_comparison",
    "plot_sde_comparison",
    "plot_training_loss_curves",
    "plot_sampling_time_vs_quality",
    "plot_pareto_params_vs_fid",
    "plot_width_depth_heatmap",
    "generate_all_plots",
    "generate_leaderboard_table",
    "save_markdown_summary",
]
