"""Metrics data models and CSV persistence."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd


@dataclass
class ExperimentMetrics:
    """Standardized record of an experimental run's evaluation metrics."""

    experiment_id: str
    model: str
    sde: str
    width: int
    depth: int
    learning_rate: float
    steps: int
    parameters: int
    training_time: float
    sampling_time: float
    fid: float
    inception_score: float
    inception_score_std: float
    num_samples: int
    seed: int
    dataset: str = "cifar10"
    class_filter: str = "all"


def save_metrics_to_csv(
    metrics: Union[ExperimentMetrics, List[ExperimentMetrics]],
    csv_path: Union[str, Path] = "results/metrics/results.csv",
) -> pd.DataFrame:
    """Save or append experiment metrics to results CSV."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = [asdict(m) for m in (metrics if isinstance(metrics, list) else [metrics])]
    new_df = pd.DataFrame(records)

    if path.exists():
        existing_df = pd.read_csv(path)
        # Avoid duplicate experiment_id entries by updating or appending
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        # Drop duplicates keeping last
        combined = combined.drop_duplicates(subset=["experiment_id"], keep="last")
    else:
        combined = new_df

    combined.to_csv(path, index=False)
    return combined


def load_metrics_from_csv(
    csv_path: Union[str, Path] = "results/metrics/results.csv"
) -> pd.DataFrame:
    """Load experimental metrics from CSV."""
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
