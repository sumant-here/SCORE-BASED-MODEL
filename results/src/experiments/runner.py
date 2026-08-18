"""Automated ablation experiment runner and benchmark coordinator."""

import time
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd

from src.utils.config import Config, load_config
from src.utils.logging import setup_logger
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import ExperimentMetrics, load_metrics_from_csv
from src.experiments.ablation import expand_ablation_matrix


class ExperimentRunner:
    """Automated ablation experiment runner across architectures, SDEs, widths, depths, and hyperparameters."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.logger = setup_logger("ExperimentRunner")
        self.metrics_file = self.output_dir / "metrics" / "results.csv"

    def run_experiment(self, config: Union[Config, dict]) -> ExperimentMetrics:
        """Run a single experiment (Train -> Evaluate -> Log)."""
        if isinstance(config, dict) and not isinstance(config, Config):
            cfg = Config(config)
        else:
            cfg = config

        exp_name = cfg.get("experiment_name", "experiment")
        self.logger.info(f"========== Starting Experiment: {exp_name} ==========")

        # 1. Train
        trainer = Trainer(cfg)
        train_results = trainer.train()

        # 2. Evaluate
        checkpoint_path = Path(cfg.get("checkpoint_dir", "checkpoints")) / f"{exp_name}_best.pt"
        if not checkpoint_path.exists():
            checkpoint_path = Path(cfg.get("checkpoint_dir", "checkpoints")) / f"{exp_name}_latest.pt"

        evaluator = Evaluator(cfg, checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None)
        evaluator.training_time = train_results.get("training_time", 0.0)

        num_eval_samples = cfg.get("evaluation", {}).get("num_samples", 64)
        sampler_name = cfg.get("evaluation", {}).get("sampler", "euler")
        sampler_steps = cfg.get("evaluation", {}).get("sampler_steps", 100)

        metrics = evaluator.evaluate(
            num_samples=num_eval_samples,
            sampler_name=sampler_name,
            sampler_steps=sampler_steps,
            output_dir=str(self.output_dir),
        )

        self.logger.info(
            f"Finished {exp_name} | FID: {metrics.fid:.2f} | IS: {metrics.inception_score:.2f} | Params: {metrics.parameters:,}"
        )
        return metrics

    def run_ablation(self, ablation_config_or_path: Union[str, Path, Config, dict]) -> pd.DataFrame:
        """Run full ablation study from config or YAML path."""
        if isinstance(ablation_config_or_path, (str, Path)):
            ablation_cfg = load_config(ablation_config_or_path)
        else:
            ablation_cfg = ablation_config_or_path

        run_configs = expand_ablation_matrix(ablation_cfg)
        self.logger.info(f"Generated {len(run_configs)} configurations for ablation sweep")

        results = []
        for i, cfg in enumerate(run_configs, 1):
            self.logger.info(f"--- Ablation Sweep Progress: [{i}/{len(run_configs)}] ---")
            try:
                metrics = self.run_experiment(cfg)
                results.append(metrics)
            except Exception as e:
                self.logger.error(f"Experiment failed for {cfg.get('experiment_name')}: {e}", exc_info=True)

        # Load and return updated metrics
        df = load_metrics_from_csv(self.metrics_file)
        return df
