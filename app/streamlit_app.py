"""Streamlit Web Application: Ablation Study on Score-Based Generative Models."""

from pathlib import Path
import time
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Score-Based Generative Models Ablation",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Imports from src
from src.utils.device import get_device, get_gpu_memory_info
from src.utils.seed import set_seed
from src.data.transforms import unnormalize_to_zero_one
from src.models import get_model, count_parameters
from src.sde import get_sde
from src.diffusion.samplers import get_sampler
from src.evaluation.metrics import load_metrics_from_csv
from src.visualization.plots import generate_all_plots
from src.visualization.comparison import generate_leaderboard_table


@st.cache_resource
def get_cached_model(model_name: str, sde_name: str, width: int = 32, depth: int = 2, ckpt_path: str = ""):
    device = get_device("auto")
    sde = get_sde(sde_name)
    model = get_model(
        model_name,
        base_channels=width,
        num_res_blocks=depth,
        channel_multipliers=(1, 2, 2),
        attention_resolutions=(16,),
    ).to(device)

    if ckpt_path and Path(ckpt_path).exists():
        from src.training.checkpoint import load_checkpoint
        load_checkpoint(ckpt_path, model=model, device=device)

    model.eval()
    return model, sde, device


def main():
    st.sidebar.title("🔬 Score-Based SGM Study")
    st.sidebar.markdown(
        "**Ablation Study on Score-Based Generative Models**\n"
        "DDPM vs DDPM++ vs NCSN++ under VP, VE, and Sub-VP SDEs on CIFAR-10."
    )

    page = st.sidebar.radio(
        "Navigate",
        [
            "🎨 Generate Images",
            "📊 Compare Models",
            "🔍 Ablation Explorer",
            "📈 Training & Convergence",
            "🏆 Experiment Leaderboard",
        ],
    )

    # Hardware status in sidebar
    device = get_device("auto")
    mem = get_gpu_memory_info()
    st.sidebar.divider()
    st.sidebar.caption(f"**Runtime Device**: `{device}`")
    if mem["device"] != "cpu":
        st.sidebar.caption(f"**GPU Memory**: {mem['allocated_mb']:.1f} MB / {mem['reserved_mb']:.1f} MB")

    csv_path = Path("results/metrics/results.csv")
    df_metrics = load_metrics_from_csv(csv_path)

    # -------------------------------------------------------------
    # PAGE 1: Generate Images
    # -------------------------------------------------------------
    if page == "🎨 Generate Images":
        st.header("🎨 Interactive Image Generation")
        st.markdown(
            "Synthesize CIFAR-10 samples ($32\\times 32$) using reverse-time SDE integration or deterministic probability flows."
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Model Configuration")
            model_type = st.selectbox("Architecture", ["DDPM", "DDPM++", "NCSN++"], index=0)
            sde_type = st.selectbox("SDE Formulation", ["VP", "VE", "Sub-VP"], index=0)
            sampler_type = st.selectbox(
                "Sampler Integrator",
                ["Euler-Maruyama (SDE)", "Predictor-Corrector (PC)", "Probability Flow (ODE)"],
                index=0,
            )

            num_samples = st.slider("Number of Samples", min_value=1, max_value=36, value=16, step=1)
            steps = st.slider("Sampling Steps", min_value=10, max_value=500, value=100, step=10)
            seed = st.number_input("Random Seed", value=42, step=1)

            # Checkpoint options
            checkpoints = list(Path("checkpoints").glob("*.pt"))
            ckpt_options = ["None (Fresh Initialized)"] + [str(p) for p in checkpoints]
            chosen_ckpt = st.selectbox("Model Checkpoint", ckpt_options, index=0)
            ckpt_path = "" if "None" in chosen_ckpt else chosen_ckpt

            generate_btn = st.button("🚀 Generate Images", type="primary", use_container_width=True)

        with col2:
            st.subheader("Generated CIFAR-10 Samples")
            if generate_btn:
                clean_model = model_type.lower().replace("+", "pp")
                clean_sde = sde_type.lower().replace("-", "")
                sampler_key = "euler" if "Euler" in sampler_type else ("pc" if "PC" in sampler_type else "ode")

                with st.spinner("Generating samples via numerical reverse SDE integration..."):
                    set_seed(int(seed))
                    model, sde, dev = get_cached_model(clean_model, clean_sde, ckpt_path=ckpt_path)
                    sampler_fn = get_sampler(sampler_key)

                    start_t = time.time()
                    shape = (num_samples, 3, 32, 32)
                    with torch.no_grad():
                        samples = sampler_fn(
                            model=model,
                            sde=sde,
                            shape=shape,
                            device=dev,
                            num_steps=steps,
                            show_progress=False,
                        )
                    elapsed = time.time() - start_t
                    samples_01 = unnormalize_to_zero_one(samples)

                st.success(f"Generated {num_samples} samples in {elapsed:.2f}s ({elapsed/num_samples*1000:.1f}ms per image)")

                # Display in grid
                grid_cols = min(6, int(num_samples ** 0.5) + 1 if num_samples > 4 else num_samples)
                rows = [samples_01[i : i + grid_cols] for i in range(0, num_samples, grid_cols)]

                for row in rows:
                    cols = st.columns(grid_cols)
                    for idx, img_t in enumerate(row):
                        img_np = img_t.cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np * 255.0).round().astype("uint8")
                        pil_img = Image.fromarray(img_np).resize((128, 128), Image.NEAREST)
                        cols[idx].image(pil_img, use_container_width=True)
            else:
                # Show sample from results if available
                sample_files = list(Path("results/generated_samples").glob("**/*.png"))
                if sample_files:
                    st.image(str(sample_files[0]), caption="Sample Grid from Previous Experiment Run", use_container_width=True)
                else:
                    st.info("Click **Generate Images** to run generative sampling.")

    # -------------------------------------------------------------
    # PAGE 2: Compare Models
    # -------------------------------------------------------------
    elif page == "📊 Compare Models":
        st.header("📊 Architecture & SDE Formulation Benchmark")
        if df_metrics.empty:
            st.warning("No experiments recorded yet. Run `python scripts/run_ablation.py` to populate data.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("FID Score Comparison (Lower is Better)")
                chart_data = df_metrics.groupby(["model", "sde"])["fid"].min().unstack()
                st.bar_chart(chart_data)

            with col2:
                st.subheader("Inception Score (Higher is Better)")
                chart_is = df_metrics.groupby(["model", "sde"])["inception_score"].max().unstack()
                st.bar_chart(chart_is)

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Parameter Count (Model Capacity)")
                params_data = df_metrics.groupby("model")["parameters"].first()
                st.bar_chart(params_data)

            with col4:
                st.subheader("Sampling Latency (Total Seconds)")
                sampling_data = df_metrics.groupby("model")["sampling_time"].mean()
                st.bar_chart(sampling_data)

    # -------------------------------------------------------------
    # PAGE 3: Ablation Explorer
    # -------------------------------------------------------------
    elif page == "🔍 Ablation Explorer":
        st.header("🔍 Ablation Study Multi-Dimensional Explorer")
        if df_metrics.empty:
            st.warning("No ablation results found. Execute an ablation study first.")
        else:
            st.markdown("Filter and inspect the interaction between hyperparameters and generative quality.")

            col1, col2, col3 = st.columns(3)
            with col1:
                models_filter = st.multiselect("Filter Models", options=df_metrics["model"].unique().tolist(), default=df_metrics["model"].unique().tolist())
            with col2:
                sde_filter = st.multiselect("Filter SDEs", options=df_metrics["sde"].unique().tolist(), default=df_metrics["sde"].unique().tolist())
            with col3:
                metric_view = st.selectbox("Primary Metric", ["fid", "inception_score", "sampling_time", "training_time"])

            filtered_df = df_metrics[
                (df_metrics["model"].isin(models_filter)) & (df_metrics["sde"].isin(sde_filter))
            ]

            st.dataframe(filtered_df, use_container_width=True)

            # Interactive Plot
            st.subheader(f"Parameter Width vs {metric_view.upper()}")
            if "width" in filtered_df.columns:
                st.scatter_chart(filtered_df, x="width", y=metric_view, color="model")

    # -------------------------------------------------------------
    # PAGE 4: Training & Convergence
    # -------------------------------------------------------------
    elif page == "📈 Training & Convergence":
        st.header("📈 Training Convergence & Research Plots")
        plots_dir = Path("results/plots")
        plot_files = sorted(list(plots_dir.glob("*.png")))

        if plot_files:
            st.markdown("Publication-ready figures generated from experimental runs:")
            for p in plot_files:
                st.subheader(p.stem.replace("_", " ").title())
                st.image(str(p), use_container_width=True)
        else:
            st.info("No plots found in `results/plots/`. Run `python scripts/generate_plots.py` to produce figures from experimental metrics.")

    # -------------------------------------------------------------
    # PAGE 5: Experiment Leaderboard
    # -------------------------------------------------------------
    elif page == "🏆 Experiment Leaderboard":
        st.header("🏆 Experiment Leaderboard")
        if df_metrics.empty:
            st.info("No experiment records found in `results/metrics/results.csv`.")
        else:
            sort_by = st.selectbox("Sort Leaderboard By", ["fid", "inception_score", "training_time", "sampling_time", "parameters"], index=0)
            ascending = True if sort_by in ["fid", "training_time", "sampling_time", "parameters"] else False
            leaderboard = generate_leaderboard_table(df_metrics, sort_by=sort_by, ascending=ascending)

            st.dataframe(leaderboard, use_container_width=True)

            csv_data = leaderboard.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Leaderboard CSV",
                data=csv_data,
                file_name="score_models_leaderboard.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
