#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Each entry: (step_name, script_path, skip_if_exists_path_or_None, category)
SCRIPTS_TO_RUN = [
    # 1. Ingestion (skip when raw data already on disk)
    ("Fetch 1min Bars", "src/ingestion/fetch_bars.py", "data/raw/candles_1m.parquet", "ingestion"),
    ("Fetch News Sentiment", "src/ingestion/fetch_sentiment.py", "data/raw/news_daily", "ingestion"),

    # 2. Canonicalization
    ("Build Daily Sentiment", "src/canonicalize/build_sentiment.py", "data/interim/canonical/sentiment_scores.parquet", "canonicalize"),

    # 3. Feature Engineering (Independent Modules)
    ("Compute Daily Features", "src/features/daily_features.py", "data/processed/features/daily_features.parquet", "features"),
    ("Compute Open Features", "src/features/open_features.py", "data/processed/features/open_features.parquet", "features"),
    ("Compute Market Context", "src/features/market_context.py", "data/processed/features/market_context.parquet", "features"),
    ("Compute Sentiment Features", "src/features/sentiment_features.py", "data/processed/features/sentiment_features.parquet", "features"),

    # 4. Assemble Features
    ("Assemble All Features", "src/features/assemble_features.py", "data/processed/features_table.parquet", "features"),

    # 5. Base Instances + Labels (instances need daily_features, so must come after step 3)
    ("Build Trade Instances", "src/dataset/build_instances.py", "data/processed/trade_instances.parquet", "dataset"),
    ("Generate Target Labels", "src/labeling/generate_labels.py", "data/processed/labels/labels.parquet", "dataset"),

    # 6. Final Dataset Assembly
    ("Assemble Final Dataset", "src/dataset/assemble_dataset.py", "data/processed/dataset_open30m.parquet", "dataset"),
]


def run_step(
    name: str,
    script_path: str,
    skip_if_exists: str | None = None,
    force_fresh: bool = False,
    skip_category: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    if skip_category:
        print("\n" + "=" * 60)
        print(f"SKIPPING CATEGORY: {name}")
        print("   (Skipped via command line flag)")
        print("=" * 60)
        return

    if not force_fresh and skip_if_exists is not None:
        abs_skip = os.path.join(PROJECT_ROOT, skip_if_exists)
        should_skip = False

        if os.path.exists(abs_skip):
            if os.path.isdir(abs_skip):
                # If it is a directory, only skip if it contains any files.
                if any(os.scandir(abs_skip)):
                    should_skip = True
            else:
                should_skip = True

        if should_skip:
            print("\n" + "=" * 60)
            print(f"SKIPPING: {name}")
            print(f"   Output already exists: {abs_skip}")
            print("=" * 60)
            return

    print("\n" + "=" * 60)
    print(f"RUNNING: {name}")
    print(f"   Script: {script_path}")
    print("=" * 60)

    script_path = os.path.normpath(script_path)
    abs_script_path = os.path.join(PROJECT_ROOT, script_path)

    if not os.path.exists(abs_script_path):
        print(f"\nERROR: Script not found at {abs_script_path}")
        sys.exit(1)

    cmd = [sys.executable, abs_script_path]
    if extra_args:
        cmd.extend(extra_args)

    # Inject project root into PYTHONPATH so `src...` imports work globally.
    env = os.environ.copy()
    current_ppath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{current_ppath}" if current_ppath else PROJECT_ROOT

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Pipeline aborted: Step '{name}' failed with exit code {e.returncode}.")
        sys.exit(e.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the research pipeline.")
    parser.add_argument("--ingestion", action="store_true", help="Skip the 'ingestion' category scripts")
    parser.add_argument("--canonicalization", action="store_true", help="Skip the 'canonicalize' category scripts")
    parser.add_argument(
        "--feature-engineering",
        action="store_true",
        dest="feature_engineering",
        help="Skip the 'features' category scripts",
    )
    parser.add_argument("--dataset", action="store_true", help="Skip the 'dataset' category scripts")
    parser.add_argument(
        "--architecture",
        default=None,
        help="Optional architecture manifest. Forwarded to label generation so stop-distance / rr_multiples can be versioned outside labels.yaml.",
    )
    args = parser.parse_args()

    print("============================================================")
    print("               RESEARCH PIPELINE BUILDER                   ")
    print("============================================================")

    for name, script, skip_path, category in SCRIPTS_TO_RUN:
        skip_category = False
        if category == "ingestion" and args.ingestion:
            skip_category = True
        elif category == "canonicalize" and args.canonicalization:
            skip_category = True
        elif category == "features" and args.feature_engineering:
            skip_category = True
        elif category == "dataset" and args.dataset:
            skip_category = True

        extra_args = None
        if script == "src/labeling/generate_labels.py" and args.architecture:
            extra_args = ["--architecture", args.architecture]

        run_step(
            name,
            script,
            skip_path,
            force_fresh=True,
            skip_category=skip_category,
            extra_args=extra_args,
        )

    print("\nPIPELINE COMPLETED SUCCESSFULLY!")
    print("Assembled labeled dataset is ready at `data/processed/dataset_open30m.parquet`.")


if __name__ == "__main__":
    main()
