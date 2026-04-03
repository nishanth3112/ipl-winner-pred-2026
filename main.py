"""
IPL 2026 Winner Prediction - Main Entry Point

Uses REAL IPL ball-by-ball data (IPL.csv, 2008-2025, 1169 matches).

Usage:
  python main.py --mode setup      # Extract data from IPL.csv and engineer features
  python main.py --mode train      # Train all models
  python main.py --mode predict    # Predict 2026 winner
  python main.py --mode all        # Run full pipeline end-to-end
  python main.py --mode visualize  # Generate charts (needs predict to run first)
"""
import argparse
import logging
import os
import sys
import time

import config
from config import LOG_FILE, LOG_LEVEL

# Logging setup
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def configure_dataset_path(dataset_path: str | None = None):
    if dataset_path:
        resolved_path = os.path.abspath(os.path.expanduser(dataset_path))
        os.environ["IPL_CSV_PATH"] = resolved_path
        config.IPL_SOURCE_CSV = resolved_path
        return resolved_path
    return config.IPL_SOURCE_CSV


def mode_setup(dataset_path: str | None = None):
    logger.info("=== SETUP: Extracting real IPL data and building features ===")
    t0 = time.time()

    configure_dataset_path(dataset_path)

    from src.data.create_dataset import (
        save_teams_json, build_all_matches, save_matches_csv, save_player_stats_csv,
    )
    from src.data.db_setup     import setup_database
    from src.data.ingest       import run_ingestion
    from src.data.preprocess   import run_preprocessing
    from src.features.engineer import run_feature_engineering

    logger.info("Step 1/5: Extracting match data from the IPL source CSV...")
    save_teams_json()
    matches, player_stats = build_all_matches(
        return_format="dataframes",
        dataset_path=dataset_path,
    )
    save_matches_csv(matches)
    save_player_stats_csv(player_stats)

    logger.info("Step 2/5: Creating SQLite database schema...")
    setup_database()

    logger.info("Step 3/5: Ingesting data into SQLite...")
    run_ingestion()

    logger.info("Step 4/5: Preprocessing matches...")
    run_preprocessing()

    logger.info("Step 5/5: Engineering features...")
    run_feature_engineering()

    logger.info(f"Setup complete in {time.time()-t0:.1f}s")


def mode_train():
    logger.info("=== TRAIN: Training all models ===")
    t0 = time.time()

    from src.models.trainer import run_training
    results = run_training()

    logger.info(f"Training complete in {time.time()-t0:.1f}s")
    return results


def mode_predict(dataset_path: str | None = None):
    logger.info("=== PREDICT: IPL 2026 Winner Prediction ===")
    t0 = time.time()

    configure_dataset_path(dataset_path)

    from src.prediction.predict_2026 import (
        predict_2026_winner, print_predictions, save_predictions,
    )
    rankings = predict_2026_winner()
    print_predictions(rankings)
    save_predictions(rankings)

    logger.info(f"Prediction complete in {time.time()-t0:.1f}s")
    return rankings


def mode_visualize():
    logger.info("=== VISUALIZE: Generating charts ===")
    from src.prediction.visualize import generate_all_charts
    generate_all_charts()


def mode_all(dataset_path: str | None = None):
    mode_setup(dataset_path=dataset_path)
    mode_train()
    rankings = mode_predict(dataset_path=dataset_path)
    try:
        mode_visualize()
    except Exception as e:
        logger.warning(f"Visualization failed (non-critical): {e}")
    return rankings


def parse_args():
    parser = argparse.ArgumentParser(
        description="IPL 2026 Winner Prediction (Real Data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["setup", "train", "predict", "visualize", "all"],
        default="all",
        help="Pipeline mode to run (default: all)",
    )
    parser.add_argument(
        "--ipl-csv",
        default=None,
        help="Path to the source IPL ball-by-ball CSV. "
             "If omitted, uses IPL_CSV_PATH or ./IPL.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting IPL 2026 prediction pipeline | mode={args.mode}")

    if args.mode == "setup":
        mode_setup(dataset_path=args.ipl_csv)
    elif args.mode == "train":
        mode_train()
    elif args.mode == "predict":
        mode_predict(dataset_path=args.ipl_csv)
    elif args.mode == "visualize":
        mode_visualize()
    elif args.mode == "all":
        mode_all(dataset_path=args.ipl_csv)
