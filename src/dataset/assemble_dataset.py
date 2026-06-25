import sys
from pathlib import Path
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def assemble_dataset(instances_path: str, features_path: str, labels_path: str, output_path: str) -> pd.DataFrame:
    """
    Joins the base instances with their corresponding generated features and labels.
    Handles matching correctly to form the final flat dataset.
    """
    print(f"Loading instances from {instances_path}...")
    df_instances = pd.read_parquet(instances_path)
    
    print(f"Loading features from {features_path}...")
    df_features = pd.read_parquet(features_path)
    
    print(f"Loading labels from {labels_path}...")
    df_labels = pd.read_parquet(labels_path)
    
    # 1. Join Instances with Features
    # Typically features are agnostic to 'side', so we join solely on [date, ticker]
    print("Joining instances with features...")
    df_combined = pd.merge(
        df_instances, 
        df_features, 
        on=['date', 'ticker'], 
        how='inner'  # Drop instances outside the features date range (e.g. warm-up period)
    )
    
    # 2. Join combined data with Labels
    # Labels inherently depend on [date, ticker, side] since outcome differs
    print("Joining combined data with labels...")
    join_keys_labels = ['date', 'ticker', 'side']
    
    # Verify the target columns exist in the labels dataframe
    missing_keys = [k for k in join_keys_labels if k not in df_labels.columns]
    if missing_keys:
         raise ValueError(f"Labels dataframe is missing required join key(s): {missing_keys}")
         
    df_final = pd.merge(
        df_combined,
        df_labels,
        on=join_keys_labels,
        how='inner' # Enforce that only rows with valid labels are included in the final dataset
    )
    
    print(f"Final assembled dataset shape: {df_final.shape}")
    
    # Save the dataframe
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_final.to_parquet(output_path, index=False)
    print(f"Saved completed dataset to {output_path}")
    
    return df_final

if __name__ == "__main__":
    # Example Usage (paths would typically come from pipeline.yaml or argparse)
    root = Path(project_root)
    
    inst_path = str(root / "data" / "processed" / "trade_instances.parquet")
    feat_path = str(root / "data" / "processed" / "features_table.parquet")
    lbl_path  = str(root / "data" / "processed" / "labels" / "labels.parquet") # Output naming from `generate_labels.py`
    out_path  = str(root / "data" / "processed" / "dataset_open30m.parquet")
    
    # Check dependencies before running
    if Path(inst_path).exists() and Path(feat_path).exists() and Path(lbl_path).exists():
        assemble_dataset(inst_path, feat_path, lbl_path, out_path)
    else:
        print("Missing one or more dependency files (instances, features, or labels). Cannot assemble dataset.")
