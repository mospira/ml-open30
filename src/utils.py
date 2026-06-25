import os
import json
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

PIPELINE_CFG = os.path.join(PROJECT_ROOT, "configs", "pipeline.yaml")

def parse_date(date_val, tz="America/New_York") -> int:
    """Parse a date string to MS epoch, or return as-is if already int/float."""
    import pandas as pd
    if isinstance(date_val, (int, float)): 
        return int(date_val)
    # Parse string, ensure tz-awareness
    dt = pd.to_datetime(date_val)
    if dt.tzinfo is None:
        dt = dt.tz_localize(tz)
    return int(dt.timestamp() * 1000)

def load_pipeline_config() -> dict:
    """Load pipeline.yaml and return config dict."""
    with open(PIPELINE_CFG, "r") as f:
        cfg = yaml.safe_load(f)
        
    tz = cfg.get("project", {}).get("timezone", "America/New_York")
    dw = cfg.get("data_window", {})
    if "start_date" in dw: 
        dw["start_date"] = parse_date(dw["start_date"], tz)
    if "end_date" in dw: 
        dw["end_date"] = parse_date(dw["end_date"], tz)
        
    return cfg

def load_universe() -> list[str]:
    """Return list of tickers from the universe file + benchmark ETFs."""
    cfg = load_pipeline_config()
    universe_rel_path = cfg.get("universe", {}).get("file", "data/interim/canonical/universe_sp500.json")
    universe_path = os.path.join(PROJECT_ROOT, *universe_rel_path.split("/"))
    with open(universe_path, "r") as f:
        tickers = json.load(f)
    # Merge in benchmark tickers (SPY, QQQ) — they're ETFs not in S&P 500 index
    benchmarks = cfg.get("benchmarks", {}).get("tickers", [])
    for b in benchmarks:
        if b not in tickers:
            tickers.append(b)
    return tickers
