"""
Compare features_table_linux.parquet and features_table_windows.parquet.

Run from a directory containing both files:
    python compare_features.py

Exits 0 if tables are identical, 1 otherwise.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np


def compare_features(dir_path: Path = Path(".")) -> bool:
    linux_path = dir_path / "features_table_linux.parquet"
    windows_path = dir_path / "features_table_windows.parquet"

    for p in (linux_path, windows_path):
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(2)

    df_linux = pd.read_parquet(linux_path)
    df_win = pd.read_parquet(windows_path)

    identical = True

    # --- Shape ---
    print(f"Linux  shape: {df_linux.shape}")
    print(f"Windows shape: {df_win.shape}")
    if df_linux.shape != df_win.shape:
        print("MISMATCH: shapes differ")
        identical = False

    # --- Columns ---
    cols_linux = set(df_linux.columns)
    cols_win = set(df_win.columns)
    only_linux = cols_linux - cols_win
    only_win = cols_win - cols_linux
    if only_linux:
        print(f"Columns only in Linux:   {sorted(only_linux)}")
        identical = False
    if only_win:
        print(f"Columns only in Windows: {sorted(only_win)}")
        identical = False

    common_cols = sorted(cols_linux & cols_win)
    if not common_cols:
        print("No common columns to compare.")
        return False

    # Column order
    if list(df_linux.columns) != list(df_win.columns):
        print("NOTE: column order differs (comparing on common columns regardless)")

    # --- Index ---
    if not df_linux.index.equals(df_win.index):
        print("MISMATCH: row indices differ")
        # Show first few differing indices
        if len(df_linux) == len(df_win):
            mask = df_linux.index != df_win.index
            n_diff = mask.sum()
            print(f"  {n_diff} / {len(df_linux)} index values differ")
            if n_diff > 0:
                print(f"  First few linux:   {df_linux.index[mask][:5].tolist()}")
                print(f"  First few windows: {df_win.index[mask][:5].tolist()}")
        identical = False

    # --- Dtypes ---
    dtype_mismatches = []
    for col in common_cols:
        if df_linux[col].dtype != df_win[col].dtype:
            dtype_mismatches.append((col, df_linux[col].dtype, df_win[col].dtype))
    if dtype_mismatches:
        print(f"\nDtype mismatches ({len(dtype_mismatches)}):")
        for col, dt_l, dt_w in dtype_mismatches:
            print(f"  {col}: linux={dt_l}, windows={dt_w}")
        identical = False

    # --- Value comparison (column-by-column) ---
    print(f"\nComparing {len(common_cols)} common columns...")
    mismatched_cols = []

    for col in common_cols:
        s_l = df_linux[col].reset_index(drop=True)
        s_w = df_win[col].reset_index(drop=True)

        # Truncate to shorter length if shapes differ
        min_len = min(len(s_l), len(s_w))
        s_l = s_l.iloc[:min_len]
        s_w = s_w.iloc[:min_len]

        if s_l.dtype.kind == "f" or s_w.dtype.kind == "f":
            # Float comparison with tolerance
            both_nan = s_l.isna() & s_w.isna()
            one_nan = s_l.isna() ^ s_w.isna()
            n_one_nan = one_nan.sum()

            neither_nan = ~(s_l.isna() | s_w.isna())
            if neither_nan.any():
                abs_diff = (s_l[neither_nan] - s_w[neither_nan]).abs()
                # Use relative + absolute tolerance like np.allclose defaults
                not_close = abs_diff > (1e-8 + 1e-5 * s_w[neither_nan].abs())
                n_not_close = not_close.sum()
            else:
                n_not_close = 0

            n_diff = n_one_nan + n_not_close
            if n_diff > 0:
                identical = False
                # Detailed stats for numeric diffs
                detail = f"  {col}: {n_diff}/{min_len} values differ"
                if n_one_nan > 0:
                    detail += f" ({n_one_nan} NaN mismatches)"
                if n_not_close > 0 and neither_nan.any():
                    max_abs = abs_diff.max()
                    max_rel = (abs_diff / (s_w[neither_nan].abs() + 1e-15)).max()
                    detail += f" (max abs diff={max_abs:.6e}, max rel diff={max_rel:.6e})"
                mismatched_cols.append(detail)

        else:
            # Exact comparison for non-float types
            # Treat NaN == NaN as equal
            eq = s_l == s_w
            both_na = s_l.isna() & s_w.isna()
            eq = eq | both_na
            n_diff = (~eq).sum()
            if n_diff > 0:
                identical = False
                mismatched_cols.append(
                    f"  {col}: {n_diff}/{min_len} values differ (dtype={s_l.dtype})"
                )

    if mismatched_cols:
        print(f"\nColumn value mismatches ({len(mismatched_cols)}):")
        for entry in mismatched_cols:
            print(entry)
    else:
        print("All column values match.")

    # --- Summary ---
    print()
    if identical:
        print("RESULT: Tables are IDENTICAL")
    else:
        print("RESULT: Tables DIFFER")

    return identical


if __name__ == "__main__":
    ok = compare_features()
    sys.exit(0 if ok else 1)
