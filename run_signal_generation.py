"""
Integration script to run signal generation pipeline.
Run this after executing json_metadata_builder.py
"""
import pandas as pd
from json_metadata_builder import main as build_metadata
from signal_generator import main as generate_signals

# Step 1: Build metadata (if not already done)
print("Step 1: Building metadata from JSON...")
meta_df = build_metadata()

# Step 2: Generate synthetic signals
print("\nStep 2: Generating synthetic signals...")
file_paths = generate_signals(meta_df)

print("\n" + "="*70)
print("PIPELINE COMPLETE")
print("="*70)
print("\nYou can now use:")
print("  - meta_df: Metadata DataFrame")
print("  - file_paths: Dictionary of generated signal paths")
