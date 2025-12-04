"""
Complete end-to-end pipeline execution script.
Runs: Metadata Building → Signal Generation → Analysis Pipeline
"""
import pandas as pd
from json_metadata_builder import main as build_metadata
from signal_generator import main as generate_signals
from analysis_pipeline import main as run_analysis

def main():
    print("="*70)
    print("COMPLETE END-TO-END PIPELINE EXECUTION")
    print("="*70)
    
    # Step 1: Build metadata
    print("\n### STEP 1: Building Metadata ###")
    meta_df = build_metadata()
    
    # Step 2: Generate synthetic signals
    print("\n### STEP 2: Generating Synthetic Signals ###")
    file_paths = generate_signals(meta_df)
    
    # Step 3: Run analysis pipeline
    print("\n### STEP 3: Running Analysis Pipeline ###")
    summary_df = run_analysis(meta_df)
    
    # Final summary
    print("\n" + "="*70)
    print("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print("="*70)
    print("\nGenerated Outputs:")
    print("  Metadata:")
    print("    - results/metadata_preview.csv")
    print("    - results/metadata_table.png")
    print("\n  Raw Signals:")
    print(f"    - {len(file_paths)} channels in results/raw/")
    print("    - results/raw_overview.png")
    print("    - results/slide_raw_*.png")
    print("\n  Analysis Results:")
    print("    - results/final/metrics_classical_vs_dl.csv")
    print("    - results/final/analysis_results.json")
    print("    - results/final/presentation_notes.txt")
    print(f"    - {len(summary_df)} pipeline panel PNGs")
    print("    - results/final/compare_snr_bar.png")
    print("    - results/final/peak_detection_table.png")
    
    return meta_df, summary_df

if __name__ == "__main__":
    meta_df, summary_df = main()
