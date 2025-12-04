import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

def synth_channel_signal(
    wavelength_nm: float,
    att_value: int,
    length: int = 1000,
    seed: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Synthesize realistic spark-OES/LIBS-like intensity trace.
    
    Signal composition:
    - 1-3 Gaussian peaks (emission lines)
    - Slowly-varying baseline (linear drift + low-frequency component)
    - Gaussian noise (scales inversely with att_value)
    
    Args:
        wavelength_nm: Wavelength of channel (affects peak positions)
        att_value: Attenuator value [0-99] (higher = stronger signal, less noise)
        length: Number of time points
        seed: Random seed for reproducibility
        params: Optional dict with generation parameters
        
    Returns:
        (x, y, ground_truth_peaks) where:
        - x: time/scan index array
        - y: intensity values
        - ground_truth_peaks: list of dicts with {index, amplitude, width, position}
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default parameters
    if params is None:
        params = {}
    
    num_peaks = params.get('num_peaks', np.random.randint(1, 4))  # 1-3 peaks
    base_intensity = params.get('base_intensity', 50 + att_value * 0.5)
    
    # Create x-axis (time/scan index)
    x = np.arange(length)
    
    # Initialize signal with zeros
    y = np.zeros(length)
    
    # 1. Generate baseline (slowly varying)
    # Linear drift component
    drift_slope = np.random.uniform(-0.02, 0.02)
    linear_baseline = base_intensity + drift_slope * x
    
    # Low-frequency sinusoidal component (simulates instrumental drift)
    freq = np.random.uniform(0.001, 0.005)
    phase = np.random.uniform(0, 2 * np.pi)
    sine_baseline = base_intensity * 0.15 * np.sin(2 * np.pi * freq * x + phase)
    
    baseline = linear_baseline + sine_baseline
    y += baseline
    
    # 2. Generate Gaussian peaks (emission lines)
    ground_truth_peaks = []
    
    # Peak heights scale with att_value (higher attenuator = stronger peaks)
    peak_scale = 0.5 + (att_value / 99.0) * 1.5  # Scale factor [0.5, 2.0]
    
    for i in range(num_peaks):
        # Peak position (avoid edges)
        peak_center = np.random.uniform(length * 0.15, length * 0.85)
        
        # Peak amplitude (scales with att_value)
        base_amplitude = np.random.uniform(100, 300)
        amplitude = base_amplitude * peak_scale
        
        # Peak width (FWHM in scan points)
        width = np.random.uniform(15, 40)
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        
        # Generate Gaussian peak
        gaussian = amplitude * np.exp(-0.5 * ((x - peak_center) / sigma) ** 2)
        y += gaussian
        
        # Store ground truth
        ground_truth_peaks.append({
            'index': i,
            'position': float(peak_center),
            'amplitude': float(amplitude),
            'width': float(width),
            'sigma': float(sigma),
            'wavelength_nm': float(wavelength_nm),
            'att_value': int(att_value)
        })
    
    # Sort peaks by position
    ground_truth_peaks.sort(key=lambda p: p['position'])
    for i, peak in enumerate(ground_truth_peaks):
        peak['index'] = i
    
    # 3. Add noise (inversely proportional to att_value)
    # Lower att_value = higher noise (less signal attenuation = more noise)
    noise_std = 30 * (100 - att_value) / 100 + 5
    noise = np.random.normal(0, noise_std, length)
    y += noise
    
    # Ensure no negative values (physical constraint for intensity)
    y = np.maximum(y, 0)
    
    return x, y, ground_truth_peaks

def generate_all_signals(
    meta_df: pd.DataFrame,
    length: int = 1000,
    save_folder: str = 'results/raw'
) -> Dict[str, Dict[str, str]]:
    """
    Generate synthetic signals for all channels in metadata.
    
    Args:
        meta_df: Metadata DataFrame with channel information
        length: Signal length (number of time points)
        save_folder: Folder to save raw signals and ground truth
        
    Returns:
        Dict mapping channel_id to {'csv': path, 'json': path}
    """
    print("\n=== Generating Synthetic Signals ===")
    
    # Create save folder
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Generate signals for each channel
    for idx, row in meta_df.iterrows():
        channel_id = row['channel_id']
        wavelength = row['wavelength_nm']
        att_value = row['att_value']
        ele_name = row['ele_name']
        
        # Skip if wavelength is invalid
        if pd.isna(wavelength) or wavelength <= 0:
            print(f"⚠ Skipping {channel_id}: invalid wavelength")
            continue
        
        # Create reproducible seed from channel_id
        seed = abs(hash(channel_id)) % (2**31)
        
        # Generate signal
        x, y, peaks = synth_channel_signal(
            wavelength_nm=wavelength,
            att_value=att_value,
            length=length,
            seed=seed
        )
        
        # Save raw signal as CSV
        csv_path = save_path / f"raw_channel_{channel_id}.csv"
        signal_df = pd.DataFrame({'x': x, 'y': y})
        signal_df.to_csv(csv_path, index=False)
        
        # Save ground truth as JSON
        json_path = save_path / f"ground_truth_{channel_id}.json"
        ground_truth = {
            'channel_id': channel_id,
            'element': ele_name,
            'wavelength_nm': float(wavelength),
            'att_value': int(att_value),
            'signal_length': length,
            'peaks': peaks
        }
        with open(json_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        file_paths[channel_id] = {
            'csv': str(csv_path),
            'json': str(json_path)
        }
        
        if (idx + 1) % 5 == 0:
            print(f"  Generated {idx + 1}/{len(meta_df)} signals...")
    
    print(f"✓ Generated signals for {len(file_paths)} channels")
    print(f"✓ Saved to: {save_folder}")
    
    return file_paths

def plot_raw_overview(
    meta_df: pd.DataFrame,
    chosen_channel_ids: List[str],
    save_folder: str = 'results/raw',
    output_path: str = 'results/raw_overview.png'
) -> None:
    """
    Create multi-panel overview of raw signals for selected channels.
    
    Args:
        meta_df: Metadata DataFrame
        chosen_channel_ids: List of channel IDs to plot (typically 3)
        save_folder: Folder containing raw signal CSVs
        output_path: Output path for overview plot
    """
    print(f"\n=== Creating Raw Signal Overview ===")
    
    n_channels = len(chosen_channel_ids)
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3.5 * n_channels), sharex=True)
    
    if n_channels == 1:
        axes = [axes]
    
    for idx, channel_id in enumerate(chosen_channel_ids):
        # Load signal
        csv_path = Path(save_folder) / f"raw_channel_{channel_id}.csv"
        signal_df = pd.read_csv(csv_path)
        
        # Get metadata
        channel_meta = meta_df[meta_df['channel_id'] == channel_id].iloc[0]
        ele_name = channel_meta['ele_name']
        wavelength = channel_meta['wavelength_nm']
        att_value = channel_meta['att_value']
        
        # Plot
        ax = axes[idx]
        ax.plot(signal_df['x'], signal_df['y'], linewidth=1.0, color='#2E86AB', alpha=0.8)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Labels and title
        title = f"{ele_name} ({wavelength:.3f} nm) | Attenuator: {att_value}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold')
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
    
    # Common x-label
    axes[-1].set_xlabel('Scan Index', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved overview to: {output_path}")

def plot_slide_raw(
    meta_df: pd.DataFrame,
    channel_id: str,
    save_folder: str = 'results/raw',
    output_path: Optional[str] = None
) -> None:
    """
    Create slide-ready plot of single raw signal with annotated peaks.
    
    Args:
        meta_df: Metadata DataFrame
        channel_id: Channel ID to plot
        save_folder: Folder containing raw signals and ground truth
        output_path: Output path for plot (default: results/slide_raw_channel{channel_id}.png)
    """
    if output_path is None:
        output_path = f"results/slide_raw_{channel_id}.png"
    
    print(f"\n=== Creating Slide Plot for {channel_id} ===")
    
    # Load signal
    csv_path = Path(save_folder) / f"raw_channel_{channel_id}.csv"
    signal_df = pd.read_csv(csv_path)
    
    # Load ground truth
    json_path = Path(save_folder) / f"ground_truth_{channel_id}.json"
    with open(json_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Get metadata
    channel_meta = meta_df[meta_df['channel_id'] == channel_id].iloc[0]
    ele_name = channel_meta['ele_name']
    wavelength = channel_meta['wavelength_nm']
    att_value = channel_meta['att_value']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot signal
    ax.plot(signal_df['x'], signal_df['y'], linewidth=1.5, color='#2E86AB', 
            alpha=0.9, label='Raw Intensity')
    
    # Annotate ground truth peaks
    peaks = ground_truth['peaks']
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(peaks)))
    
    for i, peak in enumerate(peaks):
        position = peak['position']
        width = peak['width']
        amplitude = peak['amplitude']
        
        # Draw translucent vertical box around peak
        left = position - width
        right = position + width
        ax.axvspan(left, right, alpha=0.2, color=colors[i], label=f"Peak {i+1}")
        
        # Annotate peak
        ax.axvline(position, color=colors[i], linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Add text annotation
        y_pos = signal_df['y'].max() * 0.9
        ax.text(position, y_pos, f"P{i+1}\n{amplitude:.0f}", 
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
    
    # Labels and title
    title = f"Raw Signal: {ele_name} at {wavelength:.3f} nm (Att={att_value})"
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Scan Index', fontsize=16, fontweight='bold')
    ax.set_ylabel('Intensity (a.u.)', fontsize=16, fontweight='bold')
    
    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved slide plot to: {output_path}")

def main(meta_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Main execution: generate all signals and create visualizations.
    
    Args:
        meta_df: Metadata DataFrame from metadata builder
        
    Returns:
        Dictionary of file paths for all generated signals
    """
    print("\n" + "="*70)
    print("SYNTHETIC SIGNAL GENERATION PIPELINE")
    print("="*70)
    
    # Generate all signals
    file_paths = generate_all_signals(meta_df, length=1000, save_folder='results/raw')
    
    # Select 3 representative channels for visualization
    # Pick channels with different elements and varying attenuator values
    unique_elements = meta_df['ele_name'].unique()[:3]
    chosen_channels = []
    
    for elem in unique_elements:
        elem_channels = meta_df[meta_df['ele_name'] == elem]
        if not elem_channels.empty:
            # Pick channel with mid-range attenuator value
            mid_att_channel = elem_channels.iloc[len(elem_channels) // 2]
            chosen_channels.append(mid_att_channel['channel_id'])
    
    # Ensure we have exactly 3 channels
    if len(chosen_channels) < 3:
        all_channels = meta_df['channel_id'].tolist()
        while len(chosen_channels) < 3 and len(chosen_channels) < len(all_channels):
            for ch in all_channels:
                if ch not in chosen_channels:
                    chosen_channels.append(ch)
                    break
    
    chosen_channels = chosen_channels[:3]
    
    print(f"\nSelected channels for visualization: {chosen_channels}")
    
    # Create overview plot
    plot_raw_overview(meta_df, chosen_channels, output_path='results/raw_overview.png')
    
    # Create slide-ready plots for each chosen channel
    for channel_id in chosen_channels:
        plot_slide_raw(meta_df, channel_id, output_path=f"results/slide_raw_{channel_id}.png")
    
    # Print summary
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    print(f"Total signals generated: {len(file_paths)}")
    print(f"Visualization channels: {len(chosen_channels)}")
    print(f"\nOutput files:")
    print(f"  - {len(file_paths)} CSV files in results/raw/")
    print(f"  - {len(file_paths)} JSON ground truth files in results/raw/")
    print(f"  - 1 overview plot: results/raw_overview.png")
    print(f"  - {len(chosen_channels)} slide plots: results/slide_raw_*.png")
    
    return file_paths

# Example usage (run this after loading meta_df from metadata builder)
if __name__ == "__main__":
    # This assumes meta_df is available from the metadata builder
    # For standalone testing, load from CSV:
    # meta_df = pd.read_csv('results/metadata_preview.csv')
    
    print("Note: Import this module and call main(meta_df) with your metadata DataFrame")
