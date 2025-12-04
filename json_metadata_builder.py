import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

def load_json(path: str) -> dict:
    """
    Load JSON file with error handling.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Parsed JSON as dictionary
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Successfully loaded JSON from: {path}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def extract_sections(json_obj: dict) -> dict:
    """
    Extract major sections from JSON with defensive key handling.
    
    Args:
        json_obj: Parsed JSON object
        
    Returns:
        Dictionary with keys: attenuator_info, element_info, channel_info, analytical_conditions
    """
    sections = {}
    
    # Navigate to data section
    data = json_obj.get('data', json_obj)
    
    # Extract attenuator information
    att_keys = ['attenuator_information', 'attenuator_info', 'attenuator']
    for key in att_keys:
        if key in data:
            sections['attenuator_info'] = data[key]
            print(f"✓ Found attenuator info under key: '{key}'")
            break
    
    # Extract element information
    elem_keys = ['element_information', 'element_info', 'elements']
    for key in elem_keys:
        if key in data:
            sections['element_info'] = data[key]
            print(f"✓ Found element info under key: '{key}'")
            break
    
    # Extract channel information
    chan_keys = ['channel_information', 'channel_info', 'channels']
    for key in chan_keys:
        if key in data:
            sections['channel_info'] = data[key]
            print(f"✓ Found channel info under key: '{key}'")
            break
    
    # Extract analytical conditions
    cond_keys = ['analytical_conditions', 'conditions', 'analytical_params']
    for key in cond_keys:
        if key in data:
            sections['analytical_conditions'] = data[key]
            print(f"✓ Found analytical conditions under key: '{key}'")
            break
    
    print(f"Extracted {len(sections)} sections from JSON")
    return sections

def _safe_get_value(obj: dict, keys: List[str], default: Any = None) -> Any:
    """
    Try multiple keys to extract a value, case-insensitive.
    """
    for key in keys:
        if key in obj:
            return obj[key]
        # Try case-insensitive match
        for obj_key in obj.keys():
            if obj_key.lower() == key.lower():
                return obj[obj_key]
    return default

def _parse_wavelength(value: Any) -> Optional[float]:
    """
    Parse wavelength value to float, handling various formats.
    """
    if value is None or value == '':
        return None
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        warnings.warn(f"Could not parse wavelength: {value}")
        return None

def _parse_att_value(value: Any, default: int = 50) -> int:
    """
    Parse attenuator value to int, with range validation.
    """
    if value is None or value == '':
        return default
    try:
        val = int(float(str(value).strip()))
        return max(0, min(99, val))  # Clamp to [0, 99]
    except (ValueError, TypeError):
        return default

def build_metadata(json_obj: dict) -> pd.DataFrame:
    """
    Build canonical metadata DataFrame from JSON.
    
    Columns: group_id, group_name, condition_id, sequence_no, element_id, ele_name,
             wavelength_nm, att_value, channel_id, notes
             
    Args:
        json_obj: Parsed JSON object
        
    Returns:
        Metadata DataFrame
    """
    print("\n=== Building Metadata DataFrame ===")
    sections = extract_sections(json_obj)
    
    rows = []
    
    # Extract attenuator information
    att_info = sections.get('attenuator_info', {})
    elem_info = sections.get('element_info', {})
    chan_info = sections.get('channel_info', {})
    cond_info = sections.get('analytical_conditions', {})
    
    # Get bulk records from each section
    att_records = att_info.get('bulk', {}).get('records', [])
    elem_records = elem_info.get('bulk', {}).get('records', [])
    chan_records = chan_info.get('bulk', {}).get('records', [])
    cond_records = cond_info.get('bulk', {}).get('records', [])
    
    print(f"Found {len(att_records)} attenuator records")
    print(f"Found {len(elem_records)} element records")
    print(f"Found {len(chan_records)} channel records")
    print(f"Found {len(cond_records)} analytical condition records")
    
    # Build lookup dictionaries
    analytical_groups = {}
    
    # Process attenuator records
    for att_rec in att_records:
        group_name = _safe_get_value(att_rec, ['analytical_group', 'group_name', 'group'], 'Unknown')
        group_id = _safe_get_value(att_rec, ['id'], 0)
        
        analytical_groups[group_id] = group_name
        
        # Process left_table (primary elements)
        left_table = att_rec.get('left_table', [])
        for idx, elem in enumerate(left_table):
            element_name = _safe_get_value(elem, ['element', 'ele_name', 'Element'], f'Elem_{idx}')
            wavelength = _parse_wavelength(_safe_get_value(elem, ['ele_value', 'wavelength', 'w_lengh', 'wavelength_nm']))
            att_val = _parse_att_value(_safe_get_value(elem, ['att_value', 'attenuator']))
            
            channel_id = f"G{group_id}_C1_E{idx+1}"
            
            rows.append({
                'group_id': group_id,
                'group_name': group_name,
                'condition_id': 1,
                'sequence_no': idx + 1,
                'element_id': idx + 1,
                'ele_name': element_name,
                'wavelength_nm': wavelength,
                'att_value': att_val,
                'channel_id': channel_id,
                'notes': json.dumps({'source': 'left_table', 'table': 'attenuator'})
            })
        
        # Process right_table (secondary elements)
        right_table = att_rec.get('right_table', [])
        for idx, elem in enumerate(right_table):
            element_name = _safe_get_value(elem, ['element', 'ele_name', 'Element'], f'Elem_R{idx}')
            wavelength = _parse_wavelength(_safe_get_value(elem, ['ele_value', 'wavelength', 'w_lengh', 'wavelength_nm']))
            att_val = _parse_att_value(_safe_get_value(elem, ['att_value', 'attenuator']))
            
            channel_id = f"G{group_id}_C2_E{idx+1}"
            
            rows.append({
                'group_id': group_id,
                'group_name': group_name,
                'condition_id': 2,
                'sequence_no': idx + 1,
                'element_id': len(left_table) + idx + 1,
                'ele_name': element_name,
                'wavelength_nm': wavelength,
                'att_value': att_val,
                'channel_id': channel_id,
                'notes': json.dumps({'source': 'right_table', 'table': 'attenuator'})
            })
    
    # Enhance with element information
    for elem_rec in elem_records:
        group_name = _safe_get_value(elem_rec, ['analytical_group', 'group_name'], 'Unknown')
        elements = elem_rec.get('elements', [])
        
        for elem in elements:
            element_name = _safe_get_value(elem, ['ele_name', 'element', 'chemic_ele'])
            # Match with existing rows or create new
            found = False
            for row in rows:
                if row['ele_name'].upper() == str(element_name).upper() and row['group_name'] == group_name:
                    # Enhance with additional element info
                    notes = json.loads(row['notes'])
                    notes['full_element'] = _safe_get_value(elem, ['element', 'full_name'])
                    notes['range_min'] = _safe_get_value(elem, ['analytical_range_min'])
                    notes['range_max'] = _safe_get_value(elem, ['analytical_range_max'])
                    row['notes'] = json.dumps(notes)
                    found = True
    
    # Enhance with channel information
    for chan_rec in chan_records:
        channels = chan_rec.get('channels', [])
        
        for chan in channels:
            element_name = _safe_get_value(chan, ['ele_name', 'element'])
            wavelength = _parse_wavelength(_safe_get_value(chan, ['w_lengh', 'wavelength', 'w_length', 'wavelength_nm']))
            seq = _safe_get_value(chan, ['seq', 'sequence'], 1)
            
            # Update wavelength and sequence if found
            for row in rows:
                if row['ele_name'].upper() == str(element_name).upper():
                    if wavelength and not row['wavelength_nm']:
                        row['wavelength_nm'] = wavelength
                    row['sequence_no'] = int(seq) if seq else row['sequence_no']
    
    # Create DataFrame
    meta_df = pd.DataFrame(rows)
    
    # Ensure proper types
    if not meta_df.empty:
        meta_df['group_id'] = meta_df['group_id'].astype(int)
        meta_df['condition_id'] = meta_df['condition_id'].astype(int)
        meta_df['sequence_no'] = meta_df['sequence_no'].astype(int)
        meta_df['element_id'] = meta_df['element_id'].astype(int)
        meta_df['att_value'] = meta_df['att_value'].astype(int)
        meta_df['wavelength_nm'] = pd.to_numeric(meta_df['wavelength_nm'], errors='coerce')
    
    print(f"\n✓ Built metadata DataFrame with {len(meta_df)} rows")
    print(f"  - Groups: {meta_df['group_id'].nunique()}")
    print(f"  - Conditions: {meta_df['condition_id'].nunique()}")
    print(f"  - Elements: {meta_df['ele_name'].nunique()}")
    print(f"  - Channels: {meta_df['channel_id'].nunique()}")
    
    return meta_df

def validate_metadata(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and fix metadata DataFrame.
    
    - Ensures wavelength_nm is numeric
    - Ensures att_value is in [0, 99]
    - Logs any fixes applied
    
    Args:
        meta_df: Metadata DataFrame
        
    Returns:
        Validated DataFrame
    """
    print("\n=== Validating Metadata ===")
    
    if meta_df.empty:
        warnings.warn("Metadata DataFrame is empty")
        return meta_df
    
    # Validate wavelength_nm
    null_wavelengths = meta_df['wavelength_nm'].isna().sum()
    if null_wavelengths > 0:
        print(f"⚠ Warning: {null_wavelengths} rows have missing wavelength values")
    
    invalid_wavelengths = ((meta_df['wavelength_nm'] < 100) | (meta_df['wavelength_nm'] > 1000)).sum()
    if invalid_wavelengths > 0:
        print(f"⚠ Warning: {invalid_wavelengths} rows have suspicious wavelength values (expected 100-1000 nm)")
    
    # Validate att_value range
    out_of_range = ((meta_df['att_value'] < 0) | (meta_df['att_value'] > 99)).sum()
    if out_of_range > 0:
        print(f"⚠ Fixing {out_of_range} att_value entries outside [0, 99] range")
        meta_df['att_value'] = meta_df['att_value'].clip(0, 99)
    
    # Check for duplicates
    duplicates = meta_df['channel_id'].duplicated().sum()
    if duplicates > 0:
        print(f"⚠ Warning: {duplicates} duplicate channel_id values found")
    
    print(f"✓ Validation complete")
    return meta_df

def save_metadata_preview(meta_df: pd.DataFrame, path_csv: str, path_png: str) -> None:
    """
    Save metadata preview as CSV and PNG table.
    
    Args:
        meta_df: Metadata DataFrame
        path_csv: Path to save CSV (first 50 rows)
        path_png: Path to save PNG table (first 8 rows)
    """
    print(f"\n=== Saving Metadata Preview ===")
    
    # Create results directory
    Path(path_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV (first 50 rows)
    preview_df = meta_df.head(50)
    preview_df.to_csv(path_csv, index=False)
    print(f"✓ Saved CSV preview to: {path_csv}")
    
    # Create PNG table (first 8 rows)
    table_df = meta_df.head(8).copy()
    
    # Format for display
    display_df = table_df[['group_name', 'condition_id', 'sequence_no', 'ele_name', 
                            'wavelength_nm', 'att_value', 'channel_id']].copy()
    display_df.columns = ['Group', 'Cond', 'Seq', 'Element', 'λ (nm)', 'Att', 'Channel ID']
    
    # Round wavelength
    display_df['λ (nm)'] = display_df['λ (nm)'].round(3)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=display_df.values,
                     colLabels=display_df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Header styling
    for i in range(len(display_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('#FFFFFF')
    
    plt.title('Metadata Preview - First 8 Rows', fontsize=16, weight='bold', pad=20)
    plt.savefig(path_png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved PNG table to: {path_png}")

# Main execution function
def main():
    """
    Main execution: load JSON, build metadata, validate, and save previews.
    """
    # Configuration
    json_path = r"D:\CementIndustryMlModel\attenuator_info.json"
    csv_path = "results/metadata_preview.csv"
    png_path = "results/metadata_table.png"
    
    # Execute pipeline
    json_data = load_json(json_path)
    meta_df = build_metadata(json_data)
    meta_df = validate_metadata(meta_df)
    save_metadata_preview(meta_df, csv_path, png_path)
    
    # Print summary
    print("\n" + "="*60)
    print("METADATA SUMMARY")
    print("="*60)
    print(f"Total records: {len(meta_df)}")
    print(f"Unique groups: {meta_df['group_id'].nunique()}")
    print(f"Unique conditions: {meta_df['condition_id'].nunique()}")
    print(f"Unique elements: {meta_df['ele_name'].nunique()}")
    print(f"Unique channels: {meta_df['channel_id'].nunique()}")
    print("\nFirst 5 rows:")
    print(meta_df.head())
    
    return meta_df

if __name__ == "__main__":
    meta_df = main()
