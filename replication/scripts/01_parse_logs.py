"""
Log Parsing Script using Drain Parser

This script parses raw log files into structured JSON format using the Drain algorithm.
Supports HDFS and BGL datasets with dataset-specific configurations.

Reference:
    He et al. (2017). Drain: An Online Log Parsing Approach with Fixed Depth Tree. ICWS 2017.
"""

import sys
import os
import json
import re
from pathlib import Path

# Add logparser to path if installed via git
try:
    from logparser import Drain
    LOGPARSER_AVAILABLE = True
except ImportError:
    LOGPARSER_AVAILABLE = False
    # Don't exit - we can still copy sample data


# Dataset configurations
DATASET_CONFIGS = {
    "HDFS": {
        "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "regex": [
            r'blk_-?\d+',  # block id
            r'(\d+\.){3}\d+(:\d+)?'  # IP
        ],
        "st": 0.5,  # Similarity threshold
        "depth": 4  # Depth of all leaf nodes
    },

    "BGL": {
        "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        # Refined regex patterns from material/README.md
        "regex": [
            r'core\.\d+',
            r'(?<=:)(\ [A-Z][+-]?)+(?![a-z])',  # match X+ A C Y+......
            r'(?<=r)\d{1,2}',
            r'(?<=fpr)\d{1,2}',
            r'(0x)?[0-9a-fA-F]{8}',
            r'(?<=\.\.)0[xX][0-9a-fA-F]+',
            r'(?<=\.\.)\d+(?!x)',
            r'\d+(?=:)',
            r'^\d+$',  # only numbers
            r'(?<=\=)\d+(?!x)',
            r'(?<=\=)0[xX][0-9a-fA-F]+'  # for hexadecimal
        ],
        "st": 0.5,
        "depth": 4
    }
}


def parse_logs_drain(dataset_name, input_file, output_dir, config=None):
    """
    Parse log file using Drain algorithm

    Args:
        dataset_name: Name of the dataset (HDFS, BGL, etc.)
        input_file: Path to raw log file
        output_dir: Directory to save parsed results
        config: Optional custom configuration (uses defaults if None)

    Returns:
        Path to output structured JSON file
    """

    if config is None:
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
        config = DATASET_CONFIGS[dataset_name]

    print(f"\n{'='*60}")
    print(f"Parsing {dataset_name} logs with Drain")
    print(f"{'='*60}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Configuration:")
    print(f"  Log format: {config['log_format']}")
    print(f"  Similarity threshold: {config['st']}")
    print(f"  Depth: {config['depth']}")
    print(f"  Regex patterns: {len(config['regex'])} patterns")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Drain parser
    parser = Drain.LogParser(
        log_format=config['log_format'],
        indir=str(Path(input_file).parent),
        outdir=output_dir,
        depth=config['depth'],
        st=config['st'],
        rex=config['regex']
    )

    # Parse logs
    log_filename = Path(input_file).name
    print(f"\nParsing file: {log_filename}")
    parser.parse(log_filename)

    # Output files
    structured_file = Path(output_dir) / f"{log_filename}_structured.csv"
    template_file = Path(output_dir) / f"{log_filename}_templates.csv"

    print(f"\nParsing completed!")
    print(f"Structured output: {structured_file}")
    print(f"Templates output: {template_file}")

    # Convert CSV to JSON for easier processing
    json_output = convert_csv_to_json(structured_file)

    return json_output


def convert_csv_to_json(csv_file):
    """
    Convert Drain CSV output to JSON format

    Args:
        csv_file: Path to structured CSV file

    Returns:
        Path to JSON output file
    """
    import csv

    json_file = str(csv_file).replace('_structured.csv', '_structured.json')

    print(f"\nConverting to JSON format...")
    print(f"Input: {csv_file}")
    print(f"Output: {json_file}")

    # Read CSV and convert to JSON
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # Write JSON
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Converted {len(data)} log entries to JSON")

    return json_file


def preprocess_logs(input_file, output_file=None, dataset_name=None):
    """
    Pre-processing step before parsing (following Qin et al. approach)

    This can include:
    - Removing duplicate entries
    - Normalizing timestamps
    - Filtering irrelevant logs

    Args:
        input_file: Path to raw log file
        output_file: Path to save preprocessed logs (default: input_file + ".preprocessed")
        dataset_name: Name of dataset for dataset-specific preprocessing

    Returns:
        Path to preprocessed log file
    """
    if output_file is None:
        output_file = str(input_file) + ".preprocessed"

    print(f"\n{'='*60}")
    print(f"Pre-processing logs")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    # Simple preprocessing: remove empty lines and duplicates
    seen_lines = set()
    processed_count = 0

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            line = line.strip()
            if line and line not in seen_lines:
                fout.write(line + '\n')
                seen_lines.add(line)
                processed_count += 1

    print(f"Processed {processed_count} unique log entries")

    return output_file


def main():
    """Main function for log parsing"""

    # Default paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Parse log files using Drain')
    parser.add_argument('dataset', nargs='?', default='HDFS',
                       help='Dataset name (HDFS, BGL, etc.)')
    parser.add_argument('input_file', nargs='?',
                       help='Path to raw log file')
    parser.add_argument('--use-qin-preprocessing', action='store_true',
                       help='Apply Qin et al. preprocessing before parsing')
    parser.add_argument('--skip-sample-copy', action='store_true',
                       help='Skip copying sample data')

    args = parser.parse_args()

    # Example usage for HDFS
    print("="*60)
    print("Log Parsing Pipeline")
    print("="*60)

    # Check if we should use the sample data from material
    material_sample = project_root.parent / "material" / "samples" / "HDFS_100k.log_structured.json"

    if material_sample.exists() and not args.skip_sample_copy and not args.input_file:
        print(f"\nFound existing HDFS sample data: {material_sample}")
        print("This data is already parsed. Copying to replication directory...")

        # Copy to replication data directory
        dest_dir = project_root / "data" / "parsed"
        dest_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        dest_file = dest_dir / "HDFS_100k.log_structured.json"
        shutil.copy(material_sample, dest_file)

        print(f"✓ Copied parsed data to: {dest_file}")

        # Also copy the split data
        material_split = project_root.parent / "material" / "samples" / "HDFS_100k.log_0.7_splitted.npz"
        if material_split.exists():
            split_dir = project_root / "data" / "split"
            split_dir.mkdir(parents=True, exist_ok=True)
            dest_split = split_dir / "HDFS_100k.log_0.7_splitted.npz"
            shutil.copy(material_split, dest_split)
            print(f"✓ Copied split data to: {dest_split}")

        print("\n" + "="*60)
        print("✓ Data preparation completed successfully!")
        print("="*60)
        print("\nSample data is ready for use:")
        print(f"  - Parsed: {dest_file}")
        if material_split.exists():
            print(f"  - Split: {dest_split}")

        return 0  # Success

    # Parse actual log files
    elif args.input_file:
        dataset_name = args.dataset
        input_file = Path(args.input_file)

        if not input_file.exists():
            print(f"❌ Error: Input file not found: {input_file}")
            return 1

        # Apply Qin et al. preprocessing if requested
        if args.use_qin_preprocessing:
            print("\nApplying Qin et al. preprocessing...")
            try:
                from preprocessing_qin import preprocess_with_qin_approach

                preprocessed_file = input_file.parent / f"{input_file.stem}_qin_preprocessed{input_file.suffix}"
                preprocess_with_qin_approach(
                    str(input_file),
                    str(preprocessed_file),
                    dataset_name=dataset_name
                )
                # Use preprocessed file for parsing
                input_file = preprocessed_file
                print(f"✓ Preprocessing completed: {preprocessed_file}")

            except ImportError:
                print("⚠️  Warning: Could not import Qin preprocessor. Skipping preprocessing.")

        # Parse logs
        output_dir = project_root / "data" / "parsed"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            parse_logs_drain(
                dataset_name=dataset_name,
                input_file=str(input_file),
                output_dir=str(output_dir)
            )
            print("\n✓ Parsing completed successfully!")
            return 0

        except Exception as e:
            print(f"\n❌ Error during parsing: {e}")
            import traceback
            traceback.print_exc()
            return 1

    else:
        print("\nNo sample data found. To parse new logs:")
        print("1. Download raw log files using scripts/00_download_datasets.py")
        print("2. Run this script with the raw log file path")
        print("\nExample:")
        print("  python 01_parse_logs.py HDFS data/raw/HDFS/HDFS.log")

        print("\n" + "="*60)
        print("For BGL dataset, download raw logs and run:")
        print("  python 01_parse_logs.py BGL data/raw/BGL/BGL.log")
        print("="*60)

        return 1  # Failure


if __name__ == "__main__":
    if len(sys.argv) > 2:
        # Command line usage: python 01_parse_logs.py <dataset> <input_file>
        dataset = sys.argv[1]
        input_file = sys.argv[2]

        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_dir = project_root / "data" / "parsed"

        # Optional: preprocess
        # preprocessed = preprocess_logs(input_file, dataset_name=dataset)
        # input_file = preprocessed

        # Parse logs
        json_output = parse_logs_drain(dataset, input_file, output_dir)
        print(f"\nSuccess! Structured logs saved to: {json_output}")
    else:
        # Run main demo/copy
        main()
