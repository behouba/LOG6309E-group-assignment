import sys
import os
import json
import re
from pathlib import Path

try:
    from logparser import Drain
    LOGPARSER_AVAILABLE = True
except ImportError:
    LOGPARSER_AVAILABLE = False


DATASET_CONFIGS = {
    "HDFS": {
        "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "regex": [
            r'blk_-?\d+',
            r'(\d+\.){3}\d+(:\d+)?'
        ],
        "st": 0.5,
        "depth": 4
    },

    "BGL": {
        "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        "regex": [
            r'core\.\d+',
            r'(?<=:)(\ [A-Z][+-]?)+(?![a-z])',
            r'(?<=r)\d{1,2}',
            r'(?<=fpr)\d{1,2}',
            r'(0x)?[0-9a-fA-F]{8}',
            r'(?<=\.\.)0[xX][0-9a-fA-F]+',
            r'(?<=\.\.)\d+(?!x)',
            r'\d+(?=:)',
            r'^\d+$',
            r'(?<=\=)\d+(?!x)',
            r'(?<=\=)0[xX][0-9a-fA-F]+'
        ],
        "st": 0.5,
        "depth": 4
    }
}


def parse_logs_drain(dataset_name, input_file, output_dir, config=None):
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

    os.makedirs(output_dir, exist_ok=True)
    parser = Drain.LogParser(
        log_format=config['log_format'],
        indir=str(Path(input_file).parent),
        outdir=output_dir,
        depth=config['depth'],
        st=config['st'],
        rex=config['regex']
    )

    log_filename = Path(input_file).name
    print(f"\nParsing file: {log_filename}")
    parser.parse(log_filename)

    structured_file = Path(output_dir) / f"{log_filename}_structured.csv"
    template_file = Path(output_dir) / f"{log_filename}_templates.csv"

    print(f"\nParsing completed!")
    print(f"Structured output: {structured_file}")
    print(f"Templates output: {template_file}")

    json_output = convert_csv_to_json(structured_file)

    return json_output


def convert_csv_to_json(csv_file):
    import csv

    json_file = str(csv_file).replace('_structured.csv', '_structured.json')

    print(f"\nConverting to JSON format...")
    print(f"Input: {csv_file}")
    print(f"Output: {json_file}")

    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Converted {len(data)} log entries to JSON")

    return json_file


def preprocess_logs(input_file, output_file=None, dataset_name=None):
    if output_file is None:
        output_file = str(input_file) + ".preprocessed"

    print(f"\n{'='*60}")
    print(f"Pre-processing logs")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

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
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
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

    print("="*60)
    print("Log Parsing Pipeline")
    print("="*60)

    material_sample = project_root.parent / "material" / "samples" / "HDFS_100k.log_structured.json"

    if material_sample.exists() and not args.skip_sample_copy and not args.input_file:
        print(f"\nFound existing HDFS sample data: {material_sample}")
        print("This data is already parsed. Copying to replication directory...")

        dest_dir = project_root / "data" / "parsed"
        dest_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        dest_file = dest_dir / "HDFS_100k.log_structured.json"
        shutil.copy(material_sample, dest_file)

        print(f"Copied parsed data to: {dest_file}")

        material_split = project_root.parent / "material" / "samples" / "HDFS_100k.log_0.7_splitted.npz"
        if material_split.exists():
            split_dir = project_root / "data" / "split"
            split_dir.mkdir(parents=True, exist_ok=True)
            dest_split = split_dir / "HDFS_100k.log_0.7_splitted.npz"
            shutil.copy(material_split, dest_split)
            print(f"Copied split data to: {dest_split}")

        print("\n" + "="*60)
        print("Data preparation completed successfully.")
        print("="*60)
        print("\nSample data is ready for use:")
        print(f"  - Parsed: {dest_file}")
        if material_split.exists():
            print(f"  - Split: {dest_split}")

        return 0

    elif args.input_file:
        dataset_name = args.dataset
        input_file = Path(args.input_file)

        if not input_file.exists():
            print(f"❌ Error: Input file not found: {input_file}")
            return 1

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
                input_file = preprocessed_file
                print(f"Preprocessing completed: {preprocessed_file}")

            except ImportError:
                print("Warning: Could not import Qin preprocessor. Skipping preprocessing.")

        output_dir = project_root / "data" / "parsed"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            parse_logs_drain(
                dataset_name=dataset_name,
                input_file=str(input_file),
                output_dir=str(output_dir)
            )
            print("\nParsing completed successfully.")
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

        return 1


if __name__ == "__main__":
    if len(sys.argv) > 2:
        dataset = sys.argv[1]
        input_file = sys.argv[2]

        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_dir = project_root / "data" / "parsed"

        json_output = parse_logs_drain(dataset, input_file, output_dir)
        print(f"\nSuccess! Structured logs saved to: {json_output}")
    else:
        main()
