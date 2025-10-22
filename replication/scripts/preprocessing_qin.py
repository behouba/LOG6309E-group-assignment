import re
from typing import List, Tuple, Dict
from pathlib import Path


class QinPreprocessor:
    def __init__(self, dataset_name: str = "generic"):
        self.dataset_name = dataset_name
        self.stats = {
            'total_lines': 0,
            'preprocessed_lines': 0,
            'duplicates_removed': 0,
            'variables_masked': {}
        }

        # General variable patterns (applicable to most log datasets)
        self.general_patterns = self._get_general_patterns()

        # Dataset-specific patterns
        self.dataset_patterns = self._get_dataset_patterns(dataset_name)

    def _get_general_patterns(self) -> List[Tuple[str, str, str]]:
        patterns = [
            # IP addresses (IPv4 and IPv6)
            ('ipv4', r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b(?::[0-9]+)?', '<IP>'),
            ('ipv6', r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b', '<IP>'),

            # Hexadecimal values (addresses, IDs)
            ('hex_long', r'\b0x[0-9a-fA-F]{8,}\b', '<HEX>'),
            ('hex_short', r'\b0x[0-9a-fA-F]+\b', '<HEX>'),

            # Numbers (integers and floats)
            ('float', r'\b\d+\.\d+(?:e[+-]?\d+)?\b', '<NUM>'),
            ('integer', r'\b\d{4,}\b', '<NUM>'),  # Long numbers (4+ digits)

            # File paths (Unix and Windows)
            ('unix_path', r'/(?:[a-zA-Z0-9_\-\.]+/)+[a-zA-Z0-9_\-\.]*', '<PATH>'),
            ('windows_path', r'[A-Z]:\\(?:[a-zA-Z0-9_\-\.]+\\)+[a-zA-Z0-9_\-\.]*', '<PATH>'),

            # URLs and URIs
            ('url', r'https?://[^\s]+', '<URL>'),
            ('uri', r'[a-z]+://[^\s]+', '<URI>'),

            # UUIDs and GUIDs
            ('uuid', r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '<UUID>'),

            # MAC addresses
            ('mac', r'\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b', '<MAC>'),

            # Timestamps (various formats)
            ('timestamp_unix', r'\b\d{10,13}\b', '<TIMESTAMP>'),  # Unix timestamp
            ('timestamp_iso', r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?', '<TIMESTAMP>'),

            # Memory addresses and sizes
            ('mem_addr', r'\b0x[0-9a-fA-F]{8,16}\b', '<ADDR>'),
            ('mem_size', r'\b\d+(?:KB|MB|GB|TB|B)\b', '<SIZE>'),

            # Durations and time intervals
            ('duration_ms', r'\b\d+ms\b', '<DURATION>'),
            ('duration_s', r'\b\d+s\b', '<DURATION>'),

            # Version numbers
            ('version', r'\bv?\d+\.\d+(?:\.\d+)*(?:-[a-zA-Z0-9]+)?\b', '<VERSION>'),
        ]

        return patterns

    def _get_dataset_patterns(self, dataset_name: str) -> List[Tuple[str, str, str]]:
        if dataset_name.upper() == "HDFS":
            return [
                ('block_id', r'blk_-?\d+', '<BLOCK>'),
                ('datanode', r'/\d{2,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+', '<DATANODE>'),
            ]

        elif dataset_name.upper() == "BGL":
            return [
                ('core_id', r'core\.\d+', '<CORE>'),
                ('node_id', r'(?<=:)(\ [A-Z][+-]?)+(?![a-z])', '<NODE>'),
                ('register', r'(?<=r)\d{1,2}', '<REG>'),
                ('fpr', r'(?<=fpr)\d{1,2}', '<FPR>'),
            ]

        elif dataset_name.upper() in ["SPIRIT", "THUNDERBIRD"]:
            return [
                ('process_id', r'\[\d+\]', '<PID>'),
                ('kernel_addr', r'\[<[0-9a-f]+>\]', '<KADDR>'),
            ]

        else:
            return []

    def preprocess_line(self, line: str) -> Tuple[str, int]:
        original_line = line
        variables_masked = 0

        # Strip whitespace
        line = line.strip()

        # Skip empty lines
        if not line:
            return "", 0

        # Apply dataset-specific patterns first (more specific)
        for pattern_name, regex, placeholder in self.dataset_patterns:
            matches = len(re.findall(regex, line))
            if matches > 0:
                line = re.sub(regex, placeholder, line)
                variables_masked += matches
                self.stats['variables_masked'][pattern_name] = \
                    self.stats['variables_masked'].get(pattern_name, 0) + matches

        # Apply general patterns
        for pattern_name, regex, placeholder in self.general_patterns:
            matches = len(re.findall(regex, line))
            if matches > 0:
                line = re.sub(regex, placeholder, line)
                variables_masked += matches
                self.stats['variables_masked'][pattern_name] = \
                    self.stats['variables_masked'].get(pattern_name, 0) + matches

        # Additional cleaning
        line = self._clean_line(line)

        return line, variables_masked

    def _clean_line(self, line: str) -> str:
        # Remove extra whitespace
        line = re.sub(r'\s+', ' ', line)

        # Remove leading/trailing whitespace
        line = line.strip()

        return line

    def preprocess_file(self, input_file: Path, output_file: Path,
                        remove_duplicates: bool = True) -> Dict:











        print(f"\n{'='*70}")
        print(f"Qin et al. Preprocessing Pipeline")
        print(f"{'='*70}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Input:   {input_file}")
        print(f"Output:  {output_file}")
        print(f"Remove duplicates: {remove_duplicates}")

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Process file
        seen_lines = set() if remove_duplicates else None
        total_variables_masked = 0

        with open(input_file, 'r', encoding='utf-8', errors='ignore') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:

            for line_num, line in enumerate(fin, 1):
                self.stats['total_lines'] += 1

                # Preprocess line
                preprocessed, num_masked = self.preprocess_line(line)
                total_variables_masked += num_masked

                # Skip empty lines
                if not preprocessed:
                    continue

                # Check for duplicates
                if remove_duplicates:
                    if preprocessed in seen_lines:
                        self.stats['duplicates_removed'] += 1
                        continue
                    seen_lines.add(preprocessed)

                # Write preprocessed line
                fout.write(preprocessed + '\n')
                self.stats['preprocessed_lines'] += 1

                # Progress indicator
                if line_num % 100000 == 0:
                    print(f"  Processed {line_num:,} lines...")

        # Print summary
        print(f"\n{'='*70}")
        print(f"Preprocessing Summary")
        print(f"{'='*70}")
        print(f"Total lines:           {self.stats['total_lines']:,}")
        print(f"Preprocessed lines:    {self.stats['preprocessed_lines']:,}")
        print(f"Duplicates removed:    {self.stats['duplicates_removed']:,}")
        print(f"Variables masked:      {total_variables_masked:,}")

        if self.stats['variables_masked']:
            print(f"\nVariable types masked:")
            for var_type, count in sorted(self.stats['variables_masked'].items(),
                                         key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {var_type:20s}: {count:,}")

        print(f"{'='*70}")

        return self.stats


def preprocess_with_qin_approach(input_file: str, output_file: str = None,
                                 dataset_name: str = "generic",
                                 remove_duplicates: bool = True) -> Dict:












    input_path = Path(input_file)

    if output_file is None:
        output_path = input_path.parent / f"{input_path.stem}_preprocessed{input_path.suffix}"
    else:
        output_path = Path(output_file)

    # Create preprocessor
    preprocessor = QinPreprocessor(dataset_name=dataset_name)

    # Preprocess file
    stats = preprocessor.preprocess_file(input_path, output_path, remove_duplicates)

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing_qin.py <input_file> [output_file] [dataset_name]")
        print("\nExample:")
        print("  python preprocessing_qin.py data/raw/HDFS/HDFS.log data/preprocessed/HDFS_preprocessed.log HDFS")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else "generic"

    stats = preprocess_with_qin_approach(input_file, output_file, dataset_name)

    print("\nPreprocessing completed successfully.")
    print(f"  Output saved to: {output_file or Path(input_file).parent / f'{Path(input_file).stem}_preprocessed{Path(input_file).suffix}'}")
